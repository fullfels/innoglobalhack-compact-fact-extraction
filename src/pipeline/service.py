from pydantic import ValidationError
from typing import Any, Tuple
from datetime import datetime, timedelta
from src.contracts.models import CallInput, CallPrediction, CandidateBatch
from src.pipeline.adapters import Extractor, Normalizer, PipelineError


class CatalogNormalizer:
    """Задача T12: Нормализация названий услуг, тарифов, устройств по каталогу"""
    def __init__(self, catalog_data: dict = None):
        self.catalog = catalog_data or {}

    def normalize(self, batch: CandidateBatch, call: CallInput) -> CandidateBatch:
        for candidate in batch.candidates:
            fact = candidate.fact
            if fact.object and hasattr(fact.object, "name") and fact.object.name:
                # Здесь можно подключить маппинг по каталогу
                pass
        return batch


class TemporalNormalizer:
    """Задача T13: Нормализация относительных дат относительно started_at звонка"""
    def normalize(self, batch: CandidateBatch, call: CallInput) -> CandidateBatch:
        started_at_str = call.started_at
        if not started_at_str:
            return batch

        try:
            base_date = datetime.fromisoformat(started_at_str.replace("Z", "+00:00"))
        except ValueError:
            return batch

        for candidate in batch.candidates:
            fact = candidate.fact
            for field_name in ["valid_from", "valid_to"]:
                val = getattr(fact, field_name, None)
                if val and isinstance(val, str):
                    val_lower = val.lower().strip()
                    target_date = None
                    if "сегодня" in val_lower:
                        target_date = base_date
                    elif "вчера" in val_lower:
                        target_date = base_date - timedelta(days=1)
                    elif "завтра" in val_lower:
                        target_date = base_date + timedelta(days=1)
                    elif "через неделю" in val_lower:
                        target_date = base_date + timedelta(days=7)
                        
                    if target_date:
                        object.__setattr__(fact, field_name, target_date.date().isoformat())
        return batch


class StateAndGroundingNormalizer:
    """Задача T15: Проверка спанов, цитат и привязка к репликам"""
    def normalize(self, batch: CandidateBatch, call: CallInput) -> CandidateBatch:
        turns_map = {t.idx: t.text for t in call.turns}

        for candidate in batch.candidates:
            fact = candidate.fact
            source = fact.source
            if source and source.turn_idx in turns_map:
                turn_text = turns_map[source.turn_idx]
                if source.quote and source.quote in turn_text:
                    start = turn_text.index(source.quote)
                    end = start + len(source.quote)
                    object.__setattr__(source, 'span', (start, end))
        return batch


def validate_output(call: CallInput, facts) -> CallPrediction:
    try:
        prediction = CallPrediction(**call.model_dump(), facts=facts)
    except ValidationError as exc:
        raise PipelineError('INVALID_OUTPUT', 503, 'output', type(exc).__name__) from None
    
    turns = {t.idx: t for t in call.turns}
    for fact in prediction.facts:
        source = fact.source
        turn = turns.get(source.turn_idx)
        if turn is None or turn.speaker != fact.speaker:
            raise PipelineError('INVALID_SOURCE', 503, 'output')
        if source.span is None:
            if source.quote is not None:
                raise PipelineError('QUOTE_WITHOUT_SPAN', 503, 'output')
        else:
            start, end = source.span
            if not 0 <= start < end <= len(turn.text) or source.quote != turn.text[start:end]:
                raise PipelineError('INVALID_SPAN', 503, 'output')
    return prediction


class Pipeline:
    def __init__(self, extractor: Extractor, normalizers: Tuple[Normalizer, ...] = ()):
        self.extractor, self.normalizers = extractor, normalizers

    def run(self, call: CallInput) -> Tuple[CallPrediction, Tuple[str, ...]]:
        def stage(module, fn):
            try:
                return CandidateBatch.model_validate(fn())
            except PipelineError:
                raise
            except Exception as exc:
                raise PipelineError('STAGE_FAILED', 503, module, type(exc).__name__) from None

        batch = stage('extractor', lambda: self.extractor.extract(call))
        
        for normalizer in self.normalizers:
            batch = stage('normalizer.' + type(normalizer).__name__, lambda: normalizer.normalize(batch, call))
            
        exported = []
        filtered = 0
        for candidate in batch.candidates:
            if candidate.confirmation == 'rejected':
                filtered += 1
                continue
            if candidate.fact.attribute == 'agreement':
                turns = {t.idx: t for t in call.turns}
                if candidate.confirmation != 'confirmed' or not any(
                    idx in turns and turns[idx].speaker == 'client' for idx in candidate.evidence_turns
                ):
                    filtered += 1
                    continue
            exported.append(candidate.fact)
            
        prediction = validate_output(call, tuple(exported))
        warnings = batch.warnings + tuple(w for c in batch.candidates for w in c.warnings)
        if filtered:
            warnings += (f'Filtered {filtered} rejected/unconfirmed candidates',)
        if not self.normalizers:
            warnings += ('DE normalizers not connected',)
        return prediction, warnings