from typing import Tuple
from pydantic import ValidationError

from src.contracts.models import CallInput, CallPrediction, CandidateBatch
from src.pipeline.adapters import Extractor, Normalizer, PipelineError
from src.pipeline.normalizers import build_default_normalizers


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
        self.extractor = extractor
        self.normalizers = normalizers

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
            batch = stage(
                'normalizer.' + type(normalizer).__name__,
                lambda n=normalizer, b=batch: n.normalize(b, call),
            )

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