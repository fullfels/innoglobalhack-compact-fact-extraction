import re
from datetime import datetime, timedelta
from src.contracts.models import CandidateBatch, CallInput, Fact, Source


class CatalogNormalizer:
    """Нормализатор каталога с поиском по синонимам и алиасам"""
    def __init__(self, catalog_data: dict = None):
        self.catalog = catalog_data or {}
        self.index = self._build_index()

    def _build_index(self) -> dict:
        index = {}
        for category, items in self.catalog.items():
            for item in items:
                canonical_name = item["name"]
                variants = [canonical_name.lower()] + [alias.lower() for alias in item.get("aliases", [])]
                for variant in variants:
                    index[variant] = canonical_name
        return index

    def normalize(self, batch: CandidateBatch, call: CallInput) -> CandidateBatch:
        updated_candidates = []
        for candidate in batch.candidates:
            fact = candidate.fact
            if fact.object and getattr(fact.object, "name", None):
                raw_name = fact.object.name.lower().strip()
                raw_name_clean = re.sub(r'[^\w\s]', '', raw_name)
                
                matched_canonical = None
                if raw_name_clean in self.index:
                    matched_canonical = self.index[raw_name_clean]
                else:
                    for variant, canonical in self.index.items():
                        if variant in raw_name_clean or raw_name_clean in variant:
                            matched_canonical = canonical
                            break
                
                if matched_canonical:
                    new_obj = fact.object.model_copy(update={"name": matched_canonical})
                    fact = fact.model_copy(update={"object": new_obj})

            new_candidate = candidate.model_copy(update={"fact": fact})
            updated_candidates.append(new_candidate)

        return batch.model_copy(update={"candidates": tuple(updated_candidates)})


class TemporalNormalizer:
    """Нормализация относительных дат относительно started_at звонка"""
    def normalize(self, batch: CandidateBatch, call: CallInput) -> CandidateBatch:
        started_at = call.started_at
        if not started_at:
            return batch

        if isinstance(started_at, datetime):
            base_date = started_at
        elif isinstance(started_at, str):
            try:
                base_date = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
            except ValueError:
                return batch
        else:
            try:
                base_date = datetime.fromisoformat(str(started_at))
            except Exception:
                return batch

        updated_candidates = []
        for candidate in batch.candidates:
            fact = candidate.fact
            fact_updates = {}
            
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
                        fact_updates[field_name] = target_date.date().isoformat()

            if fact_updates:
                new_fact = fact.model_copy(update=fact_updates)
                new_candidate = candidate.model_copy(update={"fact": new_fact})
                updated_candidates.append(new_candidate)
            else:
                updated_candidates.append(candidate)

        return batch.model_copy(update={"candidates": tuple(updated_candidates)})


class StateAndGroundingNormalizer:
    """Проверка спанов, цитат и привязка к репликам"""
    def normalize(self, batch: CandidateBatch, call: CallInput) -> CandidateBatch:
        turns_map = {t.idx: t.text for t in call.turns}

        updated_candidates = []
        for candidate in batch.candidates:
            fact = candidate.fact
            source = fact.source
            if source and source.turn_idx in turns_map:
                turn_text = turns_map[source.turn_idx]
                if source.quote and source.quote in turn_text:
                    start = turn_text.index(source.quote)
                    end = start + len(source.quote)
                    new_source = source.model_copy(update={"span": (start, end)})
                    new_fact = fact.model_copy(update={"source": new_source})
                    new_candidate = candidate.model_copy(update={"fact": new_fact})
                    updated_candidates.append(new_candidate)
                    continue
            updated_candidates.append(candidate)

        return batch.model_copy(update={"candidates": tuple(updated_candidates)})