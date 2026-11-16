"""Narrow official-catalog cleanup; never infer new facts, names, dates or sources."""
import re
from src.contracts.models import CandidateBatch

POLICY_ID = 'tariff-catalog-v1'

def normalized(text):
    return text.lower().replace('ё', 'е')

class TariffValueNormalizer:
    def __init__(self, catalog):
        entries=catalog.get('tariffs', [])
        self.patterns=[]
        for item in entries:
            name=item if isinstance(item,str) else item['name']
            self.patterns.append((name,re.compile(r'(?<![\w+])'+re.escape(normalized(name))+r'(?![\w+])')))

    def canonical(self, value):
        text=normalized(value)
        if re.search(r'\b(?:не|нет|вместо|кроме|или|либо|но|а)\b|→|->',text):
            return None
        matches=[(m.start(),m.end(),name) for name,p in self.patterns for m in p.finditer(text)]
        # A full nested name (e.g. Новый оптимальный) takes precedence over its substring.
        maximal=[(s,e,name) for s,e,name in matches if not any(
            a<=s and e<=b and (a<s or e<b) for a,b,_ in matches)]
        names={name for _,_,name in maximal}
        return next(iter(names)) if len(names)==1 else None

    def normalize(self, batch, call):
        candidates=[]
        for candidate in batch.candidates:
            fact=candidate.fact
            target=self.canonical(fact.value) if fact.attribute=='tariff' else None
            if target is not None and target!=fact.value:
                candidate=candidate.model_copy(update={
                    'fact':fact.model_copy(update={'value':target}),
                    'provenance':candidate.provenance+(POLICY_ID,),
                    'warnings':candidate.warnings+('Tariff value normalized to official catalog',)})
            candidates.append(candidate)
        return batch.model_copy(update={'candidates':tuple(candidates)})
