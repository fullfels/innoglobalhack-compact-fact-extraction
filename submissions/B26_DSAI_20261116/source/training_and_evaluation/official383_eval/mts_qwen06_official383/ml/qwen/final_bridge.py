"""Partial validation using the existing shared CandidateBatch; rejected facts stay visible."""
from dataclasses import dataclass
import json
from ml.qwen.bridge import strict_json,parse_candidates,model_input
from ml.qwen.calendar_v23 import normalize_fact,explicit_acceptance,linked_interval_updates
from src.contracts.models import CandidateBatch,CandidateFact,Fact
from src.pipeline.adapters import PipelineError

@dataclass(frozen=True)
class Processed:
    batch:CandidateBatch
    status:str
    rejected:tuple
    changes:tuple
    generated_count:int

def confirmation(candidate,call):
    fact=candidate.fact.model_dump(mode='json')
    proof=explicit_acceptance(fact,model_input(call))
    if not proof:return candidate
    return CandidateFact(**{**candidate.model_dump(),'confirmation':'confirmed',
        'evidence_turns':tuple(sorted({candidate.fact.source.turn_idx,proof['proposal_turn_idx'],proof['acceptance_turn_idx']})),
        'provenance':candidate.provenance+('v22-unique-proposal-explicit-client-acceptance',)})

def process_output(raw,call,*,calendar=True):
    try:value=strict_json(raw)
    except (ValueError,TypeError):raise PipelineError('QWEN_INVALID_JSON',503,'qwen.parse') from None
    if not isinstance(value,dict) or set(value)!={'facts'} or not isinstance(value['facts'],list):
        raise PipelineError('QWEN_INVALID_OUTPUT_SCHEMA',503,'qwen.parse')
    accepted=[];indexed=[];rejected=[];changes=[];input_call=model_input(call)
    for i,original in enumerate(value['facts']):
        row=original;modifications=[]
        if calendar and isinstance(original,dict):row,modifications=normalize_fact(original,input_call)
        try:
            batch=parse_candidates(json.dumps({'facts':[row]},ensure_ascii=False,allow_nan=False),call,confirmation_resolver=confirmation)
            if modifications:
                proof=modifications[0].get('proof',{})
                if proof.get('clock_note'):
                    batch=CandidateBatch(candidates=tuple(CandidateFact(**{**c.model_dump(),
                        'temporal_expression':proof.get('source_clause'),'warnings':c.warnings+('Дата определена до дня; точный час не указан.',)}) for c in batch.candidates))
            accepted.extend(batch.candidates)
            indexed.extend((i,c) for c in batch.candidates)
            changes.extend({'fact_index':i,**m} for m in modifications)
        except PipelineError as exc:
            rejected.append({'fact_index':i,'error_code':exc.code,'module':exc.module})
    if value['facts'] and not accepted:
        error=PipelineError('QWEN_ALL_FACTS_REJECTED',503,'qwen.parse')
        error.rejected=tuple(rejected)
        raise error
    if calendar:
        updates=linked_interval_updates(indexed,value['facts'],changes)
        mapping={u['fact_index']:u for u in updates}
        accepted=[]
        for index,c in indexed:
            if index in mapping:
                update=mapping[index];data=c.fact.model_dump(mode='json');data['valid_to']=update['after']
                c=CandidateFact(**{**c.model_dump(),'fact':Fact.model_validate(data),'provenance':c.provenance+('v22-preserved-existing-boundary',)})
            accepted.append(c)
        changes.extend(updates)
    status='partial' if rejected else 'ok'
    warnings=tuple(f"Rejected generated fact {r['fact_index']}: {r['error_code']}" for r in rejected)
    batch=CandidateBatch(candidates=tuple(accepted),warnings=warnings)
    return Processed(batch,status,tuple(rejected),tuple(changes),len(value['facts']))

class FinalBridge:
    def __init__(self,engine,*,calendar=True):self.engine=engine;self.calendar=calendar;self.last_result=None
    def __call__(self,call,resources=None):
        self.last_result=None
        result=process_output(self.engine.generate(call),call,calendar=self.calendar)
        self.last_result=result
        return result.batch
