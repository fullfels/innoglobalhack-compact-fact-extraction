"""Official2.1 compatibility using the project's existing CandidateBatch, without role/index guessing."""
import json
from ml.qwen.bridge import parse_candidates,strict_json,model_input
from ml.qwen.temporal_v21 import normalize_generated,acceptance
from src.contracts.models import CandidateFact
from src.pipeline.adapters import PipelineError

def confirmation(candidate,call):
    proof=acceptance(candidate.fact.model_dump(mode='json'),model_input(call))
    if proof is None:return candidate
    return CandidateFact(**{**candidate.model_dump(),'confirmation':'confirmed',
        'evidence_turns':tuple(sorted({candidate.fact.source.turn_idx,proof['proposal_turn_idx'],proof['acceptance_turn_idx']})),
        'provenance':candidate.provenance+('unique-event-linked-explicit-client-acceptance',)})

def process_output(raw,call,*,normalize=True):
    changes=[];processed=raw
    if normalize:
        try:value=strict_json(raw)
        except (ValueError,TypeError):raise PipelineError('QWEN_INVALID_JSON',503,'qwen.parse') from None
        value,changes=normalize_generated(value,model_input(call))
        processed=json.dumps(value,ensure_ascii=False,separators=(',',':'),allow_nan=False)
    batch=parse_candidates(processed,call,confirmation_resolver=confirmation)
    return batch,changes,processed

class V21Bridge:
    def __init__(self,engine):self.engine=engine
    def __call__(self,call,resources=None):
        return process_output(self.engine.generate(call),call)[0]
