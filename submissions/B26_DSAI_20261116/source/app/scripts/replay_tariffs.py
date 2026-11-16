"""Apply the frozen tariff-only policy to saved serving predictions; no model calls."""
from pathlib import Path
import argparse,hashlib,json
from ml.qwen.postprocessing import TariffValueNormalizer,POLICY_ID
from ml.qwen.replay_v22 import load_evaluator
from src.contracts.models import CallInput,Fact,CandidateFact,CandidateBatch
from src.pipeline.service import validate_output

def rows(path):
    data=[json.loads(s) for s in path.read_text(encoding='utf-8').splitlines() if s.strip()]
    assert len({r['call_id'] for r in data})==len(data),'Duplicate call ID'
    return {r['call_id']:r for r in data}

def run(gold_path,pred_path,catalog_path,evaluator_path,out):
    gold,pred=rows(gold_path),rows(pred_path)
    assert set(pred)<=set(gold),'Foreign call IDs'
    normalizer=TariffValueNormalizer(json.loads(catalog_path.read_text(encoding='utf-8')))
    after={};changes=[]
    for cid,row in pred.items():
        c=gold[cid];call=CallInput.model_validate({k:c[k] for k in ['call_id','started_at','turns']})
        # These are already exported serving facts. Never re-filter agreements or invent failures.
        batch=CandidateBatch(candidates=tuple(CandidateFact(fact=Fact.model_validate(f)) for f in row['facts']))
        normalized=normalizer.normalize(batch,call)
        facts=[c.fact for c in normalized.candidates]
        validate_output(call,facts)
        after[cid]={'call_id':cid,'facts':[f.model_dump(mode='json') for f in facts]}
        for index,(a,b) in enumerate(zip(batch.candidates,normalized.candidates)):
            if a.fact!=b.fact:changes.append({'call_id':cid,'fact_index':index,'before':a.fact.value,'after':b.fact.value})
    ev=load_evaluator(evaluator_path)
    report={'policy':POLICY_ID,'complete_ids':set(gold)==set(pred),'n_gold_calls':len(gold),
        'n_predicted_calls':len(pred),'before':ev.evaluate(gold,pred),'after':ev.evaluate(gold,after),
        'changes':changes,'new_model_calls':0,
        'scope':'Quality replay of saved serving facts; consult original raw records for generation/schema failures',
        'sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [gold_path,pred_path,catalog_path,evaluator_path,Path(__file__).parents[1]/'ml/qwen/postprocessing.py']}}
    out.mkdir(parents=True,exist_ok=True)
    (out/'predictions.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in after.values()),encoding='utf-8')
    (out/'comparison.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--gold',type=Path,required=True)
    p.add_argument('--predictions',type=Path,required=True);p.add_argument('--catalog',type=Path,required=True)
    p.add_argument('--evaluator',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    result=run(a.gold,a.predictions,a.catalog,a.evaluator,a.out)
    print(json.dumps({k:result[k] for k in ['policy','complete_ids','before','after','new_model_calls']},ensure_ascii=False,indent=2))
