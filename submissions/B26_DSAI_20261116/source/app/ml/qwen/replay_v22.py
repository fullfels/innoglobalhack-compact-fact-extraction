"""Ablate saved complete generations. No inference, gold-derived rules, or historic edits."""
import collections,importlib.util,json
from pathlib import Path
from ml.qwen.final_bridge import process_output
from src.contracts.models import CallInput
from src.pipeline.adapters import InjectedAdapter,ModelResources,PipelineError
from src.pipeline.service import Pipeline

def load_evaluator(path):
    spec=importlib.util.spec_from_file_location('official_v21_for_v22',path)
    ev=importlib.util.module_from_spec(spec);spec.loader.exec_module(ev)
    ev.load_catalog(str(Path(path).with_name('catalog.json')))
    return ev

def predictions(result,call):
    normalized={'call_id':call.call_id,'facts':[c.fact.model_dump(mode='json') for c in result.batch.candidates]}
    adapter=InjectedAdapter(ModelResources('',None),lambda c,res:result.batch)
    served,warnings=Pipeline(adapter).run(call)
    return normalized,{'call_id':call.call_id,'facts':[f.model_dump(mode='json') for f in served.facts]},list(warnings)

def replay(run,root,out):
    out.mkdir(parents=True,exist_ok=True)
    gold={r['call_id']:r for r in map(json.loads,(root/'data/dev32_gold_official.jsonl').read_text(encoding='utf-8').splitlines())}
    ev=load_evaluator(root/'evaluation/case_metrics.py');summaries={}
    for checkpoint in ['selection_update250','final_update000500']:
        for calendar in [False,True]:
            name=checkpoint+('_calendar' if calendar else '_partial');folder=out/name;folder.mkdir(exist_ok=True)
            records=[];norm={};serving={}
            for path in sorted((run/checkpoint/'records').glob('*.json')):
                old=json.loads(path.read_text(encoding='utf-8'));cid=old['call_id'];call=CallInput.model_validate({k:gold[cid][k] for k in ['call_id','started_at','turns']})
                row={'call_id':cid,'original_status':old['status'],'status':'failed','facts':None,'serving_facts':None,'error_code':None}
                try:
                    if old['error_code'] in ['QWEN_GENERATION_TIMEOUT','QWEN_OUTPUT_LIMIT','QWEN_OUT_OF_MEMORY']:raise PipelineError(old['error_code'])
                    result=process_output(old['raw'],call,calendar=calendar)
                    n,s,w=predictions(result,call);norm[cid]=n;serving[cid]=s
                    row.update(status=result.status,facts=n['facts'],serving_facts=s['facts'],rejected=result.rejected,changes=result.changes,warnings=w)
                except PipelineError as exc:
                    row.update(error_code=exc.code,rejected=getattr(exc,'rejected',[]));norm[cid]={'call_id':cid,'facts':[]};serving[cid]={'call_id':cid,'facts':[]}
                records.append(row)
            assert set(norm)==set(gold) and len(records)==len(gold)
            summaries[name]={'scope':'Counterfactual replay on previously used inner-dev32; no new model generation or official-val383 evaluation.',
                'calendar':calendar,'normalized':ev.evaluate(gold,norm),'serving':ev.evaluate(gold,serving),
                'statuses':dict(collections.Counter(r['status'] for r in records)),
                'changed_fields':sum(len(r.get('changes',[])) for r in records),'rejected_facts':sum(len(r.get('rejected',[])) for r in records)}
            for view,preds in [('normalized',norm),('serving',serving)]:
                (folder/(view+'_predictions.jsonl')).write_text(''.join(json.dumps(v,ensure_ascii=False)+'\n' for v in preds.values()),encoding='utf-8')
            (folder/'records.jsonl').write_text(''.join(json.dumps(v,ensure_ascii=False)+'\n' for v in records),encoding='utf-8')
    (out/'comparison.json').write_text(json.dumps(summaries,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return summaries

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    print(json.dumps(replay(a.run,a.root,a.out),ensure_ascii=False,indent=2))
