"""Shared evaluation/export utilities; controllers are train_temporal.py and validate_export.py."""
import argparse,collections,csv,gc,hashlib,importlib.metadata,json,os,shutil,statistics,sys,time,zipfile
from pathlib import Path
from datetime import datetime,timezone
from ml.qwen.bridge import model_input
from ml.qwen.final_bridge import process_output
from ml.qwen.final_runtime import Engine,load_model,sha,MODEL,REVISION
from ml.qwen.replay_v22 import load_evaluator,predictions
from src.contracts.models import CallInput
from src.pipeline.adapters import PipelineError

def save(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8');os.replace(tmp,path)

def fingerprint(obj):return hashlib.sha256(json.dumps(obj,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def rows(path):
    result=[json.loads(s) for s in Path(path).read_text(encoding='utf-8').splitlines() if s.strip()]
    assert len({r['call_id'] for r in result})==len(result)
    return result
def input_call(row):return CallInput.model_validate({k:row[k] for k in ['call_id','started_at','turns']})

def verify(root):
    m=json.loads((root/'package_manifest.json').read_text(encoding='utf-8'))
    for relative,expected in m['sha256'].items():
        p=(root/relative).resolve()
        if not p.is_relative_to(root.resolve()) or sha(p)!=expected:raise ValueError('Package hash mismatch: '+relative)
    return m


def warmup(model,tok):
    import torch
    x=tok('Проверка.',return_tensors='pt').to('cuda')
    with torch.inference_mode():model.generate(**x,do_sample=False,max_new_tokens=2,use_cache=True,pad_token_id=tok.pad_token_id)
    torch.cuda.synchronize()

def reload_test(model,tok,path,out):
    import torch
    x=tok('Проверка повторной загрузки адаптера.',return_tensors='pt').to('cuda')
    with torch.inference_mode():a=model(**x,use_cache=False).logits[:,-1,:].float().cpu()
    second,other=load_model(path/'adapter',path/'tokenizer',local_only=True)
    with torch.inference_mode():b=second(**x,use_cache=False).logits[:,-1,:].float().cpu()
    error=float((a-b).abs().max());ok=bool(torch.isfinite(a).all() and torch.isfinite(b).all() and error<=.001)
    save(out,{'passed':ok,'fresh_base_and_adapter':True,'max_logit_difference':error,'adapter_sha256':sha(path/'adapter/adapter_model.safetensors')})
    del second,other,a,b;gc.collect();torch.cuda.empty_cache()
    if not ok:raise ValueError('Fresh adapter reload test failed')

def process_record(raw,call,calendar):
    result=process_output(raw,call,calendar=calendar);n,s,w=predictions(result,call)
    return {'status':result.status,'facts':n['facts'],'serving_facts':s['facts'],'rejected':result.rejected,'changes':result.changes,'warnings':w,'error_code':None}

def evaluate(engine,calls,root,out,adapter_hash,precision,deadline,*,allow_calendar_choices=True,calendar_enabled=False):
    import torch
    out.mkdir(parents=True,exist_ok=True)
    meta={'base':MODEL,'revision':REVISION,'adapter_sha256':adapter_hash,'precision':precision,'prompt_sha256':sha(root/'prompt.txt'),
          'package_manifest_sha256':sha(root/'package_manifest.json'),'inputs':{c['call_id']:fingerprint(model_input(input_call(c))) for c in calls},
          'postprocess_views':['partial','calendar'] if allow_calendar_choices else ['calendar' if calendar_enabled else 'partial'],
          'max_new_tokens':2048,'max_call_seconds':120,'use_cache':True,'thinking':False,'sampling':False,'batch_size':1}
    mp=out/'generation_manifest.json'
    if mp.exists() and json.loads(mp.read_text())!=meta:raise ValueError('Refuse stale cached predictions')
    save(mp,meta);records=[];active=0.;torch.cuda.reset_peak_memory_stats()
    for i,c in enumerate(calls):
        call=input_call(c);p=out/'records'/(hashlib.sha256(call.call_id.encode()).hexdigest()[:20]+'.json')
        if p.exists():
            r=json.loads(p.read_text(encoding='utf-8'))
            if r['input_sha256']!=meta['inputs'][call.call_id]:raise ValueError('Cached call changed')
            records.append(r);continue
        if time.time()+130>deadline:break
        r={'call_id':call.call_id,'input_sha256':meta['inputs'][call.call_id],'raw':None,'generation_error':None,'views':{}}
        torch.cuda.synchronize();start=time.perf_counter()
        try:r['raw']=engine.generate(call)
        except PipelineError as exc:r.update(generation_error=exc.code,raw=engine.last_raw)
        except Exception as exc:r.update(generation_error='UNEXPECTED_'+type(exc).__name__,raw=engine.last_raw)
        torch.cuda.synchronize();r['generation_seconds']=time.perf_counter()-start;active+=r['generation_seconds'];r['generation']=engine.last_diagnostics
        r['peak_cuda_allocated_bytes']=torch.cuda.max_memory_allocated();r['peak_cuda_reserved_bytes']=torch.cuda.max_memory_reserved()
        for calendar in ([False,True] if allow_calendar_choices else [calendar_enabled]):
            view='calendar' if calendar else 'partial'
            try:
                if r['generation_error']:raise PipelineError(r['generation_error'])
                r['views'][view]=process_record(r['raw'],call,calendar)
            except PipelineError as exc:r['views'][view]={'status':'failed','facts':None,'serving_facts':None,'error_code':exc.code,'rejected':getattr(exc,'rejected',[])}
            except Exception as exc:r['views'][view]={'status':'failed','facts':None,'serving_facts':None,'error_code':'UNEXPECTED_'+type(exc).__name__}
        save(p,r);records.append(r)
        measured=[x['generation_seconds'] for x in records];estimate=statistics.mean(measured)*(len(calls)-len(records))/60
        print(f"{out.name}: {len(records)}/{len(calls)} {call.call_id} sec={r['generation_seconds']:.2f} remaining_est_min={estimate:.1f}",flush=True)
        save(out/'progress.json',{'completed':len(records),'planned':len(calls),'estimated_generation_remaining_minutes':estimate,'deadline_utc':datetime.fromtimestamp(deadline,timezone.utc).isoformat()})
    ids={r['call_id'] for r in records};gold={c['call_id']:c for c in calls if c['call_id'] in ids};ev=load_evaluator(root/'evaluation/case_metrics.py')
    metrics={};metric_errors={};metrics_by_slice={}
    for calendar in ([False,True] if allow_calendar_choices else [calendar_enabled]):
        key='calendar' if calendar else 'partial'
        for serving in [False,True]:
            label=key+('_serving' if serving else '_normalized');field='serving_facts' if serving else 'facts'
            pred={r['call_id']:{'call_id':r['call_id'],'facts':r['views'][key][field] or []} for r in records}
            (out/(label+'_predictions.jsonl')).write_text(''.join(json.dumps(p,ensure_ascii=False)+'\n' for p in pred.values()),encoding='utf-8')
            try:metrics[label]=ev.evaluate(gold,pred)
            except Exception as exc:metrics[label]=None;metric_errors[label]={'type':type(exc).__name__,'message':str(exc),'evaluator_unmodified':True}
            if metrics[label] is not None:
                for subset,is_synthetic in [('official_real',False),('synthetic_stress',True)]:
                    subgold={cid:c for cid,c in gold.items() if cid.startswith('synthetic-v23-')==is_synthetic}
                    if subgold:metrics_by_slice.setdefault(subset,{})[label]=ev.evaluate(subgold,{cid:pred[cid] for cid in subgold})
            if serving and metrics[label] is not None:
                from ml.qwen.error_analysis_v21 import analyze
                save(out/(label+'_error_analysis.json'),analyze(ev,gold,pred))
    stats={'complete':len(records)==len(calls),'n_completed':len(records),'n_planned':len(calls),'metrics':metrics,'metrics_by_slice':metrics_by_slice,'metric_errors':metric_errors,
        'generation_seconds_total':sum(r['generation_seconds'] for r in records),'generation_seconds_this_process':active,
        'mean_seconds_per_call':statistics.mean(r['generation_seconds'] for r in records) if records else None,
        'p50_seconds_per_call':statistics.median(r['generation_seconds'] for r in records) if records else None,
        'peak_cuda_allocated_bytes':max([torch.cuda.max_memory_allocated()]+[r.get('peak_cuda_allocated_bytes',0) for r in records]),
        'peak_cuda_reserved_bytes':max([torch.cuda.max_memory_reserved()]+[r.get('peak_cuda_reserved_bytes',0) for r in records]),
        'statuses':{v:dict(collections.Counter(r['views'][v]['status'] for r in records)) for v in (records[0]['views'] if records else [])},
        'scope':'Full planned set' if len(records)==len(calls) else 'PARTIAL, NOT an official pass/fail result',
        'resource_definition':'Synchronized single-GPU generation wall time; excludes warmup/load/postprocess; no official-baseline speedup claim.'}
    save(out/'metrics.json',stats);return stats


def export_model(root,artifact,run,choice,reload,model):
    out=run/'export';out.mkdir(exist_ok=True)
    for folder in ['adapter','tokenizer']:shutil.copytree(artifact/folder,out/folder,dirs_exist_ok=True)
    for folder in ['ml','src']:shutil.copytree(root/folder,out/folder,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copyfile(root/'prompt.txt',out/'prompt.txt');shutil.copyfile(root/'official_catalog.json',out/'catalog.json')
    m={'model_id':f"qwen06-v22-step648-plus{choice['step']}",'base_model':MODEL,'base_revision':REVISION,
       'selected_new_updates':choice['step'],'calendar_enabled':choice['calendar'],'reload_verified':reload['passed'],
       'reload_test':reload,'precision':'fp16_base_fp32_adapter','total_parameters':sum(p.numel() for p in model.parameters()),
       'factory_entrypoint':'ml.qwen.final_runtime:factory','context':24576,'max_new_tokens':2048,
       'selection':choice,'official_case_gate_pass':None,'sha256':{p.relative_to(out).as_posix():sha(p) for p in out.rglob('*') if p.is_file() and p.name!='model_manifest.json'}}
    save(out/'model_manifest.json',m);return out



if __name__=="__main__":
    raise SystemExit("Use train_temporal.py for corrective training or validate_export.py for the selected export.")
