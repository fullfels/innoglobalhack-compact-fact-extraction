"""Bounded, resumable, batch=1 resource/quality pilot. No training or hidden GPU jobs."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,gc,hashlib,importlib.metadata,json,os,platform,statistics,subprocess,time,traceback

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,obj):
    p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8');os.replace(t,p)
def readrows(p):
    rows=[json.loads(s) for s in p.read_text(encoding='utf-8').splitlines() if s.strip()]
    assert len({c['call_id'] for c in rows})==len(rows)
    return rows

def run(root,out,device,modes,budget_minutes):
    import torch
    from ml.qwen.serving import verify_export,verify_base,POLICY_FILES
    from ml.qwen.final_runtime import load_model,Engine,MODEL,REVISION
    from ml.qwen.final_bridge import FinalBridge
    from ml.qwen.postprocessing import TariffValueNormalizer,POLICY_ID
    from ml.qwen.replay_v22 import load_evaluator
    from src.contracts.models import CallInput
    from src.pipeline.adapters import InjectedAdapter,ModelResources
    from src.pipeline.service import Pipeline
    from src.experiments.benchmark import MemorySampler,hardware
    from gpu_guard import check_idle_worker
    root=root.resolve();out=out.resolve();out.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((root/'package_manifest.json').read_text())
    for rel,want in manifest['sha256'].items():
        p=(root/rel).resolve();assert p.is_relative_to(root) and sha(p)==want,rel
    export,m=verify_export(root/'export')
    assert m['source_parent']=='v21_step648_plus500' and m['selected_corrective_updates']==0
    assert sha(export/'adapter/adapter_model.safetensors')=='577bc8184877de05386244791bee2ecaeb5dd894e86df749cd31ed270dbab98b'
    if device=='cuda':assert torch.cuda.is_available();check_idle_worker()
    else:
        assert modes==['fp16'],'CPU is measured as actual float32; quantized modes require GPU'
        verify_base(os.environ['MTS_MODEL_BASE_PATH'])
    absolute=datetime.fromisoformat('2026-11-16T07:15:00+03:00').timestamp()
    bp=out/'budget.json'
    if bp.exists():budget=json.loads(bp.read_text());assert budget['minutes']==budget_minutes
    else:
        origin=float(os.getenv('BENCH_ORIGIN_EPOCH',time.time()))
        assert origin<=time.time()+5
        budget={'origin_epoch':origin,'deadline_epoch':min(origin+budget_minutes*60,absolute),'minutes':budget_minutes}
        save(bp,budget)
    deadline=budget['deadline_epoch'];assert deadline-time.time()>60,'Saved benchmark budget exhausted'
    calls=readrows(root/'data/resource4.jsonl');assert len(calls)==4
    gold={c['call_id']:c for c in calls}
    catalog=json.loads((export/'catalog.json').read_text(encoding='utf-8'))
    evaluator=load_evaluator(root/'evaluation/case_metrics.py')
    protocol={'device':device,'modes':modes,'base':MODEL,'revision':REVISION,'model_id':m['model_id'],
        'adapter_sha256':sha(export/'adapter/adapter_model.safetensors'),'postprocessing':POLICY_ID,
        'postprocessing_sha256':sha(root/'ml/qwen/postprocessing.py'),'package_sha256':sha(root/'package_manifest.json'),
        'call_ids':[c['call_id'] for c in calls],'input_sha256':sha(root/'data/resource4.jsonl'),
        'batch_size':1,'do_sample':False,'thinking':False,'use_cache':True,'max_new_tokens':2048,
        'seed':42,'repeats':1,'warmup':'Two generated tokens on a short fixed phrase; not a long thermal warmup',
        'per_call_seconds':600 if device=='cpu' else 120,
        'timing':'Synchronized completed pipeline wall time including tokenization/postprocessing; excludes load, warmup and file IO',
        'rss':'Current model process RSS sampled every20ms, not full OS/process-tree memory',
        'gpu_memory':'PyTorch allocated/reserved peak, including resident model, excludes some driver/context allocations',
        'checkpoint_disk_note':'Base+adapter files remain unquantized on disk; INT8/INT4 quantization occurs at load',
        'scope':'Small preselected internal diagnostic subset, not representative383 or hidden test',
        'hardware':hardware(device),
        'versions':{n:importlib.metadata.version(n) for n in ['torch','transformers','peft','tokenizers','accelerate','safetensors','pydantic']}}
    if device=='cuda':protocol['hardware']['cuda']={'name':torch.cuda.get_device_name(0),'runtime':torch.version.cuda,'vram_bytes':torch.cuda.get_device_properties(0).total_memory}
    if device=='cuda':protocol['versions']['bitsandbytes']=importlib.metadata.version('bitsandbytes')
    protocol['hardware']['torch_num_threads']=torch.get_num_threads()
    protocol['hardware']['torch_num_interop_threads']=torch.get_num_interop_threads()
    cpuinfo=Path('/proc/cpuinfo')
    if cpuinfo.exists():protocol['hardware']['cpu_model']=next((s.split(':',1)[1].strip() for s in cpuinfo.read_text().splitlines() if s.startswith('model name')),None)
    for name in ['cpu.max','memory.max']:
        p=Path('/sys/fs/cgroup')/name
        if p.exists():protocol['hardware']['cgroup_'+name]=p.read_text().strip()
    pp=out/'protocol.json'
    if pp.exists():assert json.loads(pp.read_text())==protocol,'Resume hardware/config changed; do not mix runs'
    save(pp,protocol)
    def sync():
        if device=='cuda':torch.cuda.synchronize()
    summaries={}
    for precision in modes:
        folder=out/precision;folder.mkdir(exist_ok=True)
        if (folder/'summary.json').is_file():
            old=json.loads((folder/'summary.json').read_text())
            if old.get('complete') or old.get('load_failed'):summaries[precision]=old;continue
        model=tok=engine=bridge=pipeline=None;records=[]
        try:
            if deadline-time.time()<90:break
            print('LOAD',device,precision,flush=True)
            torch.manual_seed(42)
            if device=='cuda':torch.backends.cuda.matmul.allow_tf32=False;torch.cuda.reset_peak_memory_stats()
            start=time.perf_counter()
            with MemorySampler() as load_memory:
                model,tok=load_model(export/'adapter',export/'tokenizer',precision=precision,device=device,local_only=device=='cpu')
                sync()
            load_seconds=time.perf_counter()-start
            x=tok('Проверка.',return_tensors='pt').to(device)
            sync();start=time.perf_counter()
            with torch.inference_mode():model.generate(**x,do_sample=False,max_new_tokens=2,use_cache=True,pad_token_id=tok.pad_token_id)
            sync();warmup_seconds=time.perf_counter()-start;del x
            engine=Engine(model,tok,(export/'prompt.txt').read_text(encoding='utf-8'),catalog,
                device=device,max_call_seconds=protocol['per_call_seconds'],deadline_epoch=deadline-30)
            bridge=FinalBridge(engine,calendar=False)
            pipeline=Pipeline(InjectedAdapter(ModelResources('',None),bridge),normalizers=(TariffValueNormalizer(catalog),))
            for i,row in enumerate(calls):
                call=CallInput.model_validate({k:row[k] for k in ['call_id','started_at','turns']})
                fingerprint=hashlib.sha256(json.dumps(call.model_dump(mode='json'),sort_keys=True).encode()).hexdigest()
                cp=folder/'records'/f'{i:02d}.json'
                if cp.exists():
                    r=json.loads(cp.read_text());assert r['input_sha256']==fingerprint;records.append(r);continue
                if time.time()+protocol['per_call_seconds']+30>deadline:break
                r={'call_id':call.call_id,'input_sha256':fingerprint,'status':'failed','facts':None,'raw':None,'error_code':None}
                print('START',precision,f'{i+1}/4',call.call_id,flush=True)
                if device=='cuda':torch.cuda.reset_peak_memory_stats()
                with MemorySampler() as memory:
                    sync();start=time.perf_counter()
                    try:
                        prediction,warnings=pipeline.run(call)
                        r.update(status=bridge.last_result.status,facts=[f.model_dump(mode='json') for f in prediction.facts],warnings=list(warnings),rejected=list(bridge.last_result.rejected))
                    except Exception as exc:
                        r['error_code']=getattr(exc,'code',type(exc).__name__)
                    finally:sync();r['seconds']=time.perf_counter()-start
                r.update(raw=engine.last_raw,generation=engine.last_diagnostics,peak_rss_bytes=memory.peak,
                    peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated() if device=='cuda' else None,
                    peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved() if device=='cuda' else None)
                save(cp,r);records.append(r)
                print('DONE',precision,f'{len(records)}/4',r['status'],round(r['seconds'],2),'sec',flush=True)
            pred={r['call_id']:{'call_id':r['call_id'],'facts':r['facts'] or []} for r in records}
            # Missing/failed calls are penalized, never relabelled as successful empty generation.
            metrics=evaluator.evaluate(gold,pred)
            durations=sorted(r['seconds'] for r in records);total=sum(durations)
            accepted=sum(r['status']!='failed' for r in records)
            summary={'complete':len(records)==4,'mode':precision,'device':device,'actual_dtype':'float32' if device=='cpu' else precision,
                'load_seconds':load_seconds,'warmup_seconds':warmup_seconds,'n_planned':4,'n_attempted':len(records),
                'accepted_responses':accepted,'fully_accepted_generations':sum(r['status']=='ok' for r in records),
                'generation_failures':len(records)-accepted,'metrics':metrics,
                'mean_seconds_all_attempts':statistics.mean(durations) if durations else None,
                'p50_seconds':statistics.median(durations) if durations else None,
                'p95_nearest_rank_seconds':durations[-1] if durations else None,
                'p95_limitation':'Only four samples and one pass; descriptive, not stable population percentile',
                'successful_calls_per_hour':3600*accepted/total if total else None,
                'gpu_seconds_per_1000_attempts':1000*total/len(records) if device=='cuda' and records else None,
                'peak_rss_bytes':max([x for x in [load_memory.peak]+[r['peak_rss_bytes'] for r in records] if x is not None],default=None),
                'peak_cuda_allocated_bytes':max([r['peak_cuda_allocated_bytes'] or 0 for r in records],default=None) if device=='cuda' else None,
                'peak_cuda_reserved_bytes':max([r['peak_cuda_reserved_bytes'] or 0 for r in records],default=None) if device=='cuda' else None,
                'logical_parameters':m['total_parameters'],'disk_is_unquantized':True,
                'status':'measured' if len(records)==4 else 'partial_not_full_point'}
            (folder/'predictions.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in pred.values()),encoding='utf-8')
            save(folder/'summary.json',summary);summaries[precision]=summary
        except Exception as exc:
            summary={'mode':precision,'device':device,'complete':False,'load_failed':True,
                'error_type':type(exc).__name__,'message':str(exc)[:700],'status':'not_measured_not_a_curve_point'}
            save(folder/'summary.json',summary);summaries[precision]=summary
            traceback.clear_frames(exc.__traceback__)
        finally:
            pipeline=bridge=engine=model=tok=None;gc.collect()
            if device=='cuda':torch.cuda.empty_cache()
        save(out/'summary.json',{'device':device,'modes':summaries,'new_training':False,'official_baseline_speedup_proven':False,
            'all_three_gpu_points_complete':device=='cuda' and len(summaries)==3 and all(s.get('complete') for s in summaries.values())})
    save(out/'summary.json',{'device':device,'modes':summaries,'new_training':False,'official_baseline_speedup_proven':False,
        'all_three_gpu_points_complete':device=='cuda' and len(summaries)==3 and all(s.get('complete') for s in summaries.values())})
    print('BENCHMARK FINISHED; complete points:',sum(s.get('complete',False) for s in summaries.values()),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--device',choices=['cpu','cuda'],required=True);p.add_argument('--modes',nargs='+',choices=['fp16','int8','int4'],required=True)
    p.add_argument('--budget-minutes',type=int,required=True);a=p.parse_args()
    assert 10<=a.budget_minutes<=60
    try:run(a.root,a.out,a.device,a.modes,a.budget_minutes)
    except Exception as exc:
        save(a.out/'failure.json',{'error_type':type(exc).__name__,'message':str(exc)[:900],'new_training':False});raise
