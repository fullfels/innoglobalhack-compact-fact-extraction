"""Optional post-training official evaluation, sharing the original five-hour budget."""
import argparse,csv,gc,json,time
from pathlib import Path
from final_eval import verify,rows,save,evaluate
from ml.qwen.final_runtime import load_model,Engine,sha,MODEL,REVISION

def run(root,training_run,resources=True):
    import torch
    from gpu_guard import check_idle_worker
    check_idle_worker()
    root=root.resolve();training_run=training_run.resolve();verify(root)
    budget=json.loads((training_run/'session_budget.json').read_text());deadline=budget['deadline_epoch']
    if time.time()+300>deadline:raise RuntimeError('Shared run budget exhausted; do not reset it silently')
    export=training_run/'export';m=json.loads((export/'model_manifest.json').read_text())
    if not m['reload_verified'] or m['base_model']!=MODEL or m['base_revision']!=REVISION:raise ValueError('Model export not verified')
    for rel,expected in m['sha256'].items():
        p=(export/rel).resolve()
        if not p.is_relative_to(export) or sha(p)!=expected:raise ValueError('Export hash mismatch: '+rel)
    out=training_run/'official_phase';out.mkdir(exist_ok=True)
    meta={'model_manifest_sha256':sha(export/'model_manifest.json'),'package_manifest_sha256':sha(root/'package_manifest.json'),
          'gpu':torch.cuda.get_device_name(0),'calendar':m['calendar_enabled'],'shared_deadline_epoch':deadline}
    if (out/'config.json').exists() and json.loads((out/'config.json').read_text())!=meta:raise ValueError('Validation resume config differs')
    save(out/'config.json',meta);torch.manual_seed(42);torch.backends.cuda.matmul.allow_tf32=False
    instruction=(export/'prompt.txt').read_text(encoding='utf-8');catalog=json.loads((export/'catalog.json').read_text(encoding='utf-8'))
    measured={};model=tok=engine=None
    if resources:
        for precision in ['fp16','int8','int4']:
            if time.time()+600>deadline:measured[precision]={'status':'not_run_budget'};continue
            if model is not None:del engine,model,tok;gc.collect();torch.cuda.empty_cache()
            try:
                model,tok=load_model(export/'adapter',export/'tokenizer',precision=precision,local_only=True)
                from final_eval import warmup
                warmup(model,tok);engine=Engine(model,tok,instruction,catalog,deadline_epoch=deadline-120)
                measured[precision]=evaluate(engine,rows(root/'data/resource4.jsonl'),root,out/('resource_'+precision),sha(export/'adapter/adapter_model.safetensors'),precision,min(deadline-300,time.time()+600),allow_calendar_choices=False,calendar_enabled=m['calendar_enabled'])
            except Exception as exc:
                measured[precision]={'status':'failed','error_type':type(exc).__name__,'message':str(exc)[:500]};model=tok=engine=None;gc.collect();torch.cuda.empty_cache()
            save(out/'resource_modes.json',measured)
    if model is not None:del engine,model,tok;gc.collect();torch.cuda.empty_cache()
    model,tok=load_model(export/'adapter',export/'tokenizer',local_only=True)
    from final_eval import warmup
    warmup(model,tok);engine=Engine(model,tok,instruction,catalog,deadline_epoch=deadline-90)
    report=evaluate(engine,rows(root/'evaluation/val.jsonl'),root,out/'official_val383',sha(export/'adapter/adapter_model.safetensors'),'fp16',deadline-90,allow_calendar_choices=False,calendar_enabled=m['calendar_enabled'])
    view=('calendar' if m['calendar_enabled'] else 'partial')+'_serving';primary=report['metrics'].get(view)
    checks=None
    if report['complete'] and not report['metric_errors']:
        t=primary['temporal_recall'];s=primary['state_accuracy']
        checks={'temporal_baseline':.4706,'temporal_consideration_floor':.4206,'temporal_requested_buffer':.44,
            'meets_temporal_floor':t is not None and t>=.4206,'meets_temporal_buffer':t is not None and t>=.44,
            'state_main_target':.5145,'meets_state_main_target':s is not None and s>=.5145,
            'resource_gain_vs_official_baseline_proven':False}
    save(out/'final_report.json',{'official':report,'primary_serving_metrics':primary,'quality_checks':checks,'resources':measured,
        'all_task_requirements_proven':False,'scope':'Only full383 and error-free evaluator can support official metric comparison.'})
    with (out/'resource_curve.csv').open('w',encoding='utf-8',newline='') as f:
        names=['precision','calls','complete','seconds_per_call','peak_cuda_allocated_mib','temporal_recall','n_temporal','state_accuracy','fact_recall','status']
        w=csv.DictWriter(f,fieldnames=names);w.writeheader()
        for precision,r in measured.items():
            v=r.get('metrics',{}).get(view)or{}
            w.writerow({'precision':precision,'calls':r.get('n_completed',0),'complete':r.get('complete',False),'seconds_per_call':r.get('mean_seconds_per_call'),
                'peak_cuda_allocated_mib':r.get('peak_cuda_allocated_bytes',0)/1024**2,'temporal_recall':v.get('temporal_recall'),'n_temporal':v.get('n_temporal'),
                'state_accuracy':v.get('state_accuracy'),'fact_recall':v.get('fact_recall'),'status':r.get('status','measured')})
    print(json.dumps({'complete':report['complete'],'n_completed':report['n_completed'],'metrics':primary,'quality_checks':checks},ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--run',type=Path,required=True);p.add_argument('--no-resources',action='store_true');a=p.parse_args()
    try:run(a.root,a.run,not a.no_resources)
    except Exception as exc:save(a.run/'official_phase/failure.json',{'type':type(exc).__name__,'message':str(exc)[:1500]});raise
