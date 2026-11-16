"""Independent bounded inference-only run; immutable selected +500 and official evaluator."""
from pathlib import Path
import argparse
import importlib.metadata
import json
import os
import shutil
import time

from final_eval import verify, rows, save, evaluate, warmup
from ml.qwen.final_runtime import MODEL, REVISION, Engine, load_model, sha
from ml.qwen.replay_v22 import load_evaluator
from gpu_guard import check_idle_worker

EXPECTED_ADAPTER = '577bc8184877de05386244791bee2ecaeb5dd894e86df749cd31ed270dbab98b'

def run(root, out):
    import torch
    root=root.resolve();out=out.resolve();out.mkdir(parents=True,exist_ok=True)
    assert torch.cuda.is_available(), 'GPU required; choose A100 if available'
    verify(root)
    budget=json.loads((out/'eval_budget.json').read_text(encoding='utf-8'))
    deadline=budget['deadline_epoch']
    if time.time()+300>=deadline:raise RuntimeError('Evaluation budget expired; no silent reset')
    check_idle_worker()
    export=root/'export';m=json.loads((export/'model_manifest.json').read_text(encoding='utf-8'))
    assert m['base_model']==MODEL and m['base_revision']==REVISION
    assert m.get('selected_corrective_updates')==0 and m.get('source_parent')=='v21_step648_plus500'
    assert m.get('reload_verified') is True and m.get('reload_test',{}).get('passed') is True
    assert m.get('calendar_enabled') is False
    for rel,expected in m['sha256'].items():
        p=(export/rel).resolve()
        assert p.is_relative_to(export.resolve()) and sha(p)==expected, rel
    assert sha(export/'adapter/adapter_model.safetensors')==EXPECTED_ADAPTER
    for rel in ['ml/qwen/bridge.py','ml/qwen/final_runtime.py','ml/qwen/final_bridge.py',
                'ml/qwen/calendar_v23.py','ml/qwen/date_anchors_v21.py','ml/qwen/temporal_v21.py',
                'src/contracts/models.py','src/pipeline/adapters.py','src/pipeline/lifecycle.py','src/pipeline/service.py']:
        assert sha(root/rel)==m['sha256'][rel], 'Installed policy differs: '+rel
    assert (root/'prompt.txt').read_bytes()==(export/'prompt.txt').read_bytes()
    assert (root/'official_catalog.json').read_bytes()==(export/'catalog.json').read_bytes()
    calls=rows(root/'evaluation/val.jsonl')
    assert len(calls)==383 and sum(len(c['facts']) for c in calls)==1845
    baseline=rows(root/'evaluation/baseline_pred.jsonl')
    assert {c['call_id'] for c in calls}=={c['call_id'] for c in baseline}
    config={'model_id':m['model_id'],'model_manifest_sha256':sha(export/'model_manifest.json'),
        'package_manifest_sha256':sha(root/'package_manifest.json'),'gpu':torch.cuda.get_device_name(0),
        'adapter_sha256':EXPECTED_ADAPTER,'seed':42,'precision':'fp16','batch_size':1,
        'calendar_enabled':False,'thinking':False,'max_new_tokens':2048,'max_call_seconds':120,
        'context':24576,'use_cache':True,'training_performed':False,
        'versions':{name:importlib.metadata.version(name) for name in ['torch','transformers','peft','tokenizers','accelerate','safetensors','pydantic']}}
    cp=out/'inference_config.json'
    if cp.exists():assert json.loads(cp.read_text(encoding='utf-8'))==config,'Resume config changed'
    save(cp,config)
    print('INFERENCE ONLY: original +500, 383 official calls, GPU='+config['gpu'],flush=True)
    print('Loading pinned base and verified saved adapter. No training.',flush=True)
    torch.manual_seed(42);torch.backends.cuda.matmul.allow_tf32=False
    model,tok=load_model(export/'adapter',export/'tokenizer',local_only=False)
    warmup(model,tok)
    print('Model loaded. Starting/resuming per-call evaluation; records are saved to Drive.',flush=True)
    engine=Engine(model,tok,(export/'prompt.txt').read_text(encoding='utf-8'),
                  json.loads((export/'catalog.json').read_text(encoding='utf-8')),deadline_epoch=deadline-120)
    report=evaluate(engine,calls,root,out/'official_val383',EXPECTED_ADAPTER,'fp16',deadline-120,
                    allow_calendar_choices=False,calendar_enabled=False)
    primary=report['metrics'].get('partial_serving')
    ev=load_evaluator(root/'evaluation/case_metrics.py')
    gold={c['call_id']:c for c in calls};pred={c['call_id']:c for c in baseline}
    baseline_metrics=ev.evaluate(gold,pred)
    assert baseline_metrics['metrics_version']=='2.1'
    save(out/'baseline_recomputed.json',baseline_metrics)
    checks=None
    if report['complete'] and not report['metric_errors'] and primary is not None:
        bt,bs=baseline_metrics['temporal_recall'],baseline_metrics['state_accuracy']
        t,s=primary['temporal_recall'],primary['state_accuracy']
        checks={'temporal_main_min':round(bt-.02,4),'temporal_consideration_min':round(bt-.05,4),
                'state_main_min':round(bs-.02,4),
                'temporal_main_met':t is not None and t>=round(bt-.02,4),
                'temporal_consideration_met':t is not None and t>=round(bt-.05,4),
                'state_main_met':s is not None and s>=round(bs-.02,4),
                'resource_gain_vs_official_baseline_proven':False}
    statuses=report.get('statuses',{}).get('partial',{})
    final={'complete':report['complete'],'n_completed':report['n_completed'],'n_planned':383,
           'model_id':m['model_id'],'training_performed':False,
           'primary_serving_metrics':primary,'baseline_recomputed':baseline_metrics,
           'quality_checks':checks,'generation_statuses':statuses,
           'metric_errors':report['metric_errors'],'official_report':'official_val383/metrics.json',
           'mean_gpu_seconds_per_call':report['mean_seconds_per_call'],
           'peak_cuda_allocated_bytes':report['peak_cuda_allocated_bytes'],
           'failures_policy':'Failed generations are explicitly recorded as failed; their empty fact lists penalize official metrics, not successful empty outputs',
           'scope':'All 383 official validation calls' if report['complete'] else 'PARTIAL; no official threshold pass claimed',
           'all_task_requirements_proven':False}
    save(out/'final_report.json',final)
    shutil.copyfile(out/'official_val383/partial_serving_predictions.jsonl',out/'predictions.jsonl')
    print(json.dumps(final,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    try:run(a.root,a.run)
    except Exception as exc:
        save(a.run/'failure.json',{'type':type(exc).__name__,'message':str(exc)[:1800],
            'training_performed':False,'all_task_requirements_proven':False})
        raise
