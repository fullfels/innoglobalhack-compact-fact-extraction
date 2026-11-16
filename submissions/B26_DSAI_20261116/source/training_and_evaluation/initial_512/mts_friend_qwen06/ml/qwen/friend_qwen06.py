"""One bounded Qwen3-0.6B LoRA experiment on the sealed 512/20 subset.

Reuses the team's completion-only loss, strict bridge, and official evaluator.
No training or reported quality is simulated by this module.
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import gc
import importlib.metadata
import json
import math
from pathlib import Path
import random
import shutil
import signal
import statistics
import time

from ml.qwen.train import (read, save, sha, load_module, suffix_loss,
    check_loss_equivalence, token_loss_weights, supervised_tokens, quality_run,
    checkpoint, checkpoint_quality_key, autocast_for)
from ml.qwen.prompting import effective_example

MODEL = 'Qwen/Qwen3-0.6B'
REVISION = 'c1899de289a04d12100db370d81485cdf75e47ca'
STOP_REQUESTED = False


def request_stop(_signum, _frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True


def verify_payload(root):
    manifest = json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    for rel, expected in manifest['sha256'].items():
        path = (root/rel).resolve()
        if not path.is_relative_to(root.resolve()) or sha(path) != expected:
            raise ValueError('Payload checksum mismatch: ' + rel)
    return manifest


def load_base(root, precision):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    dtype = torch.float16 if precision == 'float16' else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(root/'tokenizer', local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION,
        dtype=dtype, attn_implementation='sdpa', trust_remote_code=False).to('cuda')
    if model.config.model_type != 'qwen3':
        raise RuntimeError('Unexpected model architecture')
    if tokenizer.eos_token_id != model.config.eos_token_id:
        raise RuntimeError('Base/tokenizer EOS mismatch')
    tokenizer.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    return model, tokenizer


def paired_view(calls, generated_dir, root, out):
    """Reuse real dev20 generations for the same4 comparison; never generate twice."""
    values = [json.loads(p.read_text(encoding='utf-8')) for p in (generated_dir/'records').glob('*.json')]
    records = {r['call_id']:r for r in values}
    if len(records)!=len(values): raise ValueError('Duplicate generation records')
    selected = [records[c['call_id']] for c in calls]
    for record in selected:
        if record['status'] not in ('ok','failed') or (record['status']=='failed' and record['facts'] is not None):
            raise ValueError('Inconsistent failed generation record')
    preds = {r['call_id']:{'call_id':r['call_id'],'facts':r['facts'] if r['status']=='ok' else []} for r in selected}
    serving = {r['call_id']:{'call_id':r['call_id'],'facts':r.get('serving_facts',[]) if r['status']=='ok' else []} for r in selected}
    evaluator=load_module('friend_paired_evaluator',root/'evaluation/case_metrics.py')
    gold={c['call_id']:c for c in calls}
    result={'metrics':evaluator.evaluate(gold,preds),'n_calls':len(calls),
        'status_counts':dict(Counter(r['status'] for r in selected)),
        'serving_metrics_without_de_normalizers':evaluator.evaluate(gold,serving),
        'failed_generations':sum(r['status']!='ok' for r in selected),
        'generation_reused_from':'after_dev20; identical raw outputs, no second generation'}
    out.mkdir(exist_ok=True)
    save(out/'metrics.json',result)
    for r in selected:
        save(out/'records'/(__import__('hashlib').sha256(r['call_id'].encode()).hexdigest()[:20]+'.json'),r)
    (out/'predictions.jsonl').write_text(''.join(json.dumps(p,ensure_ascii=False)+'\n' for p in preds.values()),encoding='utf-8')
    return result


def evaluate_saved(root, run, adapter_name, precision):
    import torch
    from peft import PeftModel
    if adapter_name not in ('last', 'best'):
        raise ValueError('Expected last or best')
    base, tokenizer = load_base(root, precision)
    model = PeftModel.from_pretrained(base, run/adapter_name/'adapter').eval()
    report = quality_run(model, tokenizer, read(root/'data/dev_20.jsonl'), root,
                         run/('recheck_' + adapter_name + '_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')))
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


def run_training(args):
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU is required; do not train on CPU silently')
    root, out = args.root.resolve(), args.out.resolve()
    manifest = verify_payload(root)
    if out.exists() and any(out.iterdir()):
        raise RuntimeError('Use a fresh run folder; existing runs are immutable')
    out.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(root/'manifest.json',out/'payload_manifest.json')
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    started = time.perf_counter()
    config = {'status': 'starting', 'base_model': MODEL, 'base_revision': REVISION,
        'seed': 42, 'precision': args.precision, 'quantization': None,
        'rank': 16, 'alpha': 32, 'learning_rate': args.learning_rate,
        'epochs_requested': 1, 'accumulation': 4, 'budget_hours': args.budget_hours,
        'budget_scope': 'Wall time of optimizer loop including intermediate diagnostics; final evaluation/reload extra',
        'loss_normalization': 'supervised_tokens_across_accumulation_window',
        'context': 16384, 'max_new_tokens': 1536, 'thinking': False,
        'sampling': False, 'cache': True, 'train_calls': 512, 'dev_calls': 20,
        'baseline_calls': 4, 'checkpoint_selection_calls': 4,
        'selection_policy': 'Fixed first4 inner-dev by factF1/recall/state/time; CE is not used for model selection',
        'dataset_manifest_sha256': sha(root/'manifest.json'),
        'parent_dataset_manifest_sha256': manifest['parent_manifest_sha256'],
        'dependencies': {p: importlib.metadata.version(p) for p in ('torch','transformers','peft','accelerate','tokenizers','safetensors')},
        'hardware': {'gpu':torch.cuda.get_device_name(0),'vram_bytes':torch.cuda.get_device_properties(0).total_memory,'cuda':torch.version.cuda},
        'source_sha256': {str(p.relative_to(root)).replace('\\','/'):sha(p) for p in sorted((root/'ml/qwen').glob('*.py'))},
        'full_official_val_run': False, 'experiment_not_a_claim_of_case_gate_pass': True}
    save(out/'training_config.json', config)
    random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    tokenize = load_module('friend_completion_tokenizer', root/'tokenize_sft.py')
    base, tokenizer = load_base(root, args.precision)
    raw_train, raw_dev = read(root/'data/train_512.jsonl'), read(root/'data/dev_20.jsonl')
    encoded = [tokenize.encode_record(effective_example(r), tokenizer, max_length=16384)
               for r in read(root/'sft/train_512.jsonl')]
    dev_encoded = [tokenize.encode_record(effective_example(r), tokenizer, max_length=16384)
                   for r in read(root/'sft/dev_20.jsonl')]
    assert len(encoded) == 512 and len(dev_encoded) == 20
    assert max(supervised_tokens(r) for r in encoded + dev_encoded) < 1536
    # Explicit selectors: no adapters on embeddings/lm_head or another architecture.
    targets = [name for name, module in base.named_modules()
               if isinstance(module, torch.nn.Linear) and '.layers.' in name]
    assert len(targets) == 28*7
    model = get_peft_model(base, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.,
        target_modules=targets, bias='none', task_type='CAUSAL_LM'))
    del base
    for p in model.parameters():
        if p.requires_grad: p.data = p.data.float()
    config.update(total_parameters=sum(p.numel() for p in model.parameters()),
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        target_modules=targets, token_lengths={'train_max':max(len(r['input_ids']) for r in encoded),
        'train_median':statistics.median(len(r['input_ids']) for r in encoded),
        'answer_max':max(supervised_tokens(r) for r in encoded+dev_encoded)})
    save(out/'training_config.json', config)
    save(out/'loss_equivalence.json', check_loss_equivalence(model, tokenizer))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    model.enable_input_require_grads(); model.config.use_cache = False
    longest = max(encoded, key=lambda r:len(r['input_ids']))
    model.train(); model.zero_grad(set_to_none=True); torch.cuda.reset_peak_memory_stats()
    with autocast_for(model): probe_loss = suffix_loss(model, longest)
    probe_loss.backward()
    norms = [p.grad.detach().float().norm() for p in model.parameters() if p.requires_grad and p.grad is not None]
    assert norms and all(bool(torch.isfinite(x)) for x in norms), 'Non-finite longest-call probe gradient'
    save(out/'long_call_probe.json', {'passed':True,'tokens':len(longest['input_ids']),
        'optimizer_update':False,'peak_cuda_allocated_mib':torch.cuda.max_memory_allocated()/1024**2})
    model.zero_grad(set_to_none=True); del probe_loss, norms
    # Pair the same four fixed dev IDs before/after; after-dev20 is a separate scope.
    baseline = quality_run(model, tokenizer, raw_dev[:4], root, out/'before_dev4')
    if STOP_REQUESTED: raise KeyboardInterrupt('Stop requested before optimizer loop')
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.learning_rate, weight_decay=0.)
    scaler = torch.amp.GradScaler('cuda', enabled=args.precision=='float16', init_scale=32.)
    order = list(range(512)); random.Random(42).shuffle(order)
    logs, best_key, best_step, last_saved_step = [], None, None, 0
    first_b_name, first_b = next((n,p) for n,p in model.named_parameters() if p.requires_grad and 'lora_B' in n)
    initial_b = first_b.detach().clone()
    loop_start = time.perf_counter(); reason = 'one_epoch_completed'
    for step in range(128):
        if STOP_REQUESTED or time.perf_counter()-loop_start >= args.budget_hours*3600:
            reason = 'user_stop' if STOP_REQUESTED else 'training_budget_reached'; break
        model.train(); optimizer.zero_grad(set_to_none=True); torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize(); t0 = time.perf_counter()
        batch = [encoded[i] for i in order[step*4:(step+1)*4]]
        multiplier = min((step+1)/7, max(0., (128-step)/(128-7)))
        for group in optimizer.param_groups: group['lr'] = args.learning_rate*multiplier
        weights = token_loss_weights(batch); values=[]
        for row,w in zip(batch,weights):
            with autocast_for(model): loss = suffix_loss(model,row)
            if not torch.isfinite(loss): raise RuntimeError('Non-finite loss; stopping before update')
            values.append(float(loss.detach())); scaler.scale(loss*w).backward()
        scaler.unscale_(optimizer); norm = torch.nn.utils.clip_grad_norm_(params,1.)
        if not torch.isfinite(norm):
            raise RuntimeError('Non-finite gradient; no optimizer update. Save logs, use explicit new FP32 run if needed')
        old_scale=scaler.get_scale(); scaler.step(optimizer); scaler.update()
        if scaler.get_scale() < old_scale: raise RuntimeError('Unexpected skipped optimizer update')
        torch.cuda.synchronize()
        item={'step':step+1,'examples_seen':(step+1)*4,'supervised_tokens':sum(supervised_tokens(r) for r in batch),
            'loss':sum(x*w for x,w in zip(values,weights)),'gradient_norm':float(norm),
            'learning_rate':optimizer.param_groups[0]['lr'],'elapsed_sec':time.perf_counter()-t0,
            'peak_cuda_allocated_mib':torch.cuda.max_memory_allocated()/1024**2,
            'peak_cuda_reserved_mib':torch.cuda.max_memory_reserved()/1024**2,'optimizer_step_skipped':False}
        logs.append(item); save(out/'training_log.json',logs)
        print('step',step+1,'/128','calls',item['examples_seen'],'loss',round(item['loss'],5),'sec',round(item['elapsed_sec'],2),flush=True)
        if step+1 == 3:
            changed=not torch.equal(initial_b,first_b.detach())
            assert changed, 'LoRA weights did not change'
            save(out/'first_updates.json',{'passed':True,'finite_updates':3,'weight_changed':changed,'parameter':first_b_name})
            del initial_b
        if (step+1)%32 == 0:
            checkpoint(model,tokenizer,out/'last',root,step+1,encoded[0]); last_saved_step=step+1
            if (step+1)%64 == 0:
                measured=quality_run(model,tokenizer,raw_dev[:4],root,out/f'selection_step{step+1}')
                key=checkpoint_quality_key(measured,0.)
                if best_key is None or key>best_key:
                    best_key,best_step=key,step+1
                    checkpoint(model,tokenizer,out/'best',root,step+1,encoded[0])
                save(out/'progress.json',{'step':step+1,'best_step':best_step,'selection':measured})
    steps=len(logs)
    if not steps: raise RuntimeError('No optimizer updates completed')
    if last_saved_step != steps: checkpoint(model,tokenizer,out/'last',root,steps,encoded[0])
    save(out/'training_complete.json',{'steps':steps,'examples_seen':steps*4,
        'one_epoch_completed':steps==128,'stop_reason':reason,'elapsed_loop_sec':time.perf_counter()-loop_start})
    if STOP_REQUESTED:
        save(out/'experiment_report.json',{'status':'stopped_after_checkpoint','training_steps':steps,
            'training_calls_seen':steps*4,'one_epoch_completed':steps==128,'stop_reason':'user_stop',
            'after_quality_measured':False,'full_official_val_run':False,'official_case_gate_pass':None})
        print('Stopped after saving last checkpoint; final generation was not started.',flush=True)
        return
    # Fresh base+saved adapter, not reusing an in-memory model.
    selected='best' if best_step is not None else 'last'; selected_step=best_step or steps
    del model,optimizer,params,first_b,loss; gc.collect(); torch.cuda.empty_cache()
    fresh,_=load_base(root,args.precision)
    model=PeftModel.from_pretrained(fresh,out/selected/'adapter').eval()
    reference=torch.load(out/selected/'reload_reference.pt',weights_only=True,map_location='cpu')
    x=reference['input_ids'].to('cuda')
    with torch.no_grad(),autocast_for(model):
        actual=model(input_ids=x,attention_mask=torch.ones_like(x),use_cache=False,logits_to_keep=1).logits.float().cpu()
    torch.testing.assert_close(reference['logits'],actual,atol=.003,rtol=.003)
    reload={'passed':True,'fresh_base_and_saved_adapter':True,'max_logit_abs_difference':float((actual-reference['logits']).abs().max())}
    save(out/'reload_test.json',reload)
    after=quality_run(model,tokenizer,raw_dev,root,out/'after_dev20')
    after_paired=paired_view(raw_dev[:4],out/'after_dev20',root,out/'after_dev4_same_ids')
    train_check=quality_run(model,tokenizer,raw_train[:4],root,out/'after_train4_diagnostic')
    export=out/'export'; shutil.copytree(out/selected,export,dirs_exist_ok=True)
    bundle_hashes={str(p.relative_to(export)).replace('\\','/'):sha(p) for p in sorted(export.rglob('*')) if p.is_file()}
    descriptor={'base_model':MODEL,'base_revision':REVISION,'dtype':args.precision,
        'total_parameters':config['total_parameters'],'trainable_parameters':config['trainable_parameters'],
        'context':16384,'max_new_tokens':1536,'enable_thinking':False,'do_sample':False,
        'schema':'src.contracts.models.CandidateBatch via ml.qwen.bridge.parse_candidates',
        'selected_step':selected_step,'reload_verified':True,'sha256':bundle_hashes,
        'artifact_state':'research_candidate_reload_verified','official_case_gates_verified':False,
        'serving_acceptance':'pending DE normalizers and FS integration; no automatic production factory claim'}
    save(export/'model_manifest.json',descriptor)
    report={'status':'completed','experiment':'bounded512_qwen3_06b','training_steps':steps,
        'training_calls_seen':steps*4,'one_epoch_completed':steps==128,'stop_reason':reason,
        'selected_checkpoint':selected,'selected_step':selected_step,'base_model':MODEL,'base_revision':REVISION,
        'dtype':args.precision,'reload':reload,'before_dev4':baseline,'after_dev4_same_ids':after_paired,
        'after_dev20':after,'after_train4_diagnostic':train_check,
        'seconds_per_microbatch_median':statistics.median(r['elapsed_sec']/4 for r in logs[-20:]),
        'elapsed_total_sec':time.perf_counter()-started,'full_official_val_run':False,
        'official_case_gate_pass':None,'limitations':'Only inner-dev20; temporal denominator is matched facts. No end-to-end DE/HTTP acceptance.'}
    save(out/'experiment_report.json',report)
    config['status']='completed';save(out/'training_config.json',config)
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--precision',choices=('float16','float32'),default='float16')
    ap.add_argument('--learning-rate',type=float,default=1e-4)
    ap.add_argument('--budget-hours',type=float,default=4.)
    ap.add_argument('--recheck',choices=('best','last'))
    args=ap.parse_args()
    if not 0 < args.budget_hours <= 4 or not 0 < args.learning_rate <= .001:
        ap.error('Expected 0 < budget_hours <= 4 and 0 < learning_rate <= 0.001')
    if args.recheck:
        verify_payload(args.root.resolve())
        evaluate_saved(args.root,args.out,args.recheck,args.precision)
        return
    existed=args.out.exists() and any(args.out.iterdir())
    try: run_training(args)
    except BaseException as exc:
        if not existed:
            save(args.out/'failure.json',{'status':'failed','error_type':type(exc).__name__,
                'message':str(exc),'quality_claimed':False,'note':'Return the existing archive/logs; do not overwrite this run.'})
        raise


if __name__=='__main__': main()
