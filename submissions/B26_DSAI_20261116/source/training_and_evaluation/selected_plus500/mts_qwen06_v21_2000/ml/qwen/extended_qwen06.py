"""Bounded Qwen3-0.6B last128 warm-start on new v2 calls; conventional completion-only SFT including all v2 dates.

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
    checkpoint as base_checkpoint, checkpoint_quality_key, autocast_for)
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


def select_language_targets(available, expected):
    """Reject changes to the measured language-only adapter layout."""
    if not expected or len(set(expected)) != len(expected):
        raise ValueError('Expected unique language targets')
    if set(available) != set(expected) or any('.layers.' not in n or 'visual' in n for n in available):
        raise ValueError('Unexpected language-only LoRA target layout')
    return list(available)


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
    """Subset already-generated records; never generate twice."""
    values = [json.loads(p.read_text(encoding='utf-8')) for p in (generated_dir/'records').glob('*.json')]
    records = {r['call_id']:r for r in values}
    if len(records)!=len(values): raise ValueError('Duplicate generation records')
    selected = [records[c['call_id']] for c in calls]
    for record in selected:
        if record['status'] not in ('ok','failed') or (record['status']=='failed' and record['facts'] is not None):
            raise ValueError('Inconsistent failed generation record')
    preds = {r['call_id']:{'call_id':r['call_id'],'facts':r['facts'] if r['status']=='ok' else []} for r in selected}
    serving = {r['call_id']:{'call_id':r['call_id'],'facts':r.get('serving_facts',[]) if r['status']=='ok' else []} for r in selected}
    evaluator=load_module('qwen35_paired_evaluator',root/'evaluation/case_metrics.py')
    gold={c['call_id']:c for c in calls}
    result={'metrics':evaluator.evaluate(gold,preds),'n_calls':len(calls),
        'status_counts':dict(Counter(r['status'] for r in selected)),
        'serving_metrics_without_de_normalizers':evaluator.evaluate(gold,serving),
        'failed_generations':sum(r['status']!='ok' for r in selected),
        'generation_reused_from':generated_dir.name+'; identical raw outputs, no second generation'}
    out.mkdir(exist_ok=True)
    save(out/'metrics.json',result)
    for r in selected:
        save(out/'records'/(__import__('hashlib').sha256(r['call_id'].encode()).hexdigest()[:20]+'.json'),r)
    (out/'predictions.jsonl').write_text(''.join(json.dumps(p,ensure_ascii=False)+'\n' for p in preds.values()),encoding='utf-8')
    return result


def phase_data(root, phase):
    selection=json.loads((root/'selection.json').read_text(encoding='utf-8'))
    train,dev=read(root/'data/train_ready.jsonl'),read(root/'data/dev_ready.jsonl')
    by={r['call_id']:r for r in dev}
    return train,[by[c] for c in selection['full_eval32_ids']],read(root/'sft/train.jsonl'),read(root/'sft/dev.jsonl')



def evaluate_saved(root, run, adapter_name, precision, phase, all_dev=False):
    from peft import PeftModel
    if adapter_name not in ('last','best'): raise ValueError('Expected last or best')
    pointer=json.loads((run/(adapter_name+'_pointer.json')).read_text(encoding='utf-8'))
    snap=run/pointer['relative_path']
    verify_snapshot(snap)
    base,tokenizer=load_base(root,precision)
    model=PeftModel.from_pretrained(base,snap/'adapter').eval()
    calls=read(root/'data/dev_ready.jsonl') if all_dev else phase_data(root,phase)[1]
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    report=quality_run(model,tokenizer,calls,root,run/f'recheck_{adapter_name}_{len(calls)}_{stamp}',max_new_tokens=2048)
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)


def verify_snapshot(path):
    path=Path(path).resolve()
    values=json.loads((path/'snapshot_hashes.json').read_text(encoding='utf-8'))
    for rel,expected in values.items():
        p=(path/rel).resolve()
        if not p.is_relative_to(path) or sha(p)!=expected:raise ValueError('Checkpoint integrity failure: '+rel)


def selection_key(report,baseline,initial=False):
    m,b=report['metrics'],baseline['metrics']
    eligible=initial or (all(m.get(k) is not None and b.get(k) is not None and m[k]>=b[k]-.02
        for k in ('fact_recall','temporal_recall','state_accuracy'))
        and report['failed_generations']<=baseline['failed_generations'])
    def value(k):return m.get(k) if m.get(k) is not None else -1.
    p,r=value('fact_precision'),value('fact_recall')
    f1=2*p*r/(p+r) if p+r>0 else 0.
    return (int(eligible),min(value('temporal_recall'),value('state_accuracy')),
        f1,r,value('grounding_turn'),-report['failed_generations'])


def validate_resume_state(state,dataset_sha,phase,goal,learning_rate,ntrain):
    if (state['dataset_sha256']!=dataset_sha or state['phase']!=phase or state['goal']!=goal
            or state['learning_rate']!=learning_rate):
        raise ValueError('Resume dataset/phase/schedule/config mismatch')
    if type(state['step']) is not int or not 0<=state['step']<goal:
        raise ValueError('Resume step already complete or invalid')
    order=state['order']
    if 'grad_scaler' not in state or not isinstance(state['grad_scaler'],dict):
        raise ValueError('Resume FP16 GradScaler state missing')
    if len(order)!=ntrain or any(type(i) is not int for i in order) or set(order)!=set(range(ntrain)):
        raise ValueError('Resume training order is not a permutation of this dataset')


def update_multiplier(step,total):
    warmup=max(1,math.ceil(total*.05))
    return min((step+1)/warmup,max(0.,(total-step)/max(1,total-warmup)))


def run_training(args):
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU is required; do not train on CPU silently')
    root, out = args.root.resolve(), args.out.resolve()
    manifest = verify_payload(root)
    raw_train,raw_dev,sft_train,sft_dev=phase_data(root,args.phase)
    ntrain=len(raw_train); goal=math.ceil(ntrain/4)
    resume_state=None
    if args.resume:
        verify_snapshot(args.resume)
        resume_state=torch.load(args.resume/'training_state.pt',map_location='cpu',weights_only=True)
        validate_resume_state(resume_state,sha(root/'manifest.json'),args.phase,goal,args.learning_rate,ntrain)
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
        'context': 24576, 'max_new_tokens': 2048, 'thinking': False,
        'sampling': False, 'cache': True, 'train_calls': ntrain, 'dev_calls': len(sft_dev), 'final_eval_calls':len(raw_dev), 'phase':args.phase,
        'baseline_calls': 32, 'checkpoint_selection_calls': 32,
        'selection_policy': 'Fixed inner-dev32; reject >0.02 declines in recall/temporal/state or more structural failures against segment start. Prioritise min(temporal,state), then F1 and recall. Internal selection, not official gates.',
        'dataset_manifest_sha256': sha(root/'manifest.json'),
        'parent_dataset_manifest_sha256': manifest['parent_manifest_sha256'],
        'dependencies': {p: importlib.metadata.version(p) for p in ('torch','transformers','peft','accelerate','tokenizers','safetensors')},
        'hardware': {'gpu':torch.cuda.get_device_name(0),'vram_bytes':torch.cuda.get_device_properties(0).total_memory,'cuda':torch.version.cuda},
        'source_sha256': {str(p.relative_to(root)).replace('\\','/'):sha(p) for p in sorted((root/'ml/qwen').glob('*.py'))},
        'full_official_val_run': False, 'experiment_not_a_claim_of_case_gate_pass': True}
    save(out/'training_config.json', config)
    random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    tokenize = load_module('qwen35_completion_tokenizer', root/'tokenize_sft.py')
    base, tokenizer = load_base(root, args.precision)
    encoded = [tokenize.encode_record(effective_example(r), tokenizer, max_length=24576)
               for r in sft_train]
    dev_encoded = [tokenize.encode_record(effective_example(r), tokenizer, max_length=24576)
                   for r in sft_dev]
    for row in encoded+dev_encoded:
        row['answer_start']=next(i for i,v in enumerate(row['labels']) if v!=-100)
        row['masked_date_tokens']=0
    assert len(encoded)==ntrain and len(dev_encoded)==len(sft_dev)
    assert max(len(r['input_ids'])-r['answer_start'] for r in encoded+dev_encoded)<2048
    assert max(r['answer_start'] for r in encoded+dev_encoded)+2048<=24576
    # Explicit selectors: no adapters on embeddings/lm_head or another architecture.
    targets = select_language_targets(
        [name for name, module in base.named_modules()
         if isinstance(module, torch.nn.Linear) and '.layers.' in name],
        json.loads((root/'expected_lora_targets.json').read_text(encoding='utf-8')))
    from ml.qwen.train import validate_lora_targets
    adapter_path=args.resume/'adapter' if args.resume else root/'initial_adapter'
    adapter_config=json.loads((adapter_path/'adapter_config.json').read_text(encoding='utf-8'))
    if adapter_config['base_model_name_or_path']!=MODEL or adapter_config['r']!=16 or adapter_config['lora_alpha']!=32:
        raise ValueError('Initial/resume adapter does not match base/layout')
    validate_lora_targets([n for n,m in base.named_modules() if isinstance(m,torch.nn.Linear)],targets,adapter_config['target_modules'])
    model=PeftModel.from_pretrained(base,adapter_path,is_trainable=True)
    del base
    for p in model.parameters():
        if p.requires_grad: p.data = p.data.float()
    config.update(total_parameters=sum(p.numel() for p in model.parameters()),
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        target_modules=targets, token_lengths={'train_max':max(len(r['input_ids']) for r in encoded),
        'train_median':statistics.median(len(r['input_ids']) for r in encoded),
        'answer_max':max(len(r['input_ids'])-r['answer_start'] for r in encoded+dev_encoded),
        'date_tokens_excluded_from_loss':sum(r['masked_date_tokens'] for r in encoded),
        'non_contiguous_assistant_mask':False,'date_loss_policy':'all_v2_dates_supervised'})
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
    # Generate the fixed 32 before training; all32 are inner selection data, not final test.
    baseline32=quality_run(model,tokenizer,raw_dev,root,out/'before_dev32',max_new_tokens=2048)
    baseline=paired_view(raw_dev[:8],out/'before_dev32',root,out/'before_dev8')
    if STOP_REQUESTED: raise KeyboardInterrupt('Stop requested before optimizer loop')
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.learning_rate, weight_decay=0.)
    scaler = torch.amp.GradScaler('cuda', enabled=args.precision=='float16', init_scale=32.)
    order=json.loads((root/'selection.json').read_text(encoding='utf-8'))['train_order_indices']
    assert sorted(order)==list(range(ntrain))
    logs,best_key,best_step,last_saved_step=[],None,None,0
    start_step=0
    if resume_state:
        optimizer.load_state_dict(resume_state['optimizer'])
        scaler.load_state_dict(resume_state['grad_scaler'])
        start_step=resume_state['step'];order=resume_state['order']
        if not 0<=start_step<goal:raise ValueError('Checkpoint already completed, or invalid step')
        torch.set_rng_state(resume_state['torch_rng']);torch.cuda.set_rng_state_all(resume_state['cuda_rng'])
        random.setstate(resume_state['python_rng'])
    def checkpoint(model,tokenizer,path,root,step,probe):
        snap=out/'checkpoints'/f'step{step:06d}'
        if not snap.exists():
            base_checkpoint(model,tokenizer,snap,root,step,probe)
            state={'optimizer':optimizer.state_dict(),'grad_scaler':scaler.state_dict(),'step':step,'order':order,'goal':goal,'phase':args.phase,
                'learning_rate':args.learning_rate,'dataset_sha256':sha(root/'manifest.json'),
                'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'python_rng':random.getstate()}
            torch.save(state,snap/'training_state.pt')
            hashes={str(p.relative_to(snap)).replace('\\','/'):sha(p) for p in sorted(snap.rglob('*')) if p.is_file()}
            save(snap/'snapshot_hashes.json',hashes)
        verify_snapshot(snap)
        save(out/(path.name+'_pointer.json'),{'relative_path':str(snap.relative_to(out)),'step':step})
    def eligible_key(measured):
        return selection_key(measured,baseline32)
    best_key=selection_key(baseline32,baseline32,initial=True);best_step=start_step
    checkpoint(model,tokenizer,out/'best',root,start_step,encoded[0])
    checkpoint(model,tokenizer,out/'last',root,start_step,encoded[0])
    save(out/'initial_selection.json',{'step':start_step,'key':best_key,'metrics':baseline32,
        'policy':'Inner-dev32 heuristic; official case gates are not evaluated'})
    first_b_name, first_b = next((n,p) for n,p in model.named_parameters() if p.requires_grad and 'lora_B' in n)
    initial_b = first_b.detach().clone()
    loop_start = time.perf_counter(); reason = 'schedule_completed'
    for step in range(start_step,goal):
        if STOP_REQUESTED or time.perf_counter()-loop_start >= args.budget_hours*3600:
            reason = 'user_stop' if STOP_REQUESTED else 'training_budget_reached'; break
        model.train(); optimizer.zero_grad(set_to_none=True); torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize(); t0 = time.perf_counter()
        batch = [encoded[i] for i in order[step*4:(step+1)*4]]
        multiplier = update_multiplier(step,goal)
        for group in optimizer.param_groups: group['lr'] = args.learning_rate*multiplier
        weights = token_loss_weights(batch); values=[]
        for row,w in zip(batch,weights):
            with autocast_for(model): loss = suffix_loss(model,row)
            if not torch.isfinite(loss): raise RuntimeError('Non-finite loss; stopping before update')
            values.append(float(loss.detach())); scaler.scale(loss*w).backward()
        scaler.unscale_(optimizer); norm = torch.nn.utils.clip_grad_norm_(params,1.)
        if not torch.isfinite(norm):
            raise RuntimeError('Non-finite gradient; no optimizer update. Return logs; no silent precision switch')
        old_scale=scaler.get_scale(); scaler.step(optimizer); scaler.update()
        if scaler.get_scale() < old_scale: raise RuntimeError('Unexpected skipped optimizer update')
        torch.cuda.synchronize()
        item={'step':step+1,'examples_seen':min((step+1)*4,len(order)),'batch_calls':len(batch),'supervised_tokens':sum(supervised_tokens(r) for r in batch),
            'loss':sum(x*w for x,w in zip(values,weights)),'gradient_norm':float(norm),
            'learning_rate':optimizer.param_groups[0]['lr'],'elapsed_sec':time.perf_counter()-t0,
            'peak_cuda_allocated_mib':torch.cuda.max_memory_allocated()/1024**2,
            'peak_cuda_reserved_mib':torch.cuda.max_memory_reserved()/1024**2,'optimizer_step_skipped':False}
        logs.append(item); save(out/'training_log.json',logs)
        print('step',step+1,'/',goal,'calls',item['examples_seen'],'loss',round(item['loss'],5),'sec',round(item['elapsed_sec'],2),flush=True)
        if len(logs) == 3:
            changed=not torch.equal(initial_b,first_b.detach())
            assert changed, 'LoRA weights did not change'
            save(out/'first_updates.json',{'passed':True,'finite_updates':3,'weight_changed':changed,'parameter':first_b_name})
            del initial_b
            remaining = (goal-step-1)*statistics.median(r['elapsed_sec'] for r in logs)
            print('TRAINING_ESTIMATE: remaining optimizer time (no evaluation):', round(remaining/3600,2), 'hours', flush=True)
        if step+1 in (16,96) or (step+1)%64==0 or step+1==goal:
            checkpoint(model,tokenizer,out/'last',root,step+1,encoded[0]);last_saved_step=step+1
        if step+1==16:
            early=quality_run(model,tokenizer,raw_dev[:8],root,out/'early_dev8_step16',max_new_tokens=2048)
            save(out/'early_check.json',early)
            if early['failed_generations']==8:
                reason='early_all_structural_failures';break
        if step+1 in (96,384) or step+1==goal:
            measured=quality_run(model,tokenizer,raw_dev,root,out/f'selection_step{step+1}',max_new_tokens=2048)
            key=eligible_key(measured)
            if key>best_key:
                best_key,best_step=key,step+1
                checkpoint(model,tokenizer,out/'best',root,step+1,encoded[0])
            save(out/'progress.json',{'step':step+1,'best_step':best_step,'selection':measured})
    steps=start_step+len(logs)
    if not logs: raise RuntimeError('No optimizer updates completed in this segment')
    if last_saved_step != steps: checkpoint(model,tokenizer,out/'last',root,steps,encoded[0])
    save(out/'training_complete.json',{'steps':steps,'examples_seen':min(steps*4,len(order)),
        'schedule_completed':steps==goal,'stop_reason':reason,'elapsed_loop_sec':time.perf_counter()-loop_start})
    if not STOP_REQUESTED and not (out/f'selection_step{steps}'/'metrics.json').exists():
        measured=quality_run(model,tokenizer,raw_dev,root,out/f'selection_step{steps}',max_new_tokens=2048)
        key=eligible_key(measured)
        if key>best_key:
            best_key,best_step=key,steps
            checkpoint(model,tokenizer,out/'best',root,steps,encoded[0])
    if STOP_REQUESTED:
        save(out/'experiment_report.json',{'status':'stopped_after_checkpoint','training_steps':steps,
            'training_calls_seen':min(steps*4,len(order)),'schedule_completed':steps==goal,'stop_reason':'user_stop',
            'after_quality_measured':False,'full_official_val_run':False,'official_case_gate_pass':None})
        print('Stopped after saving last checkpoint; final generation was not started.',flush=True)
        return
    # Fresh base+saved adapter, not reusing an in-memory model.
    selected='best' if best_step is not None else 'last'; selected_step=best_step if best_step is not None else steps
    del model,optimizer,params,first_b,loss; gc.collect(); torch.cuda.empty_cache()
    fresh,_=load_base(root,args.precision)
    model=PeftModel.from_pretrained(fresh,out/json.loads((out/(selected+'_pointer.json')).read_text())['relative_path']/'adapter').eval()
    reference=torch.load(out/json.loads((out/(selected+'_pointer.json')).read_text())['relative_path']/'reload_reference.pt',weights_only=True,map_location='cpu')
    x=reference['input_ids'].to('cuda')
    with torch.no_grad(),autocast_for(model):
        actual=model(input_ids=x,attention_mask=torch.ones_like(x),use_cache=False,logits_to_keep=1).logits.float().cpu()
    torch.testing.assert_close(reference['logits'],actual,atol=.003,rtol=.003)
    reload={'passed':True,'fresh_base_and_saved_adapter':True,'max_logit_abs_difference':float((actual-reference['logits']).abs().max())}
    save(out/'reload_test.json',reload)
    after=quality_run(model,tokenizer,raw_dev,root,out/'after_dev',max_new_tokens=2048)
    after_paired=paired_view(raw_dev[:8],out/'after_dev',root,out/'after_dev8_same_ids')
    train_check=quality_run(model,tokenizer,raw_train[:4],root,out/'after_train4_diagnostic',max_new_tokens=2048)
    export=out/'export'; shutil.copytree(out/json.loads((out/(selected+'_pointer.json')).read_text())['relative_path'],export,dirs_exist_ok=True,ignore=shutil.ignore_patterns('training_state.pt','snapshot_hashes.json'))
    bundle_hashes={str(p.relative_to(export)).replace('\\','/'):sha(p) for p in sorted(export.rglob('*')) if p.is_file()}
    descriptor={'base_model':MODEL,'base_revision':REVISION,'dtype':args.precision,
        'total_parameters':config['total_parameters'],'trainable_parameters':config['trainable_parameters'],
        'context':24576,'max_new_tokens':2048,'enable_thinking':False,'do_sample':False,
        'schema':'src.contracts.models.CandidateBatch via ml.qwen.bridge.parse_candidates',
        'selected_step':selected_step,'reload_verified':True,'sha256':bundle_hashes,
        'artifact_state':'research_candidate_reload_verified','official_case_gates_verified':False,
        'serving_acceptance':'pending DE normalizers and FS integration; no automatic production factory claim'}
    save(export/'model_manifest.json',descriptor)
    last_step=json.loads((out/'last_pointer.json').read_text(encoding='utf-8'))['step']
    after_last=after
    if last_step!=selected_step:
        del model,fresh;gc.collect();torch.cuda.empty_cache()
        last_base,_=load_base(root,args.precision)
        model=PeftModel.from_pretrained(last_base,out/json.loads((out/'last_pointer.json').read_text())['relative_path']/'adapter').eval()
        after_last=quality_run(model,tokenizer,raw_dev,root,out/'after_last_dev32',max_new_tokens=2048)
    save(out/'candidate_comparison.json',{'before':baseline32,'selected_best':after,'last':after_last,
        'selected_best_step':selected_step,'last_step':last_step,'scope':'Same32 inner-dev used for model selection; official383 not tested'})
    report={'status':'completed','experiment':'v2_remaining2863_last128_qwen3_06b','training_steps':steps,
        'training_calls_seen':min(steps*4,len(order)),'schedule_completed':steps==goal,'stop_reason':reason,
        'selected_checkpoint':selected,'selected_step':selected_step,'selected_starting_adapter':selected_step==start_step,'additional_training_improved_selection_key':selected_step!=start_step,'base_model':MODEL,'base_revision':REVISION,
        'dtype':args.precision,'reload':reload,'initialization':'resume_adapter_optimizer' if args.resume else 'last128_adapter_new_optimizer','resume_from':str(args.resume) if args.resume else None,'baseline_scope':'start_of_this_segment; not plain base when resumed','before_dev32':baseline32,'before_dev8':baseline,'after_dev8_same_ids':after_paired,
        'after_dev':after,'after_last_dev32':after_last,'after_train4_diagnostic':train_check,
        'seconds_per_microbatch_median':statistics.median(r['elapsed_sec']/r['batch_calls'] for r in logs[-20:]),
        'elapsed_total_sec':time.perf_counter()-started,'full_official_val_run':False,
        'official_case_gate_pass':None,'limitations':'Only the declared v2 inner-dev subset; 32 calls used for selection;  temporal denominator is matched facts. No end-to-end DE/HTTP acceptance.'}
    save(out/'experiment_report.json',report)
    config['status']='completed';save(out/'training_config.json',config)
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--precision',choices=('float16',),default='float16')
    ap.add_argument('--learning-rate',type=float,default=5e-5)
    ap.add_argument('--budget-hours',type=float,default=4.)
    ap.add_argument('--phase',choices=('full',),default='full')
    ap.add_argument('--resume',type=Path)
    ap.add_argument('--eval-all-dev',action='store_true')
    ap.add_argument('--recheck',choices=('best','last'))
    args=ap.parse_args()
    if not 0 < args.budget_hours <= 24 or not 0 < args.learning_rate <= .001:
        ap.error('Expected 0 < budget_hours <= 24 and 0 < learning_rate <= 0.001')
    if args.recheck:
        verify_payload(args.root.resolve())
        evaluate_saved(args.root,args.out,args.recheck,args.precision,args.phase,args.eval_all_dev)
        return
    existed=args.out.exists() and any(args.out.iterdir())
    try: run_training(args)
    except BaseException as exc:
        if not existed:
            save(args.out/'failure.json',{'status':'failed','error_type':type(exc).__name__,
                'message':str(exc),'quality_claimed':False,'note':'Return the existing archive/logs; do not overwrite this run.'})
        raise


if __name__=='__main__': main()
