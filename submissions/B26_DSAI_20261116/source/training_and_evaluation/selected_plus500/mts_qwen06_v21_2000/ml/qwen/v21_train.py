"""One bounded 2000-call warm-start from step648. GPU results are created only by actual execution."""
from pathlib import Path
from collections import Counter
from datetime import datetime,timezone
import argparse,gc,hashlib,importlib.metadata,json,math,os,random,shutil,signal,statistics,time,traceback
from ml.qwen.train import read,save,sha,load_module,autocast_for,token_loss_weights,supervised_tokens,validate_lora_targets
from ml.qwen.budget_v21 import BudgetStop,load_budget,verify_snapshot,validate_resume,score_key,midpoint_affordable,checkpoint_paths_to_prune
from ml.qwen.v21_runtime import MODEL,REVISION,Engine
from ml.qwen.v21_bridge import process_output
from ml.qwen.bridge import messages,model_input
from src.contracts.models import CallInput
from src.pipeline.adapters import PipelineError,ModelDescriptor
from src.pipeline.service import Pipeline
from tokenize_sft import encode_record,suffix_loss,loss_equivalence

STOP=False
def stop_handler(_sig,_frame):
    global STOP;STOP=True

def verify_payload(root):
    manifest=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    for rel,want in manifest['sha256'].items():
        p=(root/rel).resolve()
        if not p.is_relative_to(root.resolve()) or sha(p)!=want:raise ValueError('Payload checksum mismatch: '+rel)
    if manifest['base_model']!=MODEL or manifest['base_revision']!=REVISION or manifest['initial_step']!=648:raise ValueError('Wrong initial model')
    return manifest

def input_call(raw):return CallInput.model_validate({k:raw[k] for k in ('call_id','started_at','turns')})
def fingerprint(v):return hashlib.sha256(json.dumps(v,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def evaluate(model,tokenizer,calls,root,out,budget,adapter_hash,*,deadline=None):
    """One generation -> raw / normalized / serving views. Failed generation stays failed/null."""
    import torch
    instruction=(root/'prompt.txt').read_text(encoding='utf-8');catalog=json.loads((root/'official_catalog.json').read_text(encoding='utf-8'))
    deadline=min(deadline or budget.finish_deadline-600,budget.finish_deadline-600)
    engine=Engine(model,tokenizer,instruction,catalog,deadline_epoch=deadline,max_call_seconds=120)
    meta={'adapter_sha256':adapter_hash,'prompt_sha256':sha(root/'prompt.txt'),'tokenizer_sha256':sha(root/'tokenizer/tokenizer.json'),
          'base_revision':REVISION,'max_new_tokens':2048,'max_call_seconds':120,'thinking':False,'sampling':False,'context':24576,
          'call_ids':[c['call_id'] for c in calls],'input_sha256':{c['call_id']:fingerprint(model_input(input_call(c))) for c in calls}}
    out.mkdir(parents=True,exist_ok=True)
    if (out/'generation_manifest.json').exists():
        if json.loads((out/'generation_manifest.json').read_text(encoding='utf-8'))!=meta:raise ValueError('Refuse cached generations with different weights/prompt/inputs/decoder')
    else:save(out/'generation_manifest.json',meta)
    records=[]
    for raw_call in calls:
        call=input_call(raw_call);path=out/'records'/(hashlib.sha256(call.call_id.encode()).hexdigest()[:20]+'.json')
        if path.is_file():
            record=json.loads(path.read_text(encoding='utf-8'))
            if record['call_id']!=call.call_id or record['input_sha256']!=meta['input_sha256'][call.call_id]:raise ValueError('Cached input mismatch')
            records.append(record);continue
        if STOP or time.time()+125>deadline:break
        record={'call_id':call.call_id,'input_sha256':meta['input_sha256'][call.call_id],'status':'failed','raw':None,'facts':None,
                'raw_facts':None,'raw_error_code':None,'error_code':None,'normalizations':[]}
        start=time.perf_counter();torch.cuda.synchronize()
        try:
            record['raw']=engine.generate(call)
            try:
                unnormalized,_,_=process_output(record['raw'],call,normalize=False)
                record['raw_facts']=[c.fact.model_dump(mode='json') for c in unnormalized.candidates]
            except PipelineError as exc:record['raw_error_code']=exc.code
            batch,changes,_=process_output(record['raw'],call)
            record.update(status='ok',facts=[c.fact.model_dump(mode='json') for c in batch.candidates],normalizations=changes)
            class Measured:
                descriptor=ModelDescriptor('v21-measured','real');state='ready'
                def extract(self,_call):return batch
            served,warnings=Pipeline(Measured()).run(call)
            record['serving_facts']=[f.model_dump(mode='json') for f in served.facts]
            record['serving_warnings']=list(warnings)
            record['unresolved_agreements']=sum(c.fact.attribute=='agreement' and c.confirmation!='confirmed' for c in batch.candidates)
        except PipelineError as exc:record.update(status='failed',error_code=exc.code,facts=None)
        except Exception as exc:record.update(status='failed',error_code='UNEXPECTED_'+type(exc).__name__,facts=None)
        record['raw']=record['raw'] or engine.last_raw
        torch.cuda.synchronize();record['elapsed_sec']=time.perf_counter()-start;record['generation']=engine.last_diagnostics
        if record['status']=='failed':assert record['facts'] is None
        save(path,record);records.append(record)
        print('evaluation',call.call_id,record['status'],'facts',len(record['facts'] or []),'error',record['error_code'],'seconds',round(record['elapsed_sec'],2),flush=True)
    completed={r['call_id'] for r in records};gold={c['call_id']:c for c in calls if c['call_id'] in completed}
    evaluator=load_module('official_eval_'+out.name,root/'evaluation/case_metrics.py')
    evaluator.load_catalog(str(root/'evaluation/catalog.json'))
    views={};error=None
    for name,key in [('normalized','facts'),('raw','raw_facts'),('serving','serving_facts')]:
        preds={r['call_id']:{'call_id':r['call_id'],'facts':r.get(key) or []} for r in records}
        try:views[name]=evaluator.evaluate(gold,preds)
        except Exception as exc:error={'type':type(exc).__name__,'message':str(exc),'official_evaluator_not_modified':True};views[name]=None
        (out/(name+'_predictions.jsonl')).write_text(''.join(json.dumps(p,ensure_ascii=False)+'\n' for p in preds.values()),encoding='utf-8')
    primary=views['normalized'];counts={}
    if primary:
        counts={'matched_facts':primary['n_grounded'],'temporal_denominator':primary['n_temporal'],
                'temporal_correct':round(primary['temporal_recall']*primary['n_temporal']) if primary['temporal_recall'] is not None else None,
                'state_denominator':primary['n_state'],'state_correct':round(primary['state_accuracy']*primary['n_state']) if primary['state_accuracy'] is not None else None}
    result={'complete':len(records)==len(calls) and error is None,'planned_calls':len(calls),'completed_calls':len(records),
        'completed_ids':[r['call_id'] for r in records],'pending_ids':[c['call_id'] for c in calls if c['call_id'] not in completed],
        'metrics':primary,'views':views,'counts':counts,'failed_generations':sum(r['status']=='failed' for r in records),
        'status_counts':dict(Counter(r['status'] for r in records)),'valid_empty_answers':sum(r['status']=='ok' and not r['facts'] for r in records),
        'metric_error':error,'elapsed_generation_seconds':sum(r.get('elapsed_sec',0) for r in records),
        'scope':'Previously used inner-dev. Identical actual generations for raw/normalizer/serving. Not officialval383 or a case acceptance claim.'}
    from ml.qwen.error_analysis_v21 import analyze
    primary_preds={r['call_id']:{'facts':r['facts'] or []} for r in records}
    save(out/'error_analysis.json',analyze(evaluator,gold,primary_preds))
    save(out/'metrics.json',result);return result

def load_model(root,adapter):
    import torch
    from transformers import AutoTokenizer,AutoModelForCausalLM
    from peft import PeftModel
    tok=AutoTokenizer.from_pretrained(root/'tokenizer',local_files_only=True)
    base=AutoModelForCausalLM.from_pretrained(MODEL,revision=REVISION,dtype=torch.float16,attn_implementation='sdpa',trust_remote_code=False).to('cuda')
    if base.config.model_type!='qwen3' or base.config.eos_token_id!=tok.eos_token_id:raise ValueError('Base/tokenizer mismatch')
    expected=json.loads((root/'expected_lora_targets.json').read_text(encoding='utf-8'))
    actual=[n for n,m in base.named_modules() if isinstance(m,torch.nn.Linear) and '.layers.' in n]
    if set(actual)!=set(expected) or len(actual)!=196:raise ValueError('Unexpected196-module LoRA layout')
    cfg=json.loads((adapter/'adapter_config.json').read_text(encoding='utf-8'))
    if cfg['base_model_name_or_path']!=MODEL or cfg['r']!=16 or cfg['lora_alpha']!=32 or cfg['lora_dropout']!=0:raise ValueError('Wrong adapter base/rank/dropout')
    validate_lora_targets([n for n,m in base.named_modules() if isinstance(m,torch.nn.Linear)],expected,cfg['target_modules'])
    model=PeftModel.from_pretrained(base,adapter,is_trainable=True);del base
    for p in model.parameters():
        if p.requires_grad:p.data=p.data.float()
    if sum(p.numel() for p in model.parameters() if p.requires_grad)!=10092544:raise ValueError('Unexpected trainable parameter count')
    return model,tok

def run(args):
    import torch
    root=args.root.resolve();out=args.run.resolve();budget=load_budget(args.budget_file)
    manifest=verify_payload(root);budget.require(20*60)
    checks=json.loads((out/'input_checks.json').read_text(encoding='utf-8'))
    if not checks.get('all_checks_passed') or not checks.get('native_colab_tokenizer_audit_performed') or checks.get('payload_manifest_sha256')!=sha(root/'manifest.json'):
        raise ValueError('Run notebook cell3: native full input audit required for this exact payload')
    if not torch.cuda.is_available():raise RuntimeError('CUDA GPU required; no silent CPU training fallback')
    if tuple(map(int,torch.__version__.split('+')[0].split('.')[:2]))<(2,6):raise RuntimeError('Need Colab CUDA PyTorch>=2.6; do not replace it blindly')
    signal.signal(signal.SIGTERM,stop_handler);signal.signal(signal.SIGINT,stop_handler)
    out.mkdir(parents=True,exist_ok=True)
    if (out/'experiment_report.json').exists() and not args.resume_checkpoint:raise ValueError('Existing run report; do not rerun training blindly')
    raw_train=read(root/'data/train2000.jsonl');dev=read(root/'data/dev32_gold_official.jsonl');sft=read(root/'sft/train.jsonl')
    goal=math.ceil(len(raw_train)/4);order=list(range(len(sft)));manifest_sha=sha(root/'manifest.json');lr=2e-5
    if len(sft)!=2000 or len(raw_train)!=2000:raise ValueError('Expected2000 train rows')
    resume=None
    if args.resume_checkpoint:
        snap=verify_snapshot(args.resume_checkpoint)
        if not snap.is_relative_to(out/'checkpoints'):raise ValueError('Resume must belong to this run; old step648 optimizer is not compatible')
        resume=torch.load(snap/'training_state.pt',weights_only=True,map_location='cpu')
        validate_resume(resume,manifest_sha,goal,lr,order,budget)
        for name in ['failure.json','experiment_report.json','launcher_error.json']:
            prior=out/name
            if prior.exists():
                history=out/'report_history'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');history.mkdir(parents=True,exist_ok=True)
                shutil.move(str(prior),str(history/name))
    config={'base_model':MODEL,'base_revision':REVISION,'source_step':648,'initial_adapter_sha256':manifest['initial_adapter_sha256'],
            'train_calls':len(sft),'goal_new_updates':goal,'accumulation':4,'seed':42,'learning_rate':lr,'epochs':1,
            'dtype':'float16_base_float32_adapter','context':24576,'max_new_tokens':2048,'thinking':False,'sampling':False,
            'budget':budget.as_dict(),'manifest_sha256':manifest_sha,'date_policy':manifest['date_policy'],
            'dependencies':{n:importlib.metadata.version(n) for n in ['torch','transformers','peft','tokenizers','accelerate','safetensors','pydantic']},
            'hardware':{'gpu':torch.cuda.get_device_name(0),'vram_bytes':torch.cuda.get_device_properties(0).total_memory,'cuda':torch.version.cuda},
            'official_val_run':False,'official_case_gate_pass':None,'selection':'Same inner-dev32, raw/normalizer/serving separated; stage0 fallback.'}
    if resume and config['dependencies']!=resume['dependencies']:raise ValueError('Exact resume dependency versions changed')
    save(out/'training_config.json',config);shutil.copyfile(root/'manifest.json',out/'payload_manifest.json')
    source=out/'source';source.mkdir(exist_ok=True)
    for folder in ['ml','src']:
        shutil.copytree(root/folder,source/folder,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for name in ['tokenize_sft.py','check_inputs.py','session_control.py','requirements.txt','expected_lora_targets.json','selection.json','prompt.txt','official_catalog.json','manifest.json']:
        shutil.copyfile(root/name,source/name)
    shutil.copytree(root/'audit',source/'audit',dirs_exist_ok=True)
    save(source/'source_manifest.json',{'sha256':{str(p.relative_to(source)).replace('\\','/'):sha(p) for p in source.rglob('*') if p.is_file() and p.name!='source_manifest.json'},
        'dataset_location':'Original released payload; exact hashes in manifest.json. No dataset is reconstructed from predictions.'})
    random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    model,tok=load_model(root,args.resume_checkpoint/'adapter' if resume else root/'initial_adapter')
    encoded=[encode_record(r,tok) for r in sft]
    # Native Colab tokenizer must satisfy every chat boundary, mask and exact target.
    if max(len(r['input_ids'])-r['answer_start'] for r in encoded)>=2048 or max(r['answer_start'] for r in encoded)+2048>24576:raise ValueError('Context/generation policy insufficient; no truncation')
    config.update(total_parameters=sum(p.numel() for p in model.parameters()),trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        token_audit={'all2000_native_prefixes_checked':True,'max_tokens':max(len(r['input_ids']) for r in encoded),
            'masked_timestamp_tokens':sum(r['masked_date_tokens'] for r in encoded),'supervised_tokens':sum(supervised_tokens(r) for r in encoded)})
    save(out/'training_config.json',config)
    parameters=[p for p in model.parameters() if p.requires_grad];optimizer=torch.optim.AdamW(parameters,lr=lr,weight_decay=0.)
    scaler=torch.amp.GradScaler('cuda',enabled=True,init_scale=32.)
    step=resume['new_updates'] if resume else 0;logs=json.loads((out/'training_log.json').read_text()) if (out/'training_log.json').is_file() else []
    logs=[r for r in logs if r['new_update']<=step]
    if resume:
        optimizer.load_state_dict(resume['optimizer']);scaler.load_state_dict(resume['grad_scaler'])
        torch.set_rng_state(resume['torch_rng']);torch.cuda.set_rng_state_all(resume['cuda_rng']);random.setstate(resume['python_rng'])
    best_step=0;best_key=None
    if (out/'selection_state.json').is_file():
        old=json.loads((out/'selection_state.json').read_text());best_step=old['best_new_update'];best_key=tuple(old['key'])
    last_rng=None;inside_optimizer_step=False
    def capture_rng():return {'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'python_rng':random.getstate()}
    def snapshot(label,update):
        nonlocal last_rng
        snap=out/'checkpoints'/f'update{update:06d}'
        if not snap.exists():
            tmp=out/'checkpoints'/f'.update{update:06d}.pending'
            if tmp.exists():
                if not tmp.resolve().is_relative_to((out/'checkpoints').resolve()):raise ValueError('Unsafe snapshot staging path')
                shutil.rmtree(tmp)
            tmp.mkdir(parents=True);model.eval()
            model.save_pretrained(tmp/'adapter',safe_serialization=True,save_embedding_layers=False);tok.save_pretrained(tmp/'tokenizer')
            for name,dest in [('prompt.txt','prompt.txt'),('official_catalog.json','catalog.json')]:shutil.copyfile(root/name,tmp/dest)
            x=torch.tensor([encoded[0]['input_ids']],device='cuda')
            with torch.no_grad(),autocast_for(model):logits=model(input_ids=x,attention_mask=torch.ones_like(x),use_cache=False,logits_to_keep=1).logits.float().cpu()
            torch.save({'input_ids':x.cpu(),'logits':logits},tmp/'reload_reference.pt')
            state={'optimizer':optimizer.state_dict(),'grad_scaler':scaler.state_dict(),'new_updates':update,'order':order,
                'goal':goal,'learning_rate':lr,'manifest_sha256':manifest_sha,'budget_origin_epoch':budget.origin_epoch,
                'budget_total_minutes':budget.total_minutes,'dependencies':config['dependencies'],**(last_rng or capture_rng())}
            torch.save(state,tmp/'training_state.pt')
            save(tmp/'checkpoint.json',{'source_step':648,'new_updates':update,'initial_step_is_not_a_new_training_run':update==0})
            hashes={str(p.relative_to(tmp)).replace('\\','/'):sha(p) for p in sorted(tmp.rglob('*')) if p.is_file()}
            save(tmp/'snapshot_hashes.json',hashes);verify_snapshot(tmp);os.replace(tmp,snap)
        verify_snapshot(snap);save(out/(label+'_pointer.json'),{'relative_path':str(snap.relative_to(out)).replace('\\','/'),'source_step':648,'new_updates':update})
        for p in checkpoint_paths_to_prune(out):shutil.rmtree(p)
        return snap
    try:
        budget.require(10*60)
        if not resume:
            last_rng=capture_rng();snapshot('last',0);snapshot('best',0)
        initial=verify_snapshot(out/'checkpoints/update000000')
        if step>0 and not (out/'before_dev32/metrics.json').exists():raise ValueError('Cannot replace missing original baseline with resumed weights')
        save(out/'loss_equivalence.json',loss_equivalence(model,tok))
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});model.enable_input_require_grads();model.config.use_cache=False
        longest=max(encoded,key=lambda r:len(r['input_ids']));model.train();optimizer.zero_grad(set_to_none=True);torch.cuda.reset_peak_memory_stats()
        with autocast_for(model):probe=suffix_loss(model,longest)
        probe_scaler=torch.amp.GradScaler('cuda',enabled=True,init_scale=32.)
        probe_scaler.scale(probe).backward();probe_scaler.unscale_(optimizer)
        norms=[p.grad.detach().float().norm() for p in parameters if p.grad is not None]
        if not norms or not all(bool(torch.isfinite(n)) for n in norms):raise RuntimeError('Longest-call backward produced nonfinite gradients')
        optimizer.zero_grad(set_to_none=True)
        save(out/'long_call_probe.json',{'passed':True,'optimizer_update':False,'tokens':len(longest['input_ids']),
            'peak_cuda_allocated_mib':torch.cuda.max_memory_allocated()/1024**2});del probe,norms,probe_scaler
        if step==0:
            baseline=evaluate(model,tok,dev,root,out/'before_dev32',budget,sha(initial/'adapter/adapter_model.safetensors'),deadline=budget.train_deadline-15*60)
        else:baseline=json.loads((out/'before_dev32/metrics.json').read_text())
        if not baseline['complete']:raise BudgetStop('Baseline32 is partial; refusing incomparable main training')
        if best_key is None:best_key=score_key(baseline,baseline,initial=True)
        save(out/'selection_state.json',{'best_new_update':best_step,'key':best_key,'baseline_scope':'inner-dev32, not independent test'})
        first_b=next(p for n,p in model.named_parameters() if p.requires_grad and 'lora_B' in n);initial_b=first_b.detach().clone()
        if resume:
            torch.set_rng_state(resume['torch_rng']);torch.cuda.set_rng_state_all(resume['cuda_rng']);random.setstate(resume['python_rng'])
        last_save=time.time();reason='one_pass_completed';loop_start=time.time();last_rng=capture_rng()
        loss=None
        while step<goal:
            estimate=max(30.,statistics.median([r['elapsed_sec'] for r in logs[-20:]])*1.3) if logs else 60.
            if STOP or budget.left('train')<estimate+60:
                reason='user_stop' if STOP else 'shared_budget_training_cutoff';break
            model.train();optimizer.zero_grad(set_to_none=True);torch.cuda.reset_peak_memory_stats();t0=time.perf_counter()
            batch=[encoded[i] for i in order[step*4:(step+1)*4]];weights=token_loss_weights(batch)
            warmup=15;multiplier=min((step+1)/warmup,max(0.,(goal-step)/max(1,goal-warmup)))
            for group in optimizer.param_groups:group['lr']=lr*multiplier
            values=[]
            for row,w in zip(batch,weights):
                with autocast_for(model):loss=suffix_loss(model,row)
                if not bool(torch.isfinite(loss)):raise RuntimeError('Nonfinite training loss')
                values.append(float(loss.detach()));scaler.scale(loss*w).backward()
            scaler.unscale_(optimizer);norm=torch.nn.utils.clip_grad_norm_(parameters,1.)
            if not bool(torch.isfinite(norm)):raise RuntimeError('Nonfinite gradient; optimizer update not applied')
            oldscale=scaler.get_scale();inside_optimizer_step=True;scaler.step(optimizer);scaler.update();torch.cuda.synchronize()
            if scaler.get_scale()<oldscale:raise RuntimeError('Skipped AMP update; not counting it as training')
            inside_optimizer_step=False;step+=1;last_rng=capture_rng()
            item={'new_update':step,'source_step':648,'calls_seen':min(step*4,len(order)),'loss':sum(v*w for v,w in zip(values,weights)),
                'elapsed_sec':time.perf_counter()-t0,'learning_rate':optimizer.param_groups[0]['lr'],'gradient_norm':float(norm),
                'supervised_tokens':sum(supervised_tokens(r) for r in batch),'masked_date_tokens':sum(r['masked_date_tokens'] for r in batch),
                'peak_cuda_allocated_mib':torch.cuda.max_memory_allocated()/1024**2,'optimizer_step_skipped':False}
            logs.append(item);save(out/'training_log.json',logs);save(out/'budget_progress.json',budget.as_dict())
            print('update',step,'/',goal,'calls',item['calls_seen'],'loss',round(item['loss'],5),'sec',round(item['elapsed_sec'],2),'remaining_total_min',round(budget.left('hard')/60,1),flush=True)
            if step==16:
                if torch.equal(initial_b,first_b.detach()):raise RuntimeError('No adapter weight change after16 updates')
                save(out/'first_updates.json',{'passed':True,'new_updates':16,'adapter_weight_changed':True,
                    'remaining_updates_estimate_min':statistics.median(r['elapsed_sec'] for r in logs[-16:])*(goal-step)/60})
            if step in (16,goal) or step%32==0 or time.time()-last_save>=600:
                snapshot('last',step);last_save=time.time()
            if step==16:
                snap=snapshot('last',step);early=evaluate(model,tok,dev[:8],root,out/'early_update16_dev8',budget,sha(snap/'adapter/adapter_model.safetensors'),deadline=budget.train_deadline)
                if early['complete'] and early['failed_generations']==8:reason='pilot_all8_structural_failures';break
            if step==250:
                remaining=goal-step;step_sec=statistics.median(r['elapsed_sec'] for r in logs[-50:]);eval_sec=max(300.,baseline['elapsed_generation_seconds'])
                if midpoint_affordable(budget,remaining,step_sec,eval_sec):
                    snap=snapshot('last',step);measured=evaluate(model,tok,dev,root,out/'selection_update250',budget,sha(snap/'adapter/adapter_model.safetensors'),deadline=budget.train_deadline)
                    key=score_key(measured,baseline)
                    if key>best_key:best_key,best_step=key,step;snapshot('best',step)
                    save(out/'selection_state.json',{'best_new_update':best_step,'key':best_key})
                else:save(out/'midpoint_skipped.json',{'reason':'reserve full final evaluation and shared five-hour budget','new_update':step})
        last=snapshot('last',step)
        save(out/'training_complete.json',{'new_updates':step,'goal_new_updates':goal,'calls_seen':min(step*4,len(order)),
            'schedule_completed':step==goal,'stop_reason':reason,'loop_seconds':time.time()-loop_start,'budget':budget.as_dict()})
        if STOP:raise BudgetStop('Stopped by user after saving latest completed update')
        measured=evaluate(model,tok,dev,root,out/f'final_update{step:06d}',budget,sha(last/'adapter/adapter_model.safetensors'))
        key=score_key(measured,baseline)
        if key>best_key:best_key,best_step=key,step;snapshot('best',step)
        save(out/'selection_state.json',{'best_new_update':best_step,'key':best_key})
        selected=verify_snapshot(out/'checkpoints'/f'update{best_step:06d}')
        selected_report=baseline if best_step==0 else measured if best_step==step else json.loads((out/f'selection_update{best_step}/metrics.json').read_text())
        budget.require(6*60)
        # Release every old model/optimizer reference before a genuinely fresh load.
        del optimizer,parameters,model,first_b,initial_b,loss;gc.collect();torch.cuda.empty_cache()
        model,tok=load_model(root,selected/'adapter');model.eval()
        ref=torch.load(selected/'reload_reference.pt',weights_only=True,map_location='cpu');x=ref['input_ids'].to('cuda')
        with torch.no_grad(),autocast_for(model):actual=model(input_ids=x,attention_mask=torch.ones_like(x),use_cache=False,logits_to_keep=1).logits.float().cpu()
        torch.testing.assert_close(ref['logits'],actual,rtol=.003,atol=.003)
        reload_result={'passed':True,'fresh_base_and_saved_adapter':True,'max_logit_abs_difference':float((ref['logits']-actual).abs().max()),'selected_new_updates':best_step}
        save(out/'reload_test.json',reload_result)
        export=out/'export';export.mkdir(exist_ok=True)
        for folder in ['adapter','tokenizer']:shutil.copytree(selected/folder,export/folder,dirs_exist_ok=True)
        for name in ['prompt.txt','catalog.json']:shutil.copyfile(selected/name,export/name)
        for folder in ['ml','src']:shutil.copytree(root/folder,export/folder,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        hashes={str(p.relative_to(export)).replace('\\','/'):sha(p) for p in sorted(export.rglob('*')) if p.is_file()}
        descriptor={'model_id':'qwen06-v21-step648-plus'+str(best_step),'base_model':MODEL,'base_revision':REVISION,
            'dtype':'float16_base_float32_adapter','source_step':648,'selected_new_updates':best_step,'reload_verified':True,
            'total_parameters':config['total_parameters'],'trainable_parameters':config['trainable_parameters'],'context':24576,'max_new_tokens':2048,
            'metrics_version':'2.1','inner_dev_metrics':selected_report['metrics'],'official_case_gate_pass':None,
            'factory_entrypoint':'ml.qwen.v21_runtime:factory','sha256':hashes,'shared_contract_requirement':'ObjectRef.name permits empty string, official2.1'}
        save(export/'model_manifest.json',descriptor)
        report={'status':'completed','source_step':648,'new_updates':step,'schedule_completed':step==goal,'stop_reason':reason,
            'selected_new_updates':best_step,'selected_is_original648':best_step==0,'before':baseline,'last':measured,'selected':selected_report,
            'reload':reload_result,'elapsed_total_sec':budget.now()-budget.origin_epoch,'budget':budget.as_dict(),
            'official_val383_run':False,'official_case_gate_pass':None,'changed_prompt_and_postprocessing_used_identically_before_after':True}
        save(out/'experiment_report.json',report)
        print('TRAINING_AND_PAIRED_CHECKS_FINISHED',json.dumps({'new_updates':step,'selected_new_updates':best_step,'selected_is_original648':best_step==0,'reload_passed':True},ensure_ascii=False),flush=True)
    except BaseException as exc:
        # A possibly partially executed optimizer step is never published as a completed checkpoint.
        if not inside_optimizer_step and 'model' in locals() and 'optimizer' in locals() and budget.left('hard')>180:
            try:snapshot('last',step)
            except BaseException as backup:save(out/'checkpoint_save_error.json',{'type':type(backup).__name__,'message':str(backup)})
        save(out/'failure.json',{'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc(),
            'completed_new_updates':step,'inside_optimizer_step':inside_optimizer_step,'budget':budget.as_dict(),
            'failed_generation_is_not_successful_empty':True,'original648_payload_preserved':True})
        if isinstance(exc,BudgetStop):
            save(out/'experiment_report.json',{'status':'stopped_or_incomplete','reason':str(exc),'completed_new_updates':step,
                'official_case_gate_pass':None,'budget':budget.as_dict(),'inspect_last_pointer':True});return
        raise

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--run',type=Path,required=True)
    ap.add_argument('--budget-file',type=Path,required=True);ap.add_argument('--resume-checkpoint',type=Path)
    args=ap.parse_args()
    try:run(args)
    except BaseException as exc:
        args.run.mkdir(parents=True,exist_ok=True)
        save(args.run/'launcher_error.json',{'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc(),'phase':'setup_or_uncaught','original648_payload_preserved':True})
        raise

if __name__=='__main__':main()
