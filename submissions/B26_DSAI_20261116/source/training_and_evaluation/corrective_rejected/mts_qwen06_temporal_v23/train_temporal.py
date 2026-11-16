"""One corrective LoRA pass from +500. Original remains an eligible fallback."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,gc,hashlib,importlib.metadata,json,math,os,random,shutil,signal,statistics,time
from final_eval import save,rows,verify,evaluate,export_model
from ml.qwen.final_runtime import load_model,Engine,sha,MODEL,REVISION
from temporal_loss import encode_record,suffix_loss,loss_selftest
from quality_guard import allowed,metric_key,paired_dates,primary_metrics,stress_guard

STOP=False
def request_stop(*_):
    global STOP
    STOP=True

def verify_snapshot(path):
    path=Path(path).resolve();m=json.loads((path/'snapshot_hashes.json').read_text())
    required={'adapter/adapter_model.safetensors','adapter/adapter_config.json','training_state.pt','checkpoint.json',
        'tokenizer/tokenizer.json','tokenizer/tokenizer_config.json','tokenizer/chat_template.jinja','prompt.txt','catalog.json'}
    if not required<=set(m):raise ValueError('Incomplete checkpoint')
    for rel,want in m.items():
        p=(path/rel).resolve()
        if not p.is_relative_to(path) or sha(p)!=want:raise ValueError('Corrupt checkpoint: '+rel)
    return path

def choose_initial(report,folder,root,dev):
    choices=[]
    if not report['complete'] or report['metric_errors']:raise ValueError('Original +500 baseline is incomplete or evaluator failed')
    for calendar in [False,True]:
        view=('calendar' if calendar else 'partial')+'_serving';m=primary_metrics(report,view)
        if m is None:continue
        failed=report['statuses']['calendar' if calendar else 'partial'].get('failed',0)
        choices.append({'step':0,'calendar':calendar,'metrics':m,'failed':failed,'folder':str(folder),
            'paired':paired_dates(root,[c for c in dev if not c['call_id'].startswith('synthetic-v23-')],folder/(view+'_predictions.jsonl')),
            'metrics_scope':'24 real inner-dev calls; synthetic stress is separate','all_metrics':report['metrics'][view]})
    if not choices:raise ValueError('No valid original quality result')
    return max(choices,key=lambda c:(metric_key(c['metrics'],c['failed']),not c['calendar']))

def assess_quality(report,folder,root,dev,baseline,initial,best,step):
    options=[]
    if not report['complete'] or report['metric_errors']:return best,options
    for calendar in [False,True]:
        view=('calendar' if calendar else 'partial')+'_serving';m=primary_metrics(report,view)
        if m is None:continue
        real=[c for c in dev if not c['call_id'].startswith('synthetic-v23-')]
        paired=paired_dates(root,real,folder/(view+'_predictions.jsonl'));ok,reasons=allowed(m,initial['metrics'],paired,initial['paired'])
        base_view=('calendar' if initial['calendar'] else 'partial')+'_serving'
        stress_ok,stress_reasons=stress_guard(report,baseline,view,base_view);ok=ok and stress_ok;reasons+=stress_reasons
        failed=report['statuses']['calendar' if calendar else 'partial'].get('failed',0)
        if failed>initial['failed']:ok=False;reasons.append('more_failed_generations')
        c={'step':step,'calendar':calendar,'metrics':m,'metrics_scope':'24 real inner-dev calls; synthetic stress separate','all_metrics':report['metrics'][view],
            'failed':failed,'folder':str(folder),'paired':paired,'allowed':ok,'reasons':reasons};options.append(c)
        if ok and metric_key(m,failed)>metric_key(best['metrics'],best['failed']):best=c
    return best,options

def run(a):
    import torch
    from gpu_guard import check_idle_worker
    gpu_process_check=check_idle_worker()
    root=a.root.resolve();out=a.run.resolve();out.mkdir(parents=True,exist_ok=True);package=verify(root)
    if not torch.cuda.is_available():raise RuntimeError('CUDA required; no silent CPU training')
    torch.manual_seed(42);torch.cuda.manual_seed_all(42);random.seed(42);torch.backends.cuda.matmul.allow_tf32=False
    signal.signal(signal.SIGTERM,request_stop);signal.signal(signal.SIGINT,request_stop)
    budget=json.loads((out/'session_budget.json').read_text());deadline=budget['train_phase_deadline_epoch']
    if time.time()+600>deadline:raise RuntimeError('Less than ten minutes remain in this run budget')
    dependencies={n:importlib.metadata.version(n) for n in ['torch','transformers','peft','tokenizers','accelerate','safetensors','pydantic']}
    config={'base_model':MODEL,'base_revision':REVISION,'parent_adapter_sha256':sha(root/'parent500/adapter/adapter_model.safetensors'),
        'package_manifest_sha256':sha(root/'package_manifest.json'),'train_rows':1024,'goal_updates':256,'accumulation':4,'learning_rate':1e-5,
        'epochs':1,'date_token_weight':2.,'seed':42,'train_window_minutes':a.train_minutes,'budget':budget,'dependencies':dependencies,
        'gpu':torch.cuda.get_device_name(0),'vram_bytes':torch.cuda.get_device_properties(0).total_memory,'precision':'fp16_base_fp32_adapter',
        'no_full_weight_training':True,'official_val_used_for_selection':False}
    if config['parent_adapter_sha256']!='577bc8184877de05386244791bee2ecaeb5dd894e86df749cd31ed270dbab98b':raise ValueError('Wrong parent: requires verified +500')
    config_path=out/'training_config.json'
    if config_path.exists() and json.loads(config_path.read_text())!=config:raise ValueError('Resume config/hardware/dependencies changed')
    save(config_path,config)
    # Byte-preserved parent for reconnects; snapshots may reserialize the same tensors.
    shutil.copytree(root/'parent500',out/'parent500_original',dirs_exist_ok=True)
    save(out/'gpu_process_check.json',gpu_process_check)
    provenance=out/'provenance';provenance.mkdir(exist_ok=True)
    for relative in ['package_manifest.json','audit/data_policy.json','audit/split_ids.json','requirements.txt']:
        dest=provenance/relative;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(root/relative,dest)
    for path in root.rglob('*.py'):
        if '__pycache__' in path.parts:continue
        dest=provenance/'source'/path.relative_to(root);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,dest)
    sft=rows(root/'sft/train.jsonl');dev=rows(root/'data/selection32.jsonl');audit=rows(root/'data/audit24.jsonl')
    if len(sft)!=1024:raise ValueError('Expected 1024 rows')
    resume=None;step=0;load_from=root/'parent500'
    if a.resume:
        snap=verify_snapshot(a.resume)
        if not snap.is_relative_to(out/'checkpoints'):raise ValueError('Exact resume must belong to this run')
        latest=(out/json.loads((out/'last_pointer.json').read_text())['relative_path']).resolve()
        if snap!=latest:raise ValueError('Resume the latest committed checkpoint only; older snapshots need a separate experiment')
        resume=torch.load(snap/'training_state.pt',weights_only=True,map_location='cpu')
        if resume['config']!=config:raise ValueError('Checkpoint config mismatch')
        step=resume['step'];load_from=snap
        if type(step)is not int or not 0<=step<=256:raise ValueError('Invalid checkpoint update')
    model,tok=load_model(load_from/'adapter',load_from/'tokenizer',trainable=True)
    parameters=[p for p in model.parameters() if p.requires_grad]
    if sum(p.numel() for p in parameters)!=10092544:raise ValueError('Unexpected trainable params; never update full base')
    cfg=json.loads((load_from/'adapter/adapter_config.json').read_text())
    if cfg['r']!=16 or cfg['lora_alpha']!=32 or cfg['lora_dropout']!=0:raise ValueError('Unexpected existing LoRA config')
    total_parameters=sum(p.numel() for p in model.parameters());save(out/'parameter_counts.json',{'total':total_parameters,'trainable':sum(p.numel() for p in parameters)})
    encoded=[encode_record(r,tok,date_weight=2.) for r in sft]
    save(out/'native_token_audit.json',{'passed':True,'native_tokenizer':True,'rows':len(encoded),'min_tokens':min(len(r['input_ids']) for r in encoded),
        'max_tokens':max(len(r['input_ids']) for r in encoded),'masked_timestamp_tokens':sum(r['masked_timestamp_tokens'] for r in encoded),
        'weighted_timestamp_tokens':sum(r['weighted_timestamp_tokens'] for r in encoded),'prompt_loss_masked':True,'no_truncation':True})
    save(out/'loss_selftest.json',loss_selftest(model,tok))
    optimizer=torch.optim.AdamW(parameters,lr=1e-5,weight_decay=0.);scaler=torch.amp.GradScaler('cuda',enabled=True,init_scale=32.)
    logs=json.loads((out/'training_log.json').read_text()) if (out/'training_log.json').exists() else []
    logs=[r for r in logs if r['step']<=step]
    training_start=None
    if resume:
        optimizer.load_state_dict(resume['optimizer']);scaler.load_state_dict(resume['grad_scaler']);training_start=resume['training_start_epoch']
    def snapshot(update):
        path=out/'checkpoints'/f'update{update:06d}'
        if path.exists():return verify_snapshot(path)
        pending=path.with_name(path.name+'.pending')
        if pending.exists():
            if not pending.resolve().is_relative_to((out/'checkpoints').resolve()) or pending.is_symlink():raise ValueError('Unsafe pending checkpoint path')
            interrupted=out/'interrupted_snapshots';interrupted.mkdir(exist_ok=True)
            pending.rename(interrupted/(pending.name+'_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')))
        pending.mkdir(parents=True,exist_ok=False)
        model.eval();model.save_pretrained(pending/'adapter',safe_serialization=True,save_embedding_layers=False);tok.save_pretrained(pending/'tokenizer')
        shutil.copyfile(root/'prompt.txt',pending/'prompt.txt');shutil.copyfile(root/'official_catalog.json',pending/'catalog.json')
        state={'step':update,'config':config,'optimizer':optimizer.state_dict(),'grad_scaler':scaler.state_dict(),'training_start_epoch':training_start,
            'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'python_rng':random.getstate()}
        torch.save(state,pending/'training_state.pt');save(pending/'checkpoint.json',{'parent':'v21+500','new_updates':update,'calls_seen':update*4})
        save(pending/'snapshot_hashes.json',{p.relative_to(pending).as_posix():sha(p) for p in pending.rglob('*') if p.is_file()})
        verify_snapshot(pending);os.replace(pending,path)
        save(out/'last_pointer.json',{'relative_path':path.relative_to(out).as_posix(),'step':update});return path
    def quality(folder,snap,calls=None):
        model.eval();model.config.use_cache=True
        engine=Engine(model,tok,(root/'prompt.txt').read_text(encoding='utf-8'),json.loads((root/'official_catalog.json').read_text(encoding='utf-8')),deadline_epoch=deadline-600)
        return evaluate(engine,dev if calls is None else calls,root,folder,sha(snap/'adapter/adapter_model.safetensors'),'fp16',deadline-600)
    baseline_files=[out/'before_selection32/metrics.json',out/'before_guard24/metrics.json',out/'baseline_choice.json',out/'best_choice.json']
    if not resume or step==0 and not all(p.exists() for p in baseline_files):
        original=snapshot(0)
        baseline=quality(out/'before_selection32',original)
        if min((v.get('failed',0) for v in baseline['statuses'].values()),default=32)>=24:raise ValueError('Most original generations failed; diagnose loader/prompt before optimization')
        initial=choose_initial(baseline,out/'before_selection32',root,dev)
        original_guard=quality(out/'before_guard24',original,audit)
        if not original_guard['complete'] or original_guard['metric_errors']:raise ValueError('Original safety guard incomplete; do not train without a reference')
        save(out/'baseline_choice.json',initial);best=initial
        save(out/'best_choice.json',best);save(out/'best_pointer.json',{'relative_path':original.relative_to(out).as_posix(),'step':0})
    else:
        baseline=json.loads((out/'before_selection32/metrics.json').read_text());initial=json.loads((out/'baseline_choice.json').read_text());best=json.loads((out/'best_choice.json').read_text())
        original_guard=json.loads((out/'before_guard24/metrics.json').read_text())
        torch.set_rng_state(resume['torch_rng']);torch.cuda.set_rng_state_all(resume['cuda_rng']);random.setstate(resume['python_rng'])
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});model.enable_input_require_grads();model.config.use_cache=False
    model.train();optimizer.zero_grad(set_to_none=True);torch.cuda.reset_peak_memory_stats()
    longest=max(encoded,key=lambda r:len(r['input_ids']))
    with torch.autocast('cuda',dtype=torch.float16):probe=suffix_loss(model,longest)
    probe_scaler=torch.amp.GradScaler('cuda',enabled=True,init_scale=32.);probe_scaler.scale(probe).backward();probe_scaler.unscale_(optimizer)
    if not any(p.grad is not None for p in parameters) or not all(bool(torch.isfinite(p.grad).all()) for p in parameters if p.grad is not None):raise RuntimeError('Longest-call gradient probe failed')
    optimizer.zero_grad(set_to_none=True);save(out/'longest_probe.json',{'passed':True,'optimizer_step':False,'tokens':len(longest['input_ids']),'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated()})
    del probe,probe_scaler
    if training_start is None:training_start=time.time();save(out/'training_clock.json',{'training_start_epoch':training_start})
    # Resume RNG exactly after probes; probes intentionally perform no optimizer step.
    if resume:
        torch.set_rng_state(resume['torch_rng']);torch.cuda.set_rng_state_all(resume['cuda_rng']);random.setstate(resume['python_rng'])
    stop_reason='one_pass_completed';checks=json.loads((out/'quality_checks.json').read_text()) if resume and (out/'quality_checks.json').exists() else [];loss=None;group=None
    tracked_name=next(n for n,p in model.named_parameters() if p.requires_grad and 'lora_B' in n)
    initial_tracked=dict(model.named_parameters())[tracked_name].detach().cpu().clone()
    while step<256:
        estimate=max(45.,statistics.median(r['seconds'] for r in logs[-16:])*1.4) if logs else 90.
        train_end=min(training_start+a.train_minutes*60,deadline-40*60)
        if STOP or time.time()+estimate+120>train_end:stop_reason='user_stop' if STOP else 'training_time_limit';break
        model.train();model.config.use_cache=False;optimizer.zero_grad(set_to_none=True);torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
        group=encoded[step*4:(step+1)*4];denom=sum(r['supervised_weight'] for r in group);values=[]
        lr=1e-5*min((step+1)/8,(256-step)/248)
        for g in optimizer.param_groups:g['lr']=lr
        for row in group:
            weight=row['supervised_weight']/denom
            with torch.autocast('cuda',dtype=torch.float16):loss=suffix_loss(model,row)
            if not bool(torch.isfinite(loss)):raise RuntimeError('Non-finite loss; keep previous committed checkpoint')
            values.append(float(loss.detach())*weight);scaler.scale(loss*weight).backward()
        scaler.unscale_(optimizer);norm=torch.nn.utils.clip_grad_norm_(parameters,.5)
        if not bool(torch.isfinite(norm)):raise RuntimeError('Non-finite gradients; optimizer step not applied')
        oldscale=scaler.get_scale();scaler.step(optimizer);scaler.update();torch.cuda.synchronize()
        if scaler.get_scale()<oldscale:raise RuntimeError('AMP skipped update; not counting it as completed')
        step+=1;item={'step':step,'calls_seen':step*4,'weighted_loss':sum(values),'seconds':time.perf_counter()-start,'lr':lr,'gradient_norm':float(norm),
            'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated()};logs.append(item);save(out/'training_log.json',logs)
        print('update',step,'/256 calls',step*4,'loss',round(item['weighted_loss'],5),'sec',round(item['seconds'],2),'eta_train_min',round(statistics.median(r['seconds'] for r in logs[-16:])*(256-step)/60,1),flush=True)
        if step==16 and not resume:
            changed=not torch.equal(initial_tracked,dict(model.named_parameters())[tracked_name].detach().cpu())
            save(out/'first_updates.json',{'passed':changed,'steps':16,'adapter_weight_changed':changed,'measured_median_seconds_per_update':statistics.median(r['seconds'] for r in logs[-16:])})
            if not changed:raise RuntimeError('Adapter did not change after16 completed updates')
        if step==16 or step%32==0:snapshot(step)
        if step in [64,128,192,256]:
            snap=snapshot(step)
            if time.time()+max(300.,baseline['generation_seconds_total']*1.3)+600>deadline:
                checks.append({'step':step,'status':'skipped_budget'});continue
            folder=out/f'selection_update{step:06d}';report=quality(folder,snap)
            best,options=assess_quality(report,folder,root,dev,baseline,initial,best,step)
            checks.append({'step':step,'options':options});save(out/'quality_checks.json',checks);save(out/'best_choice.json',best)
            save(out/'best_pointer.json',{'relative_path':f"checkpoints/update{best['step']:06d}",'step':best['step']})
    last=snapshot(step);save(out/'training_complete.json',{'new_updates':step,'calls_seen':step*4,'goal_updates':256,'schedule_completed':step==256,'stop_reason':stop_reason,'training_window_seconds':time.time()-training_start})
    # Also assess the actual last checkpoint if a wall-clock cutoff was between milestones.
    if step>0 and not any(c.get('step')==step and c.get('options') for c in checks) and time.time()+max(300.,baseline['generation_seconds_total']*1.3)+600<deadline:
        folder=out/f'selection_update{step:06d}';report=quality(folder,last)
        best,options=assess_quality(report,folder,root,dev,baseline,initial,best,step)
        checks.append({'step':step,'is_last_partial_schedule':step<256,'options':options});save(out/'quality_checks.json',checks);save(out/'best_choice.json',best)
        save(out/'best_pointer.json',{'relative_path':f"checkpoints/update{best['step']:06d}",'step':best['step']})
    selected=verify_snapshot(out/'checkpoints'/f"update{best['step']:06d}")
    # A genuine fresh load, not the still-trained in-memory object.
    del model,optimizer,parameters,loss,group;gc.collect();torch.cuda.empty_cache()
    model,tok=load_model(selected/'adapter',selected/'tokenizer');model.eval()
    from final_eval import reload_test
    guard_after=None;rollback=None;engine=None
    if best['step']!=0 and time.time()+max(300.,original_guard['generation_seconds_total']*1.3)+360<deadline:
        engine=Engine(model,tok,(root/'prompt.txt').read_text(encoding='utf-8'),json.loads((root/'official_catalog.json').read_text(encoding='utf-8')),deadline_epoch=deadline-300)
        folder=out/'candidate_guard24'
        guard_after=evaluate(engine,audit,root,folder,sha(selected/'adapter/adapter_model.safetensors'),'fp16',deadline-300,allow_calendar_choices=False,calendar_enabled=best['calendar'])
        view=('calendar' if best['calendar'] else 'partial')+'_serving';old_view=('calendar' if initial['calendar'] else 'partial')+'_serving'
        if guard_after['complete'] and not guard_after['metric_errors']:
            real_audit=[c for c in audit if not c['call_id'].startswith('synthetic-v23-')]
            pc=paired_dates(root,real_audit,folder/(view+'_predictions.jsonl'));pb=paired_dates(root,real_audit,out/'before_guard24'/(old_view+'_predictions.jsonl'))
            ok,reasons=allowed(primary_metrics(guard_after,view),primary_metrics(original_guard,old_view),pc,pb)
            stress_ok,stress_reasons=stress_guard(guard_after,original_guard,view,old_view);ok=ok and stress_ok;reasons+=stress_reasons
            if not ok:rollback={'reason':'new_model_failed_inner_safety_guard','details':reasons}
        else:rollback={'reason':'new_model_safety_guard_incomplete'}
    elif best['step']!=0:rollback={'reason':'not_enough_time_to_check_new_model_safely'}
    if rollback:
        save(out/'rollback.json',{**rollback,'rejected_candidate':best,'guard_scope':'Inner safety guard, used for rollback, not an independent final test'})
        best=initial;save(out/'best_choice.json',best);save(out/'best_pointer.json',{'relative_path':'checkpoints/update000000','step':0})
        selected=verify_snapshot(out/'checkpoints/update000000');del model,engine;gc.collect();torch.cuda.empty_cache()
        model,tok=load_model(selected/'adapter',selected/'tokenizer');model.eval()
    reload_test(model,tok,selected,out/'reload_test.json');reload=json.loads((out/'reload_test.json').read_text())
    export_choice={'step':best['step'],'calendar':best['calendar'],'metrics':best['metrics'],'parent':'v21+500','selection_scope':'inner32, official validation not used'}
    export=export_model(root,selected,out,export_choice,reload,model)
    manifest=json.loads((export/'model_manifest.json').read_text());manifest.update(model_id=f"qwen06-temporal-v23-parent500-plus{best['step']}",parent_adapter_sha256=config['parent_adapter_sha256'],source_parent='v21_step648_plus500',selected_corrective_updates=best['step'],total_parameters=total_parameters)
    save(export/'model_manifest.json',manifest)
    shutil.copyfile(export/'model_manifest.json',out/'provenance/selected_model_manifest.json')
    report={'status':'completed','parent':'v21+500','new_updates':step,'selected_updates':best['step'],'selected_original_parent':best['step']==0,
        'baseline_best':initial,'selected':best,'same_policy_before_metrics':primary_metrics(baseline,('calendar' if best['calendar'] else 'partial')+'_serving'),
        'baseline_guard24':original_guard,'candidate_guard24':guard_after,'rollback':rollback,'guard24_scope':'Inner safety guard used for rollback, not independent final test',
        'reload':reload,'elapsed_total_seconds':time.time()-budget['origin_epoch'],'official_val383_run':False,
        'official_threshold_pass':None,'observed_inner_improvement':best['step']!=0,'guarantee_on_unseen_data':False}
    save(out/'experiment_report.json',report)
    print(json.dumps({'status':report['status'],'new_updates':step,'selected_updates':best['step'],'selected_original_parent':best['step']==0,
        'before_serving':initial['metrics'],'selected_serving':best['metrics'],'rollback':rollback,'export':str(export),'official_val383_run':False},ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--run',type=Path,required=True);p.add_argument('--train-minutes',type=int,default=90);p.add_argument('--resume',type=Path);a=p.parse_args()
    if not 10<=a.train_minutes<=120:raise ValueError('Training window must be 10..120 minutes')
    try:run(a)
    except Exception as exc:
        save(a.run/'failure.json',{'error_type':type(exc).__name__,'message':str(exc)[:1800],'saved_checkpoints_remain_on_drive':True});raise
