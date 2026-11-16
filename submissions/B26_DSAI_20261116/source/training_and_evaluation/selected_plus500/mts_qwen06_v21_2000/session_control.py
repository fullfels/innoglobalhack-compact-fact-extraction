"""Colab orchestration. One persisted wall-clock budget; results survive kernel reset."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,subprocess,sys,threading,time,zipfile

LOCATOR=Path('/content/mts_qwen06_v21_current.json')

def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.pending');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');os.replace(tmp,path)

def arm_disconnect(deadline):
    """VM release is best effort; process supervisor independently enforces its own cutoff."""
    prior=globals().get('_release_timer')
    if prior:prior.cancel()
    def release():
        try:
            from google.colab import runtime
            runtime.unassign()
        except Exception as exc:print('Не удалось отключить ВМ автоматически:',repr(exc),flush=True)
    timer=threading.Timer(max(0,deadline-time.time()),release);timer.daemon=True;timer.start()
    globals()['_release_timer']=timer
    return timer

def setup_session(origin,available_minutes=300,existing_run='',*,mount=True,arm=True):
    if not 60<=available_minutes<=300:raise ValueError('AVAILABLE_MINUTES: от60 до300, с учётом уже использованного сегодня времени')
    if arm:arm_disconnect(origin+available_minutes*60)
    if mount:
        from google.colab import drive
        drive.mount('/content/drive')
    if existing_run:
        run=Path(existing_run)
        if not (run/'budget.json').is_file():raise ValueError('EXISTING_RUN не содержит budget.json; не создаём пустую замену')
    elif LOCATOR.exists():
        run=Path(json.loads(LOCATOR.read_text())['run'])
        if not (run/'budget.json').is_file():raise ValueError('Предыдущая папка недоступна; проверьте Drive/EXISTING_RUN')
    else:
        run=Path('/content/drive/MyDrive/MTS_Qwen06_V21_runs')/datetime.fromtimestamp(origin,timezone.utc).strftime('v21_%Y%m%dT%H%M%SZ')
        run.mkdir(parents=True,exist_ok=False)
        atomic_json(run/'budget.json',{'origin_epoch':origin,'total_minutes':available_minutes,'scope':'Entire experiment, not a guaranteed Google quota'})
    budget=json.loads((run/'budget.json').read_text());deadline=budget['origin_epoch']+budget['total_minutes']*60
    if available_minutes!=budget['total_minutes']:raise ValueError('Повторный запуск не меняет бюджет; сохранённый размер: '+str(budget['total_minutes']))
    atomic_json(LOCATOR,{'run':str(run),'root':'/content/mts_qwen06_v21_2000'})
    if arm:arm_disconnect(deadline)
    print('Папка RUN:',run,'\nОсталось общего времени, минут:',round(max(0,deadline-time.time())/60,1),flush=True)
    print('Бюджет включает подготовку, загрузку, обучение, оценку и архив. Доступность GPU Google не гарантируется.')
    return run,budget

def recover_run(existing_run=''):
    if existing_run:run=Path(existing_run)
    elif LOCATOR.exists():run=Path(json.loads(LOCATOR.read_text())['run'])
    else:
        candidates=sorted(Path('/content/drive/MyDrive/MTS_Qwen06_V21_runs').glob('v21_*'))
        raise ValueError('Укажите EXISTING_RUN явно, чтобы не выбрать чужой запуск: '+str([str(p) for p in candidates]))
    if not (run/'budget.json').is_file():raise ValueError('Нет сохранённого budget.json: '+str(run))
    return run

def supervise(command,run,budget):
    from ml.qwen.budget_v21 import BudgetStop
    budget.require(15*60)
    if (run/'experiment_report.json').is_file():
        report=json.loads((run/'experiment_report.json').read_text())
        if report.get('status')=='completed' or '--resume-checkpoint' not in command:
            raise ValueError('В этой папке уже есть отчёт. Используйте просмотр результатов; незавершённый запуск возобновляется явно из своего чекпоинта.')
    env=dict(os.environ,TOKENIZERS_PARALLELISM='false',HF_HUB_DISABLE_PROGRESS_BARS='1',PYTHONUNBUFFERED='1')
    with (run/'console.log').open('a',encoding='utf-8',buffering=1) as log:
        process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace',bufsize=1,env=env)
        def reader():
            for line in process.stdout:
                log.write(line);print(line,end='',flush=True)
        thread=threading.Thread(target=reader,daemon=True);thread.start()
        interrupted=False
        try:
            while process.poll() is None:
                if budget.left('supervisor')<=0:raise BudgetStop('Пора сохранить результаты и освободить GPU')
                time.sleep(1)
        except (KeyboardInterrupt,BudgetStop) as exc:
            interrupted=True;print(str(exc),'— останавливаю процесс и сохраняю завершённые шаги.',flush=True)
            process.terminate()
            try:process.wait(timeout=min(60,max(1,budget.left('hard')-120)))
            except subprocess.TimeoutExpired:process.kill();process.wait(timeout=30)
        finally:
            if process.poll() is None:
                process.terminate()
                try:process.wait(timeout=30)
                except subprocess.TimeoutExpired:process.kill();process.wait(timeout=30)
            thread.join(timeout=10)
        atomic_json(run/'process_exit.json',{'returncode':process.returncode,'interrupted':interrupted,'remaining_seconds':budget.left('hard')})
        return process.returncode

def archive_results(run):
    """Full latest optimizer checkpoint + selected export; never pretend a partial snapshot is complete."""
    from ml.qwen.budget_v21 import verify_snapshot
    run=Path(run).resolve();keep=set();errors=[];latest_complete=False
    for label in ['last','best']:
        pointer=run/(label+'_pointer.json')
        if not pointer.is_file():continue
        try:
            snap=(run/json.loads(pointer.read_text())['relative_path']).resolve()
            if not snap.is_relative_to(run/'checkpoints'):raise ValueError('Unsafe pointer')
            verify_snapshot(snap)
            if label=='last':latest_complete=True
            if label=='last' or not (run/'export/model_manifest.json').exists():keep.add(snap)
        except Exception as exc:errors.append({'pointer':label,'error':str(exc)})
    archive=run.parent/(run.name+'_results.zip');temporary=archive.with_suffix('.pending.zip')
    selected=[]
    for p in sorted(run.rglob('*')):
        if not p.is_file() or p.is_symlink() or '__pycache__' in p.parts or p.suffix=='.pyc' or '.pending' in p.name:continue
        if (run/'checkpoints') in p.parents and not any(s in p.parents for s in keep):continue
        selected.append(p)
    def sha(path):
        h=hashlib.sha256()
        with path.open('rb') as f:
            for block in iter(lambda:f.read(4*1024**2),b''):h.update(block)
        return h.hexdigest()
    files={str(p.relative_to(run)).replace('\\','/'):sha(p) for p in selected}
    with zipfile.ZipFile(temporary,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as z:
        for p in selected:z.write(p,str(p.relative_to(run)).replace('\\','/'))
        z.writestr('results_archive_manifest.json',json.dumps({'files_sha256':files,'checkpoint_errors':errors,
            'includes_full_latest_training_state':latest_complete,'original_run':str(run)},ensure_ascii=False,indent=2))
    with zipfile.ZipFile(temporary) as z:
        if z.testzip() is not None:raise ValueError('Result ZIP CRC failure; original run preserved')
    os.replace(temporary,archive)
    atomic_json(run/'results_archive.json',{'path':str(archive),'sha256':sha(archive),'bytes':archive.stat().st_size,'checkpoint_errors':errors})
    print('Архив результатов на Google Drive:',archive,flush=True)
    return archive

def show_results(run):
    print('Папка:',run)
    report=run/'experiment_report.json'
    if report.is_file():
        value=json.loads(report.read_text(encoding='utf-8'))
        print('Результат:',json.dumps({k:value[k] for k in ['status','new_updates','completed_new_updates','schedule_completed','selected_new_updates','selected_is_original648','stop_reason','reason'] if k in value},ensure_ascii=False))
        for label in ['before','last','selected']:
            part=value.get(label)
            if part:print(label,json.dumps({'complete':part.get('complete'),'metrics':part.get('metrics'),'counts':part.get('counts'),'failed':part.get('failed_generations')},ensure_ascii=False))
        print('Полный отчёт:',report)
    for name in ['failure.json','launcher_error.json','training_complete.json','last_pointer.json','best_pointer.json','results_archive.json']:
        p=run/name
        if p.is_file():
            value=json.loads(p.read_text(encoding='utf-8'));value.pop('traceback',None)
            print('\n'+name+'\n'+json.dumps(value,ensure_ascii=False,indent=2))
    log=run/'training_log.json'
    if log.is_file():
        entries=json.loads(log.read_text());print('\nПоследние обновления:',json.dumps(entries[-3:],ensure_ascii=False,indent=2))
