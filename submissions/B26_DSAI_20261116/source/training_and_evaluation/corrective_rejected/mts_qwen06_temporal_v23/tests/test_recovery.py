from pathlib import Path
import hashlib,json,zipfile
import pytest
from train_temporal import verify_snapshot
from restore_parent import restore
from pack_run import pack

def make_snapshot(path):
    names=['adapter/adapter_model.safetensors','adapter/adapter_config.json','training_state.pt','checkpoint.json','tokenizer/tokenizer.json',
        'tokenizer/tokenizer_config.json','tokenizer/chat_template.jinja','prompt.txt','catalog.json']
    hashes={}
    for n in names:
        p=path/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(('fixture:'+n).encode());hashes[n]=hashlib.sha256(p.read_bytes()).hexdigest()
    (path/'snapshot_hashes.json').write_text(json.dumps(hashes));return hashes

def test_incomplete_or_corrupt_snapshot_is_rejected(tmp_path):
    p=tmp_path/'snap';make_snapshot(p);verify_snapshot(p)
    (p/'training_state.pt').write_bytes(b'changed')
    with pytest.raises(ValueError):verify_snapshot(p)
    (p/'snapshot_hashes.json').write_text('{}')
    with pytest.raises(ValueError):verify_snapshot(p)

def test_light_restore_reads_only_hash_pinned_last500_not250(tmp_path):
    root=tmp_path/'root';root.mkdir();data=b'expected500';h=hashlib.sha256(data).hexdigest()
    (root/'parent_files.json').write_text(json.dumps({'adapter/adapter_model.safetensors':h}))
    archive=tmp_path/'model.zip'
    with zipfile.ZipFile(archive,'w') as z:
        z.writestr('export/adapter/adapter_model.safetensors',b'wrong250')
        z.writestr('checkpoints/update000500/adapter/adapter_model.safetensors',data)
        z.writestr('checkpoints/update000500/training_state.pt',b'never execute this pickle')
    restore(root,archive)
    assert (root/'parent500/adapter/adapter_model.safetensors').read_bytes()==data
    assert not(root/'parent500/training_state.pt').exists()

def test_small_results_preserve_metrics_but_model_and_optimizer_remain_separate(tmp_path):
    run=tmp_path/'run';(run/'export').mkdir(parents=True);(run/'checkpoints').mkdir()
    (run/'experiment_report.json').write_text('{}');(run/'export/model.safetensors').write_bytes(b'weight');(run/'checkpoints/training_state.pt').write_bytes(b'optimizer')
    result=pack(run)
    with zipfile.ZipFile(result) as z:
        assert 'experiment_report.json'in z.namelist() and not any(n.endswith(('.pt','.safetensors')) for n in z.namelist())
    model=pack(run,True)
    with zipfile.ZipFile(model) as z:assert 'export/model.safetensors'in z.namelist()

def test_another_gpu_worker_is_not_silently_run_in_parallel(monkeypatch):
    import os,subprocess
    from gpu_guard import check_idle_worker
    monkeypatch.setattr(subprocess,'check_output',lambda *a,**kw:f'{os.getpid()}, python\n{os.getppid()}, python\n')
    assert check_idle_worker()['other_python_gpu_workers']==0
    monkeypatch.setattr(subprocess,'check_output',lambda *a,**kw:'999999999, python\n')
    with pytest.raises(RuntimeError,match='Another Python GPU worker'):check_idle_worker()
