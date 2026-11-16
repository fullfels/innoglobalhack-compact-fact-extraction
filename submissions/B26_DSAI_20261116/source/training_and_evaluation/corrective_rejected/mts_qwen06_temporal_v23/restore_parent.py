"""Optional light-payload bootstrap: restore only hash-pinned +500 files, no pickle."""
import hashlib,json,os,shutil,zipfile
from pathlib import Path

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def restore(root,extra_archive=None):
    root=Path(root).resolve();m=json.loads((root/'parent_files.json').read_text())
    def ready():return all((root/'parent500'/p).is_file() and sha(root/'parent500'/p)==h for p,h in m.items())
    if ready():return 'parent already included and verified'
    archives=[Path(extra_archive)] if extra_archive else []
    archives += [Path('/content/mts_v22_uploads/v21_20261115T123824Z_results.zip'),Path('/content/v21_20261115T123824Z_results.zip')]
    for archive in archives:
        if not archive.is_file():continue
        with zipfile.ZipFile(archive) as z:
            prefix='checkpoints/update000500/'
            if not all(prefix+p in z.namelist() for p in m):continue
            data={p:z.read(prefix+p) for p in m}
            if not all(hashlib.sha256(data[p]).hexdigest()==h for p,h in m.items()):continue
            for p,b in data.items():
                dst=(root/'parent500'/p).resolve()
                if not dst.is_relative_to(root/'parent500'):raise ValueError('Unsafe parent manifest')
                dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(b)
            if ready():return 'verified parent restored from existing uploaded results ZIP'
    search=Path('/content/drive/MyDrive/MTS_Qwen06_Final_runs')
    candidates=list(search.glob('final_*/inputs/checkpoints/update000500'))
    candidates+=list(Path('/content/drive/MyDrive/MTS_Qwen06_Temporal_runs').glob('temporal_*/parent500_original'))
    for source in sorted(candidates,reverse=True):
        if all((source/p).is_file() and sha(source/p)==h for p,h in m.items()):
            for p in m:
                dst=root/'parent500'/p;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source/p,dst)
            if ready():return 'verified parent restored from existing Drive run'
    raise FileNotFoundError('Parent +500 not found. Use the full v23 payload, or supply the verified v21 results ZIP path. Do not use export/+250.')

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--archive',type=Path);a=p.parse_args();print(restore(a.root,a.archive))
