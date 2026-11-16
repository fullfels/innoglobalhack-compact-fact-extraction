"""Small evidence ZIP for analysis; selected model is a separate artifact on Drive."""
from pathlib import Path
import hashlib,json,zipfile

def pack(run,model=False):
    run=Path(run).resolve();suffix='_model.zip' if model else '_results.zip';out=run.parent/(run.name+suffix)
    hashes={}
    with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(run.rglob('*')):
            if not p.is_file() or p.name.endswith('.tmp') or '__pycache__' in p.parts or 'interrupted_snapshots' in p.parts:continue
            relative=p.relative_to(run)
            if model:
                if relative.parts[0]!='export':continue
            else:
                if relative.parts[0] in {'checkpoints','export','inputs','parent500_original'}:continue
                if p.suffix in {'.pt','.safetensors','.bin'}:continue
            data=p.read_bytes();name=relative.as_posix();hashes[name]=hashlib.sha256(data).hexdigest();z.writestr(name,data)
        z.writestr('results_manifest.json',json.dumps({'sha256':hashes,'includes_model':model,'optimizer_state_on_drive_only':True},ensure_ascii=False,indent=2))
    print(str(out),flush=True);return out

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('--model',action='store_true');a=p.parse_args();pack(a.run,a.model)
