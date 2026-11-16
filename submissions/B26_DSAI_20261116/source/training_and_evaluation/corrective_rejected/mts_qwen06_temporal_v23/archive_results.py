"""Always callable after interruption; package only existing evidence and selected export."""
from pathlib import Path
import hashlib,json,zipfile

def archive(run):
    run=Path(run).resolve();path=run.parent/(run.name+'_final_results.zip')
    files=[p for p in run.rglob('*') if p.is_file() and 'inputs' not in p.relative_to(run).parts and '__pycache__' not in p.parts and p.suffix!='.pyc' and not p.name.endswith('.tmp')]
    hashes={}
    with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED) as z:
        for p in files:
            data=p.read_bytes();relative=p.relative_to(run).as_posix()
            hashes[relative]=hashlib.sha256(data).hexdigest();z.writestr(relative,data)
        z.writestr('results_manifest.json',json.dumps({'sha256':hashes,'scope':'Existing artifacts only; a ZIP does not imply a completed official evaluation.'},ensure_ascii=False,indent=2))
    print(path,flush=True);return path

if __name__=='__main__':
    import sys
    archive(sys.argv[1])
