"""Materialize the pinned base outside Git; does not load or train a model."""
import argparse
from pathlib import Path
import hashlib
import json
import shutil

MODEL='Qwen/Qwen3-0.6B'
REVISION='c1899de289a04d12100db370d81485cdf75e47ca'


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--cached-only',action='store_true')
    a=p.parse_args()
    from huggingface_hub import snapshot_download
    cached=Path(snapshot_download(MODEL,revision=REVISION,
        local_files_only=a.cached_only,allow_patterns=['*.json','*.safetensors','*.txt','*.model','*.jinja','LICENSE'])).resolve()
    root=a.out.resolve();root.mkdir(parents=True,exist_ok=True)
    if root==cached or root.is_relative_to(cached):raise RuntimeError('Use a separate output directory, not the Hub cache')
    for path in sorted(cached.rglob('*')):
        if path.is_file() and path.suffix in {'.json','.safetensors','.txt','.model','.jinja'} or path.is_file() and path.name=='LICENSE':
            dest=root/path.relative_to(cached);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,dest)
    inventory={}
    for path in sorted(root.rglob('*')):
        if not path.is_file() or '.cache' in path.parts or path.name=='base_identity.json':continue
        with path.open('rb') as f:inventory[path.relative_to(root).as_posix()]=hashlib.file_digest(f,'sha256').hexdigest()
    if 'config.json' not in inventory or not any(x.endswith('.safetensors') for x in inventory):
        raise RuntimeError('Incomplete pinned base snapshot')
    index=root/'model.safetensors.index.json'
    if index.is_file() and not set(json.loads(index.read_text())['weight_map'].values())<=inventory.keys():
        raise RuntimeError('Pinned base has missing weight shards')
    value={'repo_id':MODEL,'revision':REVISION,'sha256':inventory,
           'scope':'File provenance only; native model reload still required'}
    (root/'base_identity.json').write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'base':str(root),'files':len(inventory),'native_reload_verified':False}),flush=True)

if __name__=='__main__':main()
