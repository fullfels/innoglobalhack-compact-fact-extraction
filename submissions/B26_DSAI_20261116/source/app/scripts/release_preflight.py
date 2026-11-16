"""Checks packaging without pretending to run Qwen. Used by Docker entrypoint."""
from pathlib import Path
import importlib.metadata
import json
import os
import sys
import re
from ml.qwen.serving import configuration,verify_export,verify_base


def check():
    c=configuration()
    if os.getenv('MWS_MODE')!='real':raise ValueError('REAL_MODE_REQUIRED')
    if os.getenv('MWS_PIPELINE_FACTORY')!='ml.qwen.serving:factory':raise ValueError('RELEASE_FACTORY_REQUIRED')
    if os.getenv('HF_HUB_OFFLINE')!='1':raise ValueError('OFFLINE_MODE_REQUIRED')
    if not c['manifest_sha256'] or not c['model_id']:raise ValueError('EXPORT_ID_AND_HASH_REQUIRED')
    root,m=verify_export(c['model_dir'],expected_hash=c['manifest_sha256'],expected_model_id=c['model_id'])
    if not c['base_path']:raise ValueError('MATERIALIZED_BASE_REQUIRED')
    verify_base(c['base_path'])
    versions={name:importlib.metadata.version(name) for name in ['torch','transformers','peft','tokenizers','accelerate','safetensors','fastapi','pydantic','uvicorn']}
    app_root=Path(__file__).resolve().parents[1]
    def check_requirements(path):
        for line in path.read_text(encoding='utf-8').splitlines():
            line=line.strip()
            if not line or line.startswith('#'):continue
            if line.startswith('-r '):check_requirements(path.parent/line[3:]);continue
            name,version=line.split('==')
            if importlib.metadata.version(name)!=version:raise ValueError('RUNTIME_DEPENDENCY_VERSION_MISMATCH')
    check_requirements(app_root/'requirements.runtime.lock')
    torch_version=os.getenv('MWS_TORCH_VERSION')
    if not torch_version or versions['torch'].split('+')[0]!=torch_version.split('+')[0]:
        raise ValueError('TORCH_DISTRIBUTION_VERSION_REQUIRED_OR_MISMATCH')
    return {'status':'packaging_verified','model_id':m['model_id'],'device':c['device'],
            'versions':versions,'model_loaded':False,'quality_threshold_pass':None}


def main():
    try:
        print(json.dumps(check()),flush=True)
        if sys.argv[1:]==['--check-only']:return 0
        if not sys.argv[1:]:raise ValueError('SERVER_COMMAND_REQUIRED')
        os.execvp(sys.argv[1],sys.argv[1:])
    except Exception as exc:
        print(json.dumps({'status':'blocked','code':getattr(exc,'code','RELEASE_PREFLIGHT_FAILED'),
                          'error_type':type(exc).__name__}),file=sys.stderr,flush=True)
        return 2

if __name__=='__main__':raise SystemExit(main())
