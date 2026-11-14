"""Offline negative profile checks on an intentionally ML-incomplete base image."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import uuid
import pytest
from src.experiments.common import sha


@pytest.mark.skipif(os.getenv('MWS_DOCKER_TESTS') != '1', reason='Docker negative profile build is opt-in')
def test_qwen_profile_refuses_missing_corrupt_and_unavailable_runtime(tmp_path):
    root = Path(__file__).resolve().parents[1]
    context = tmp_path/'context';context.mkdir()
    tracked = subprocess.check_output(['git','ls-files','-z'],cwd=root).decode().split('\0')
    names = {p for p in tracked if p.startswith(('src/','frontend/'))}
    names.update(['Dockerfile','Dockerfile.qwen','.dockerignore','requirements.lock',
                  'scripts/qwen_runtime_preflight.py','ml/__init__.py',
                  *['ml/qwen/'+name for name in ['__init__.py','bridge.py','prompting.py','runtime.py','requirements.txt']]])
    for name in names:
        destination=context/name;destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(root/name,destination)
    suffix=uuid.uuid4().hex
    base='mws-qwen-negative-base:'+suffix;image='mws-qwen-negative-profile:'+suffix
    export=tmp_path/'export';export.mkdir(mode=0o755)
    cache=tmp_path/'cache';cache.mkdir(mode=0o755)
    for name in ['adapter','tokenizer']:(export/name).mkdir(mode=0o755)
    for name in ['prompt.txt','catalog.json']:(export/name).write_text('SYNTHETIC_PACKAGING_SENTINEL')
    manifest=export/'model_manifest.json'
    def run(environment,expected):
        args=['docker','run','--rm','--network','none','--read-only',
              '-v',str(export)+':/models/export:ro','-v',str(cache)+':/hf-cache:ro']
        for key,value in environment.items():args.extend(['-e',key+'='+value])
        result=subprocess.run([*args,image,'--check-only'],capture_output=True,text=True,timeout=30)
        assert result.returncode==2, result.stdout+result.stderr
        report=json.loads(result.stderr.strip().splitlines()[-1])
        assert report['status']=='blocked' and report['code']==expected
    try:
        subprocess.run(['docker','build','-q','-t',base,str(context)],check=True,timeout=180)
        subprocess.run(['docker','build','-q','-f',str(context/'Dockerfile.qwen'),
                        '--build-arg','MWS_ENGINE_IMAGE='+base,'-t',image,str(context)],check=True,timeout=60)
        run({},'MODEL_DIR_REQUIRED')
        env={'MTS_MODEL_DIR':'/models/export','MWS_MODEL_ID':'packaging-only',
             'MWS_TORCH_VERSION':'not-a-real-runtime-version'}
        manifest.write_text('{broken')
        env['MWS_ARTIFACT_SHA256']=sha(manifest)
        run(env,'MODEL_MANIFEST_INVALID')
        manifest.write_text('{"model_id":"packaging-only"}')
        run(env,'MANIFEST_HASH_MISMATCH')
        env['MWS_ARTIFACT_SHA256']=sha(manifest)
        run(env,'DEPENDENCY_MISSING')  # Base is deliberately missing Torch/ML packages.
    finally:
        for tag in [image,base]:
            subprocess.run(['docker','image','rm','-f',tag],capture_output=True,timeout=30)
