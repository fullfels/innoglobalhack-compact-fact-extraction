"""Opt-in real Docker test; only tracked sources and synthetic sentinel files."""
import os
from pathlib import Path
import shutil
import subprocess
import uuid

import pytest

@pytest.mark.skipif(os.getenv('MWS_DOCKER_TESTS') != '1', reason='Set MWS_DOCKER_TESTS=1 to build sentinel image')
def test_sensitive_files_excluded_from_context_and_image(tmp_path):
    root = Path(__file__).resolve().parents[1]
    # Copy tracked filenames only: never read local ignored secrets/weights.
    tracked = subprocess.check_output(['git','ls-files','-z'],cwd=root).decode().split('\0')
    for name in tracked:
        if name and (name.startswith(('src/','frontend/')) or name in {'Dockerfile','.dockerignore','requirements.lock'}):
            destination = tmp_path/name
            destination.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(root/name,destination)
    sentinels = ['.env','src/.env','src/api/.env.local','frontend/.env.production',
        'src/credentials.json','src/secrets/id_ed25519','frontend/private.pem','src/private.key',
        'src/.aws/credentials','src/ml/model.safetensors','src/ml/weights.pt','src/ml/weights.pth',
        'src/ml/state.ckpt','src/ml/model.gguf','src/ml/model.onnx','src/ml/pytorch_model.bin',
        'src/ml/checkpoint-12/optimizer.bin']
    for name in sentinels:
        p=tmp_path/name;p.parent.mkdir(parents=True,exist_ok=True)
        p.write_text('SYNTHETIC_AUDIT_SENTINEL_NOT_A_SECRET')
    # A probe COPY validates context exclusion as well as production COPY src/frontend.
    with (tmp_path/'Dockerfile').open('a') as f: f.write('\nCOPY . /context-probe\n')
    image='mws-sentinel-test:'+uuid.uuid4().hex
    try:
        subprocess.run(['docker','build','-q','-t',image,str(tmp_path)],check=True,timeout=180)
        check = """from pathlib import Path
from src.api.app import app
assert app.title == 'MWS CompactFactExtraction'
assert Path('/app/src/api/worker.py').is_file()
assert Path('/app/frontend/app.js').is_file()
for name in %r:
    assert not (Path('/context-probe') / name).exists(), name
    assert not (Path('/app') / name).exists(), name
print('PASS: sentinel files excluded; runtime sources intact')
""" % sentinels
        subprocess.run(['docker','run','--rm','--entrypoint','python',image,'-c',check],check=True,timeout=30)
    finally:
        subprocess.run(['docker','image','rm','-f',image],check=False,capture_output=True,timeout=30)
