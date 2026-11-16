"""Only packaging fixtures/metadata; no fake model or inference implementation."""
import importlib.metadata
import json
from pathlib import Path
import pytest
from scripts.qwen_runtime_preflight import check_profile, ProfileBlocked
from src.experiments.common import sha


@pytest.fixture
def profile(tmp_path):
    export = tmp_path / 'export'; export.mkdir()
    for name in ['adapter', 'tokenizer']:(export/name).mkdir()
    for name in ['prompt.txt', 'catalog.json']:(export/name).write_text('independent packaging sentinel')
    manifest = export/'model_manifest.json'; manifest.write_text('{"model_id":"packaging-fixture"}')
    cache = tmp_path/'cache';cache.mkdir()
    env = {'MWS_MODE':'real', 'MWS_PIPELINE_FACTORY':'ml.qwen.runtime:factory',
           'MTS_MODEL_DIR':str(export), 'MWS_MODEL_ID':'packaging-fixture',
           'MWS_ARTIFACT_SHA256':sha(manifest),'HF_HOME':str(cache),'HF_HUB_OFFLINE':'1',
           'MWS_TORCH_VERSION':'synthetic-version-not-runtime','MTS_MODEL_DEVICE':'cpu'}
    root = Path(__file__).resolve().parents[1]
    versions={'torch':env['MWS_TORCH_VERSION']}
    for path in [root/'requirements.lock',root/'ml/qwen/requirements.txt']:
        for line in path.read_text().splitlines():
            if line.strip() and not line.startswith('#'):
                name,value=line.split('==');versions[name]=value
    return env,versions


def test_preflight_does_not_claim_model_loaded(profile):
    env,versions=profile
    result=check_profile(env,package_version=versions.__getitem__)
    assert result['status']=='packaging_preflight_passed'
    assert not result['model_loaded'] and not result['model_readiness_verified']


@pytest.mark.parametrize('damage,code', [
    ('mode','REAL_MODE_REQUIRED'),('hash','MANIFEST_HASH_MISMATCH'),
    ('identity','MODEL_ID_MISMATCH'),('missing','MODEL_MANIFEST_MISSING'),
    ('corrupt','MODEL_MANIFEST_INVALID'),('layout','EXPORT_LAYOUT_MISSING'),
    ('cache','OFFLINE_CACHE_MISSING'),('network','OFFLINE_MODE_REQUIRED'),
    ('device','DEVICE_PROFILE_NOT_APPROVED'),('version','DEPENDENCY_VERSION_MISMATCH'),
    ('dependency','DEPENDENCY_MISSING'),('factory','UNAPPROVED_FACTORY')])
def test_runtime_profile_blocks_invalid_or_unavailable_package(profile,damage,code):
    env,versions=profile;p=Path(env['MTS_MODEL_DIR'])/'model_manifest.json'
    if damage=='mode':env['MWS_MODE']='mock'
    elif damage=='hash':env['MWS_ARTIFACT_SHA256']='0'*64
    elif damage=='identity':env['MWS_MODEL_ID']='foreign'
    elif damage=='missing':p.unlink()
    elif damage=='corrupt':p.write_text('{broken');env['MWS_ARTIFACT_SHA256']=sha(p)
    elif damage=='layout':(p.parent/'adapter').rmdir()
    elif damage=='cache':env['HF_HOME']=str(p.parent/'absent')
    elif damage=='network':env['HF_HUB_OFFLINE']='0'
    elif damage=='device':env['MTS_MODEL_DEVICE']='cuda:0'
    elif damage=='version':versions['torch']='different-version'
    elif damage=='factory':env['MWS_PIPELINE_FACTORY']='unapproved:factory'
    def version(name):
        if damage=='dependency':raise importlib.metadata.PackageNotFoundError(name)
        return versions[name]
    with pytest.raises(ProfileBlocked,match=code):check_profile(env,package_version=version)
