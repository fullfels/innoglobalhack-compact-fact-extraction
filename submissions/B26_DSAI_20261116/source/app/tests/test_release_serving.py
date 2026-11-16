"""Controlled protocol/inventory tests. No actual Qwen weights or quality claim."""
from pathlib import Path
import json
import shutil
import weakref
import pytest
from fastapi.testclient import TestClient
from src.api.app import create_app
from src.api.config import Settings
from src.contracts.models import CallInput, ObjectRef
from src.pipeline.adapters import PipelineError
from ml.qwen import serving

APP=Path(__file__).resolve().parents[1]
CALL={'call_id':'controlled-protocol-not-native-qwen','started_at':'2026-11-15T12:00:00',
      'turns':[{'idx':7,'speaker':'client','text':'Интернет не работает.'}]}
FACT={'object':{'name':'Интернет','type':'service'},'attribute':'service_state','value':'не работает',
      'polarity':1,'speaker':'client','valid_from':CALL['started_at'],'valid_to':None,
      'source':{'turn_idx':7,'span':None,'quote':None}}


@pytest.fixture
def export(tmp_path):
    for rel in serving.POLICY_FILES:
        dest=tmp_path/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(APP/rel,dest)
    placeholders={'adapter/adapter_model.safetensors':b'controlled fake, not actual weights',
        'adapter/adapter_config.json':b'{}','tokenizer/tokenizer.json':b'{}',
        'tokenizer/tokenizer_config.json':b'{}','tokenizer/chat_template.jinja':b'controlled',
        'prompt.txt':b'controlled prompt','catalog.json':b'{}'}
    for rel,data in placeholders.items():
        p=tmp_path/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
    m={'model_id':'controlled-protocol-only','base_model':serving.MODEL,'base_revision':serving.REVISION,
       'reload_verified':True,'reload_test':{'passed':True,'fresh_base_and_adapter':True,
          'adapter_sha256':serving.sha(tmp_path/'adapter/adapter_model.safetensors')},
       'calendar_enabled':False,'context':24576,'max_new_tokens':2048,'total_parameters':1,
       'sha256':{p.relative_to(tmp_path).as_posix():serving.sha(p) for p in tmp_path.rglob('*') if p.is_file()}}
    (tmp_path/'model_manifest.json').write_text(json.dumps(m),encoding='utf-8')
    return tmp_path,m


def test_verified_export_inventory_and_policy(export):
    path,m=export
    root,actual=serving.verify_export(path,expected_hash=serving.sha(path/'model_manifest.json'))
    assert root==path and actual['model_id']==m['model_id']


@pytest.mark.parametrize('kind',['missing_inventory','changed_weights','unlisted_tokenizer','reload_string','changed_revision','foreign_id','foreign_manifest','changed_source'])
def test_invalid_exports_rejected_before_native_load(export,kind):
    path,m=export;kwargs={}
    if kind=='missing_inventory':m['sha256']={}
    if kind=='changed_weights':(path/'adapter/adapter_model.safetensors').write_bytes(b'changed')
    if kind=='unlisted_tokenizer':(path/'tokenizer/extra.json').write_text('{}')
    if kind=='reload_string':m['reload_verified']='false'
    if kind=='changed_revision':m['base_revision']='other'
    if kind=='foreign_id':kwargs['expected_model_id']='another-model'
    if kind=='foreign_manifest':kwargs['expected_hash']='0'*64
    if kind=='changed_source':
        p=path/serving.POLICY_FILES[0];p.write_text('changed source')
        m['sha256'][serving.POLICY_FILES[0]]=serving.sha(p)
    (path/'model_manifest.json').write_text(json.dumps(m),encoding='utf-8')
    with pytest.raises(PipelineError):serving.verify_export(path,**kwargs)


def controlled_load(monkeypatch,raw,events):
    class Model:
        def eval(self):events.append('eval');return self
    def load(*args,**kwargs):events.append(('load',kwargs['device'],kwargs['precision'],kwargs['local_only']));return Model(),object()
    class Engine:
        def __init__(self,model,tokenizer,*args,**kwargs):self.model=model;self.tokenizer=tokenizer
        def generate(self,call):
            assert not hasattr(call,'facts');events.append('generate')
            if isinstance(raw,Exception):raise raw
            return raw
    monkeypatch.setattr(serving,'load_model',load);monkeypatch.setattr(serving,'Engine',Engine)


def test_actual_v23_bridge_through_existing_real_http_api(export,monkeypatch):
    path,m=export;events=[]
    controlled_load(monkeypatch,json.dumps({'facts':[FACT]}),events)
    pipeline=serving.evaluation_factory({'model_dir':str(path),'device':'cpu'})
    with TestClient(create_app(Settings(mode='real'),pipeline)) as client:
        assert client.get('/health').json()['is_mock'] is False
        assert client.get('/model/info').json()['manifest_id']==m['model_id']
        for _ in range(2):
            response=client.post('/extract',json=CALL);assert response.status_code==200,response.text
            assert response.json()['prediction']['facts'][0]['source']['turn_idx']==7
            assert response.json()['meta']['model_id']==m['model_id']
        assert client.post('/extract',json={**CALL,'facts':[FACT]}).status_code==422
    assert events.count(('load','cpu','fp16',True))==1
    assert events.count('generate')==2 and pipeline.extractor.state=='unloaded'


def test_all_bad_facts_do_not_become_successful_empty_response(export,monkeypatch):
    path,_=export;events=[]
    controlled_load(monkeypatch,json.dumps({'facts':[{**FACT,'speaker':'operator'}]}),events)
    with TestClient(create_app(Settings(mode='real'),serving.build({'model_dir':str(path)}))) as client:
        result=client.post('/extract',json=CALL)
        assert result.status_code==503 and 'prediction' not in result.json()
        assert result.json()['error']['code']=='QWEN_ALL_FACTS_REJECTED'


def test_load_failure_releases_partial_allocation(export,monkeypatch):
    path,_=export;released=[]
    class Allocated:pass
    def fail(*args,**kwargs):
        allocation=Allocated();weakref.finalize(allocation,released.append,'released')
        raise RuntimeError('controlled partial load failure')
    monkeypatch.setattr(serving,'load_model',fail)
    pipeline=serving.build({'model_dir':str(path)})
    with pytest.raises(PipelineError):pipeline.extractor.load()
    assert released==['released'] and pipeline.extractor.state=='failed'


def test_configuration_changes_apply_and_unknown_keys_are_not_silently_ignored(export,monkeypatch):
    path,_=export;events=[]
    controlled_load(monkeypatch,'{"facts":[]}',events)
    pipeline=serving.build({'model_dir':str(path),'device':'cuda','precision':'int4'})
    pipeline.extractor.load()
    assert ('load','cuda','int4',True) in events
    # This test is controlled: avoid importing real CUDA during close.
    monkeypatch.setitem(__import__('sys').modules,'torch',type('Torch',(),{'cuda':type('CUDA',(),{'empty_cache':staticmethod(lambda:None)})}))
    pipeline.extractor.close()
    with pytest.raises(PipelineError):serving.configuration({'ignored_option':True})
    with pytest.raises(PipelineError):serving.configuration({'model_dir':str(path),'device':'cpu','precision':'int4'})


def test_base_requires_inventory_including_weights(tmp_path):
    (tmp_path/'base_identity.json').write_text(json.dumps({'repo_id':serving.MODEL,'revision':serving.REVISION,'sha256':{}}))
    with pytest.raises(PipelineError):serving.verify_base(tmp_path)


def test_official_v21_empty_object_name_is_allowed():
    assert ObjectRef(name='').name==''


def configured_fixture(export,tmp_path):
    from scripts.configure_release import prepare
    path,m=export
    base=tmp_path/'base';base.mkdir()
    (base/'config.json').write_text('{"model_type":"qwen3"}')
    (base/'model.safetensors').write_bytes(b'controlled fake, never native weights')
    identity={'repo_id':serving.MODEL,'revision':serving.REVISION,
              'sha256':{p.name:serving.sha(p) for p in base.iterdir()}}
    (base/'base_identity.json').write_text(json.dumps(identity))
    out=tmp_path/'deployment'
    before=(path/'model_manifest.json').read_bytes()
    config=prepare(path,base,out,torch_version='2.14.1+cu130')
    assert (path/'model_manifest.json').read_bytes()==before
    return config,out,path,base


def test_configuration_generator_does_not_claim_model_load_or_modify_proof(export,tmp_path):
    c,out,path,base=configured_fixture(export,tmp_path)
    assert c['model_loaded'] is False and c['quality_threshold_pass'] is None
    assert c['torch_release']=='2.14.1' and c['weights_modified'] is False
    assert json.loads((out/'runtime_config.json').read_text())['model']['model_dir']==str(path)
    assert (out/'.env.release').is_file()


def test_generated_launch_uses_same_model_identity_and_preflight(export,tmp_path):
    from scripts.launch_release import command
    c,out,path,base=configured_fixture(export,tmp_path)
    args,env,root=command(out/'runtime_config.json',port=18083,environ={})
    assert args[1:3]==['-m','scripts.release_preflight']
    assert 'uvicorn' in args and '--workers' in args and args[-1]=='1'
    assert env['MWS_ARTIFACT_SHA256']==serving.sha(path/'model_manifest.json')
    assert env['MTS_MODEL_DIR']==str(path) and env['MTS_MODEL_BASE_PATH']==str(base)
    assert env['HF_HUB_OFFLINE']=='1' and env['MWS_MODE']=='real'


def test_check_only_does_not_include_server_command(export,tmp_path):
    from scripts.launch_release import command
    _,out,_,_=configured_fixture(export,tmp_path)
    args,_,_=command(out/'runtime_config.json',check_only=True,environ={})
    assert args[-1]=='--check-only' and 'uvicorn' not in args


def test_failed_configuration_leaves_no_deployment_files(export,tmp_path):
    from scripts.configure_release import prepare
    path,_=export;out=tmp_path/'not_created'
    with pytest.raises(PipelineError):prepare(path,tmp_path/'missing_base',out,torch_version='2.14.1')
    assert not out.exists()


def test_dotenv_literals_and_line_break_rejection():
    from scripts.configure_release import quoted
    assert quoted('C:/models/$project with spaces')=="'C:/models/$project with spaces'"
    assert quoted("Let's go") == "'Let\\'s go'"
    with pytest.raises(ValueError):quoted('bad\nSECOND_VARIABLE=1')


@pytest.mark.parametrize('device,expected', [('cpu',600),('cuda',120)])
def test_device_specific_generation_limit(monkeypatch,device,expected):
    monkeypatch.delenv('MWS_MODEL_MAX_CALL_SECONDS',raising=False)
    assert serving.configuration({'model_dir':'unused','device':device})['max_call_seconds']==expected


@pytest.mark.parametrize('value', ['0','-1','nan','inf','bad','3601'])
def test_invalid_environment_generation_limit(monkeypatch,value):
    monkeypatch.setenv('MWS_MODEL_MAX_CALL_SECONDS',value)
    with pytest.raises(PipelineError):serving.configuration({'model_dir':'unused'})


def test_explicit_generation_limit_overrides_environment(monkeypatch):
    monkeypatch.setenv('MWS_MODEL_MAX_CALL_SECONDS','900')
    assert serving.configuration({'model_dir':'unused'})['max_call_seconds']==900
    assert serving.configuration({'model_dir':'unused','max_call_seconds':300})['max_call_seconds']==300
    with pytest.raises(PipelineError):serving.configuration({'model_dir':'unused','max_call_seconds':True})


def test_cpu_limit_reaches_engine_and_public_api(export,monkeypatch):
    path,_=export;events=[];captured=[]
    monkeypatch.delenv('MWS_MODEL_MAX_CALL_SECONDS',raising=False)
    controlled_load(monkeypatch,'{"facts":[]}',events)
    original=serving.Engine
    class CapturingEngine(original):
        def __init__(self,*args,**kwargs):
            captured.append(kwargs['max_call_seconds']);super().__init__(*args,**kwargs)
    monkeypatch.setattr(serving,'Engine',CapturingEngine)
    with TestClient(create_app(Settings(mode='real'),serving.build({'model_dir':str(path)}))) as client:
        assert client.get('/model/info').json()['limits']['generation_seconds']==600
    assert captured==[600]
