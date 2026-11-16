"""Independent negative readiness contracts; no GPU/weights. Failures are blockers, not passing acceptance."""
import importlib.util, json, os, sys, types, weakref
from pathlib import Path
import pytest
from src.pipeline.adapters import PipelineError
@pytest.fixture
def runtime():
 path=os.getenv('MWS_REVIEW_RUNTIME_FILE')
 if path:
  spec=importlib.util.spec_from_file_location('reviewed_qwen_runtime',path)
  module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
 from ml.qwen import runtime as module
 return module

def manifest(runtime,tmp_path):
 value={'artifact_state':'trained_adapter_reload_verified','training_quality_gates_passed':True,
 'base_model':runtime.MODEL,'base_revision':runtime.REVISION,'sha256':{},'model_id':'synthetic-probe',
 'context':16384,'max_new_tokens':100,'total_parameters':1,'manifest_id':'synthetic','dtype':'float32'}
 (tmp_path/'model_manifest.json').write_text(json.dumps(value));return value

def test_empty_hash_inventory_rejected(runtime,tmp_path):
 manifest(runtime,tmp_path)
 with pytest.raises(PipelineError):runtime.artifact_manifest(tmp_path)

def test_unlisted_runtime_file_rejected(runtime,tmp_path):
 import hashlib
 value=manifest(runtime,tmp_path);(tmp_path/'prompt.txt').write_text('fixture')
 value['sha256']={'prompt.txt':hashlib.sha256(b'fixture').hexdigest()}
 (tmp_path/'model_manifest.json').write_text(json.dumps(value))
 (tmp_path/'tokenizer').mkdir();(tmp_path/'tokenizer/tokenizer.json').write_text('changed, unlisted')
 with pytest.raises(PipelineError):runtime.artifact_manifest(tmp_path)

def test_non_boolean_quality_gate_rejected(runtime,tmp_path):
 value=manifest(runtime,tmp_path);value['training_quality_gates_passed']='false'
 (tmp_path/'model_manifest.json').write_text(json.dumps(value))
 with pytest.raises(PipelineError):runtime.artifact_manifest(tmp_path)

def test_cuda_index_preserves_cuda_dtype(runtime,monkeypatch):
 observed=[]
 class Tokenizer:
  pad_token_id=1;eos_token_id=1
 class Model:
  def to(self,device):return self
 tokenizer=types.SimpleNamespace(from_pretrained=lambda *a,**k:Tokenizer())
 def load(*a,**kwargs):observed.append(kwargs['dtype']);return Model()
 model=types.SimpleNamespace(from_pretrained=load)
 monkeypatch.setitem(sys.modules,'torch',types.SimpleNamespace(float16='float16',float32='float32',cuda=types.SimpleNamespace(is_available=lambda:True)))
 monkeypatch.setitem(sys.modules,'transformers',types.SimpleNamespace(AutoTokenizer=tokenizer,AutoModelForImageTextToText=model))
 runtime.load_base('cuda:0',local_only=True)
 assert observed==['float16']

def test_partial_load_failure_releases_allocated_model(runtime,tmp_path,monkeypatch):
 value=manifest(runtime,tmp_path);released=[]
 monkeypatch.setattr(runtime,'artifact_manifest',lambda path:(tmp_path,value))
 class Allocated:pass
 def load(*a,**k):
  model=Allocated();weakref.finalize(model,released.append,'released');return model,object()
 def fail(*a,**k):raise RuntimeError('controlled adapter load failure')
 monkeypatch.setattr(runtime,'load_base',load)
 monkeypatch.setitem(sys.modules,'peft',types.SimpleNamespace(PeftModel=types.SimpleNamespace(from_pretrained=fail)))
 monkeypatch.setitem(sys.modules,'transformers',types.SimpleNamespace(AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda *a,**k:object())))
 monkeypatch.setenv('MTS_MODEL_DIR',str(tmp_path));monkeypatch.setenv('MTS_MODEL_DEVICE','cpu')
 adapter=runtime.factory().extractor
 with pytest.raises(PipelineError) as failure:adapter.load()
 assert failure.value.code=='MODEL_LOAD_FAILED'
 assert released==['released']

def test_config_aware_evaluation_factory_exists(runtime):
 assert callable(getattr(runtime,'evaluation_factory',None)), 'API factory() is not evaluation factory(config)'

def test_benchmark_factory_exists(runtime):
 assert callable(getattr(runtime,'benchmark_factory',None)), 'BenchmarkTarget factory/config/device/load/close not supplied'
