import json
import pytest
from src.contracts.models import CallInput
from src.pipeline.adapters import MockExtractor
from src.pipeline.service import Pipeline
from src.experiments.benchmark import BenchmarkTarget, benchmark, summarize, checkpoint_size

CALL=CallInput(call_id='bench-fixture',started_at='2026-11-13T12:00:00',turns=[{'idx':1,'speaker':'client','text':'Интернет не работает.'}])

class Clock:
    def __init__(self):self.t=0
    def __call__(self):self.t+=.25;return self.t

def test_cold_warm_raw_and_recomputation(tmp_path):
    loads=[];closed=[]
    def build(config):
        loads.append(config)
        return BenchmarkTarget(Pipeline(MockExtractor()),'cpu',synthetic=True,close=lambda:closed.append(1))
    out=tmp_path/'bench'
    result=benchmark(build,[CALL],out=out,config={},repeats=3,cold_trials=2,warmup_calls=[CALL],clock=Clock())
    raw=[json.loads(line) for line in (out/'benchmark_raw.jsonl').read_text().splitlines()]
    assert len(loads)==len(closed)==2
    assert len([r for r in raw if r['phase']=='warmup'])==1
    assert len([r for r in raw if r['phase']=='cold'])==2
    assert result['median_latency_sec']==.25
    assert result['successes']==3 and result['errors']==0 and result['schema_valid_percent']==100
    assert result['synthetic'] and result['peak_vram_bytes'] is None and result['gpu_seconds_per_1000'] is None
    for key,val in summarize(raw).items():assert result[key]==val
    hw=json.loads((out/'hardware.json').read_text());assert hw['device']=='cpu' and hw['machine']
    assert result['calls_per_hour']==result['calls_per_sec']*3600

def test_errors_are_counted_and_do_not_fabricate_success(tmp_path):
    class Broken:
        extractor=MockExtractor()
        def run(self,call):raise RuntimeError('sensitive input')
    result=benchmark(lambda c:BenchmarkTarget(Broken(),'cpu',synthetic=True),[CALL],out=tmp_path/'bad',config={},cold_trials=1,repeats=2)
    assert result['errors']==2 and result['schema_valid_percent']==0
    assert result['mean_latency_sec'] is None and result['calls_per_sec']==0
    assert 'sensitive input' not in (tmp_path/'bad'/'benchmark_raw.jsonl').read_text()

def test_summary_known_samples():
    raw=[{'phase':'warm','elapsed_sec':i,'status':'ok','schema_valid':True} for i in [1,2,3,4]]
    result=summarize(raw)
    assert result['mean_latency_sec']==2.5 and result['p95_latency_sec']==4
    assert result['calls_per_sec']==.4

def test_unavailable_gpu_never_builds_target(tmp_path,monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules,'torch',None)
    def forbidden(c):raise AssertionError('loader should not run')
    assert benchmark(forbidden,[CALL],out=tmp_path/'gpu',config={},device='cuda:0') is None
    manifest=json.loads((tmp_path/'gpu'/'benchmark_manifest.json').read_text())
    assert manifest['status']=='not_measured'
    assert (tmp_path/'gpu'/'benchmark_raw.jsonl').read_text()==''

def test_loader_failure_records_failed_run(tmp_path):
    def broken(config):raise RuntimeError('secret')
    with pytest.raises(RuntimeError):benchmark(broken,[CALL],out=tmp_path/'failed',config={},cold_trials=1)
    assert json.loads((tmp_path/'failed'/'benchmark_manifest.json').read_text())['status']=='failed'
    assert 'secret' not in (tmp_path/'failed'/'benchmark_raw.jsonl').read_text()

def test_checkpoint_size(tmp_path):
    (tmp_path/'weights').write_bytes(b'123');(tmp_path/'config').write_bytes(b'12')
    assert checkpoint_size([str(tmp_path),str(tmp_path/'weights')])==5
    assert checkpoint_size([]) is None


def test_cuda_synchronization_and_allocation_formula_with_fake_runtime(tmp_path,monkeypatch):
    import sys
    from types import SimpleNamespace
    events=[]
    cuda=SimpleNamespace(is_available=lambda:True,set_device=lambda d:events.append('device'),
        get_device_name=lambda d:'SYNTHETIC CUDA',get_device_properties=lambda d:SimpleNamespace(total_memory=1000),
        synchronize=lambda d:events.append('sync'),reset_peak_memory_stats=lambda d:events.append('reset'),max_memory_reserved=lambda d:200)
    monkeypatch.setitem(sys.modules,'torch',SimpleNamespace(cuda=cuda,version=SimpleNamespace(cuda='synthetic')))
    result=benchmark(lambda c:BenchmarkTarget(Pipeline(MockExtractor()),'cuda:0',synthetic=True),[CALL],
        out=tmp_path/'gpu-fake',config={},device='cuda:0',repeats=2,cold_trials=1,exclusive_gpu=True,clock=Clock())
    assert events.count('sync')>=8 and 'reset' in events
    assert result['peak_vram_bytes']==200 and result['synthetic']
    assert result['gpu_seconds_per_1000']==result['warm_wall_sec']*1000/2


def test_foreign_prediction_is_not_schema_success(tmp_path):
    class WrongCall:
        extractor=MockExtractor()
        def run(self,call):
            prediction,warnings=Pipeline(self.extractor).run(call)
            return prediction.model_copy(update={'call_id':'foreign-call'}),warnings
    result=benchmark(lambda c:BenchmarkTarget(WrongCall(),'cpu',synthetic=True),[CALL],out=tmp_path/'wrong',config={},cold_trials=1,repeats=1)
    assert result['successes']==0 and result['schema_valid_percent']==0


def test_cleanup_failure_invalidates_completed_manifest_once(tmp_path):
    closed=[]
    def close():closed.append(1);raise RuntimeError('private engine detail')
    with pytest.raises(RuntimeError):
        benchmark(lambda c:BenchmarkTarget(Pipeline(MockExtractor()),'cpu',synthetic=True,close=close),[CALL],out=tmp_path/'close',config={},cold_trials=1,repeats=1)
    assert closed==[1]
    assert json.loads((tmp_path/'close/benchmark_manifest.json').read_text())['status']=='failed'
    assert json.loads((tmp_path/'close/benchmark_summary.json').read_text())['status']=='failed'


def test_intermediate_cleanup_is_not_repeated_on_failure(tmp_path):
    closed=[]
    def close():closed.append(1);raise RuntimeError('close failure')
    with pytest.raises(RuntimeError):
        benchmark(lambda c:BenchmarkTarget(Pipeline(MockExtractor()),'cpu',synthetic=True,close=close),[CALL],out=tmp_path/'close',config={},cold_trials=2,repeats=1)
    assert closed==[1]


def test_failed_warmup_invalidates_resource_comparison(tmp_path):
    count=[]
    class OnceBroken:
        extractor=MockExtractor()
        def run(self,call):
            count.append(1)
            if len(count)==2:raise RuntimeError('warmup failed')
            return Pipeline(self.extractor).run(call)
    result=benchmark(lambda c:BenchmarkTarget(OnceBroken(),'cpu',synthetic=True),[CALL],warmup_calls=[CALL],out=tmp_path/'warmup',config={},cold_trials=1,repeats=1)
    assert not result['benchmark_valid'] and result['phase_errors']['warmup']==1


def test_changed_model_between_cold_trials_rejected(tmp_path):
    from src.pipeline.adapters import ModelDescriptor
    count=[]
    def build(c):
        count.append(1);extractor=MockExtractor();extractor.descriptor=ModelDescriptor('model-'+str(len(count)),'mock')
        return BenchmarkTarget(Pipeline(extractor),'cpu',synthetic=True)
    with pytest.raises(ValueError):benchmark(build,[CALL],out=tmp_path/'changed',config={},cold_trials=2,repeats=1)


def test_unawaited_async_pipeline_is_rejected_without_success(tmp_path):
    class AsyncPipeline:
        extractor=MockExtractor()
        async def run(self,call):raise AssertionError('must not time a coroutine as completed inference')
    result=benchmark(lambda c:BenchmarkTarget(AsyncPipeline(),'cpu',synthetic=True),[CALL],out=tmp_path/'async',config={},cold_trials=1,repeats=1)
    raw=[json.loads(s) for s in (tmp_path/'async/benchmark_raw.jsonl').read_text().splitlines()]
    assert result['successes']==0 and not result['benchmark_valid']
    assert all(r['error_type']=='TypeError' for r in raw if r['phase'] in {'cold','warm'})


def test_nonfinite_clock_cannot_create_complete_benchmark(tmp_path):
    with pytest.raises(ValueError):
        benchmark(lambda c:BenchmarkTarget(Pipeline(MockExtractor()),'cpu',synthetic=True),[CALL],out=tmp_path/'nan',config={},cold_trials=1,clock=lambda:float('nan'))
    assert json.loads((tmp_path/'nan/benchmark_manifest.json').read_text())['status']=='failed'


def test_declared_missing_checkpoint_is_error():
    with pytest.raises(ValueError):checkpoint_size(['/this/checkpoint/does/not/exist'])
