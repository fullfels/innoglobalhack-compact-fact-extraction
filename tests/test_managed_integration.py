import threading
from concurrent.futures import ThreadPoolExecutor
import pytest
from fastapi.testclient import TestClient
from src.api.app import create_app
from src.api.config import Settings
from src.pipeline.adapters import ModelDescriptor, ModelResources, MockExtractor, PipelineError
from src.pipeline.lifecycle import LoadedModel, ManagedAdapter
from src.pipeline.service import Pipeline

CALL={'call_id':'integration','started_at':'2026-11-13T12:00:00','turns':[{'idx':4,'speaker':'client','text':'Интернет не работает.'}]}

def adapter(loader):
    return ManagedAdapter(ModelResources('/private/test-checkpoint',object(),{'device':'cpu'}),loader,
        descriptor=ModelDescriptor('synthetic-contract-test','real','controlled-fake'),revision='fixture-revision',artifact_sha256='fixture-not-a-real-hash')

def test_loader_once_and_release_after_serving():
    loaded=[];closed=[]
    def loader(resources):
        loaded.append(resources)
        return LoadedModel(lambda c,r:MockExtractor().extract(c),lambda:closed.append(1))
    a=adapter(loader)
    with TestClient(create_app(Settings(mode='real'),Pipeline(a))) as c:
        assert c.get('/health').json()['ready']
        a.load()
        for _ in range(3):
            response=c.post('/extract',json=CALL)
            assert response.status_code==200 and not response.json()['meta']['is_mock']
        info=c.get('/model/info').json()
        assert info['revision']=='fixture-revision' and '/private/' not in str(info)
    assert len(loaded)==len(closed)==1 and a.state=='unloaded'


def test_concurrent_load_is_single_initialization():
    loaded=[]
    a=adapter(lambda r:(loaded.append(1) or LoadedModel(lambda c,r:MockExtractor().extract(c))))
    with ThreadPoolExecutor(4) as pool:list(pool.map(lambda _:a.load(),range(4)))
    assert loaded==[1];a.close()


def test_load_failure_is_not_ready_or_silent(caplog):
    def loader(r):raise RuntimeError('SECRET PATH /private/checkpoint')
    a=adapter(loader)
    with TestClient(create_app(Settings(mode='real'),Pipeline(a))) as c:
        assert c.get('/health').status_code==503 and a.state=='failed'
        assert c.post('/extract',json=CALL).status_code==503
        with pytest.raises(PipelineError):a.load()
    assert 'RuntimeError' in caplog.text and 'SECRET PATH' not in caplog.text


def test_timeout_keeps_slot_and_shutdown_drains():
    entered,release,closed=threading.Event(),threading.Event(),threading.Event()
    def infer(call,r):
        entered.set();assert release.wait(5)
        return MockExtractor().extract(call)
    a=adapter(lambda r:LoadedModel(infer,closed.set))
    with TestClient(create_app(Settings(mode='real',inference_timeout_sec=.03),Pipeline(a))) as c:
        try:
            response=c.post('/extract',json=CALL)
            assert entered.is_set() and response.status_code==504
            assert response.json()['error']['code']=='INFERENCE_TIMEOUT_SLOT_RETAINED'
            assert c.post('/extract',json=CALL).status_code==429
            assert not closed.is_set()
        finally:release.set()
    assert closed.is_set() and a.state=='unloaded'

@pytest.mark.parametrize('v',[0,-1,float('inf'),float('nan')])
def test_invalid_timeout(v):
    with pytest.raises(ValueError):Settings(inference_timeout_sec=v)
