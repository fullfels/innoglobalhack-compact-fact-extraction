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


def test_close_waits_for_direct_adapter_inference():
    from src.contracts.models import CallInput
    entered,release,closed=threading.Event(),threading.Event(),threading.Event()
    def infer(call,r):entered.set();assert release.wait(5);assert not closed.is_set();return MockExtractor().extract(call)
    a=adapter(lambda r:LoadedModel(infer,closed.set));a.load()
    with ThreadPoolExecutor(2) as pool:
        inference=pool.submit(a.extract,CallInput.model_validate(CALL))
        try:
            assert entered.wait(1)
            closing=pool.submit(a.close)
            assert not closed.wait(.1), 'engine released while physical inference is active'
        finally:release.set()
        inference.result();closing.result()
    assert closed.is_set() and a.state=='unloaded'


def test_invalid_loaded_bridge_releases_allocated_session():
    closed=[];a=adapter(lambda r:LoadedModel(None,lambda:closed.append(1)))
    with pytest.raises(PipelineError):a.load()
    assert a.state=='failed' and closed==[1]
    a.close();assert closed==[1]


def test_cancelled_startup_waits_for_loader_and_releases_resources():
    import asyncio
    async def scenario():
        entered,release,closed=threading.Event(),threading.Event(),threading.Event()
        def loader(r):entered.set();assert release.wait(5);return LoadedModel(lambda c,r:MockExtractor().extract(c),closed.set)
        a=adapter(loader);app=create_app(Settings(mode='real'),Pipeline(a))
        async def start():
            async with app.router.lifespan_context(app):pass
        startup=asyncio.create_task(start())
        try:
            assert await asyncio.to_thread(entered.wait,1)
            startup.cancel();await asyncio.sleep(0);await asyncio.sleep(0)
            assert not startup.done(), 'startup cancelled while loader owns untracked resources'
            startup.cancel();await asyncio.sleep(0)
        finally:release.set()
        with pytest.raises(asyncio.CancelledError):await startup
        assert closed.is_set() and a.state=='unloaded'
    asyncio.run(scenario())


def test_stopping_health_and_concurrent_close_are_safe():
    entered,release=threading.Event(),threading.Event();closed=[]
    def close():closed.append(1);entered.set();assert release.wait(5)
    a=adapter(lambda r:LoadedModel(lambda c,r:MockExtractor().extract(c),close))
    with TestClient(create_app(Settings(mode='real'),Pipeline(a))) as client, ThreadPoolExecutor(2) as pool:
        first=pool.submit(a.close)
        try:
            assert entered.wait(1)
            second=pool.submit(a.close)
            assert client.get('/health').json()['model_state']=='stopping'
            assert client.get('/health').status_code==503
            assert client.post('/extract',json=CALL).status_code==503
        finally:release.set()
        first.result();second.result()
    assert closed==[1]


def test_cancelled_startup_failure_is_still_logged(caplog):
    import asyncio
    async def scenario():
        entered,release=threading.Event(),threading.Event()
        def loader(r):entered.set();assert release.wait(5);raise RuntimeError('PRIVATE LOAD DETAILS')
        a=adapter(loader);app=create_app(Settings(mode='real'),Pipeline(a))
        async def start():
            async with app.router.lifespan_context(app):pass
        t=asyncio.create_task(start())
        assert await asyncio.to_thread(entered.wait,1)
        t.cancel();release.set()
        with pytest.raises(asyncio.CancelledError):await t
    asyncio.run(scenario())
    assert 'MODEL_LOAD_FAILED' in caplog.text and 'RuntimeError' in caplog.text
    assert 'PRIVATE LOAD DETAILS' not in caplog.text
