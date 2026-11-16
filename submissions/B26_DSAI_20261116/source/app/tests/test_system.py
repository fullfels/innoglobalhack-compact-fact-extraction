from copy import deepcopy
from datetime import datetime
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from src.api.app import create_app
from src.api.config import Settings
from src.contracts.models import CallInput, Fact, CandidateBatch, CandidateFact
from src.pipeline.adapters import MockExtractor, InjectedAdapter, ModelResources, PipelineError
from src.pipeline.service import Pipeline, validate_output

EXAMPLE = {'call_id': 'demo-001', 'started_at': '2026-11-13T12:00:00', 'turns': [{'idx': 2, 'speaker': 'client', 'text': 'Интернет не работает.'}, {'idx': 7, 'speaker': 'operator', 'text': 'Заявка зарегистрирована.'}]}

@pytest.fixture
def client():
    with TestClient(create_app()) as c:
        yield c

@pytest.mark.parametrize('change', [lambda d:d.update(facts=[]), lambda d:d.update(started_at='yesterday'), lambda d:d.update(turns=[]), lambda d:d['turns'].append(d['turns'][0]), lambda d:d['turns'][0].update(idx=-1), lambda d:d['turns'][0].update(speaker='other'), lambda d:d.update(noise_level=2), lambda d:d.update(duration_sec=True)])
def test_invalid(client, change):
    d=deepcopy(EXAMPLE);change(d)
    r=client.post('/extract',json=d)
    assert r.status_code==422
    assert 'Интернет' not in r.text

def test_roundtrip():
    call=CallInput.model_validate(EXAMPLE)
    assert CallInput.model_validate_json(call.model_dump_json())==call
    assert call.started_at.tzinfo is None
    assert [t.idx for t in call.turns]==[2,7]

def test_api(client):
    a=client.post('/extract',json=EXAMPLE);b=client.post('/extract',json=EXAMPLE)
    assert a.status_code==200
    x=a.json();assert x['prediction']==b.json()['prediction']
    assert x['meta']['is_mock'] and x['meta']['schema_valid']
    assert len(x['prediction']['facts'])==2
    assert x['prediction']['facts'][0]['polarity']==1
    assert x['prediction']['facts'][0]['source']['turn_idx']==2
    assert client.get('/health').json()['ready']
    assert client.get('/model/info').json()['is_mock']
    assert client.get('/').status_code==200
    assert client.get('/static/app.js').status_code==200

def test_negative(client):
    d=deepcopy(EXAMPLE);d['turns']=[{'idx':9,'speaker':'client','text':'Я не использую роутер.'}]
    assert client.post('/extract',json=d).json()['prediction']['facts'][0]['polarity']==-1

def test_unknown(client):
    d=deepcopy(EXAMPLE);d['turns'][0]['text']='Произвольный текст'
    r=client.post('/extract',json=d)
    assert r.status_code==422 and r.json()['error']['code']=='MOCK_EXAMPLE_NOT_SUPPORTED'

def test_malformed_and_large(client):
    assert client.post('/extract',content='{').status_code==422
    assert client.post('/extract',content=b'x'*262145).status_code==413

def test_real_not_ready():
    with TestClient(create_app(Settings(mode='real',model_id='pending'))) as c:
        assert c.get('/health').status_code==503
        assert not c.get('/health').json()['is_mock']
        assert c.post('/extract',json=EXAMPLE).status_code==503

@pytest.mark.parametrize('patch', [{'polarity':True},{'polarity':0},{'attribute':'bad'},{'valid_to':'2020-01-01T00:00:00'}])
def test_fact_schema(patch):
    fact=MockExtractor().extract(CallInput.model_validate(EXAMPLE)).candidates[0].fact.model_dump()
    fact.update(patch)
    with pytest.raises(ValidationError):Fact.model_validate(fact)

@pytest.mark.parametrize('source', [{'turn_idx':99},{'turn_idx':2,'span':(0,999),'quote':'bad'},{'turn_idx':2,'span':(0,1),'quote':'bad'},{'turn_idx':2,'quote':'bad'}])
def test_grounding(source):
    call=CallInput.model_validate(EXAMPLE)
    d=MockExtractor().extract(call).candidates[0].fact.model_dump();d['source']=source
    with pytest.raises(PipelineError) as caught:
        validate_output(call,[Fact.model_validate(d)])
    assert caught.value.module == 'output'
    assert caught.value.code == ('INVALID_SOURCE' if source['turn_idx'] == 99 else 'QUOTE_WITHOUT_SPAN' if source.get('span') is None else 'INVALID_SPAN')

def test_injected_adapter_and_normalizer():
    call=CallInput.model_validate(EXAMPLE)
    resources=ModelResources('/models/test',object(),{'max_new_tokens':32})
    def bridge(input_call, supplied):
        assert input_call==call and supplied is resources
        assert 'facts' not in input_call.model_dump()
        return MockExtractor().extract(input_call)
    class DemoNormalizer:
        def normalize(self,batch,input_call):
            assert input_call is call
            return batch.model_copy(update={'warnings':batch.warnings+('normalizer-called',)})
    prediction,warnings=Pipeline(InjectedAdapter(resources,bridge),(DemoNormalizer(),)).run(call)
    assert len(prediction.facts)==2 and 'normalizer-called' in warnings

def test_failure_no_fallback():
    class Broken:
        def extract(self,call):raise RuntimeError('SECRET input')
    with TestClient(create_app(pipeline=Pipeline(Broken()))) as c:
        r=c.post('/extract',json=EXAMPLE)
        assert r.status_code==503 and 'SECRET' not in r.text and 'prediction' not in r.json()

def test_busy_worker():
    import threading
    from concurrent.futures import ThreadPoolExecutor
    entered,release=threading.Event(),threading.Event()
    class Blocking:
        def extract(self,call):
            entered.set()
            assert release.wait(5)
            return MockExtractor().extract(call)
    with TestClient(create_app(pipeline=Pipeline(Blocking()))) as c:
        with ThreadPoolExecutor() as pool:
            pending=pool.submit(c.post,'/extract',json=EXAMPLE)
            try:
                assert entered.wait(5)
                r=c.post('/extract',json=EXAMPLE)
                assert r.status_code==429
            finally:
                release.set()
            assert pending.result().status_code==200

def test_official_required_fields(client):
    prediction=client.post('/extract',json=EXAMPLE).json()['prediction']
    assert prediction['call_id']==EXAMPLE['call_id']
    for fact in prediction['facts']:
        assert {'attribute','value','polarity','valid_from','valid_to','source','object'} <= fact.keys()
        assert fact['object']['name'] and 'turn_idx' in fact['source']
    d=deepcopy(EXAMPLE);d['started_at']='2026-11-13T12:00:00+03:00'
    assert client.post('/extract',json=d).json()['prediction']['started_at']==d['started_at']
