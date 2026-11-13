import asyncio
from datetime import datetime
import logging
import threading
from dataclasses import replace

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from src.api.app import create_app
from src.api.config import Settings
from src.api.worker import InferenceWorker
from src.contracts.models import CallInput, CandidateBatch, CandidateFact, Fact
from src.pipeline.adapters import InjectedAdapter, MockExtractor, ModelDescriptor, ModelResources, PendingRealAdapter, PipelineError
from src.pipeline.service import Pipeline, validate_output

CALL = CallInput(call_id='regression', started_at='2026-11-13T12:00:00', turns=(
    {'idx': 7, 'speaker': 'operator', 'text': 'Инженер приедет.'},
    {'idx': 12, 'speaker': 'client', 'text': '😀 Да, подходит!'},
))
FACT = Fact(object={'name': 'Интернет'}, attribute='service_state', value='не работает',
    polarity=1, speaker='operator', valid_from=CALL.started_at,
    source={'turn_idx': 7, 'span': (0, len(CALL.turns[0].text)), 'quote': CALL.turns[0].text})

class FixedExtractor:
    def __init__(self, batch): self.batch = batch
    def extract(self, call): return self.batch


def batch_for(fact=FACT, **kwargs):
    # Deliberately bypass validation exactly as an external normalizer could.
    c = CandidateFact(fact=FACT).model_copy(update={'fact': fact, **kwargs})
    return CandidateBatch(candidates=()).model_copy(update={'candidates': (c,)})

@pytest.mark.parametrize('patch', [
    {'polarity': 0}, {'polarity': True}, {'attribute': 'invalid'},
    {'valid_from': 'not-a-date'}, {'valid_from': 0},
    {'valid_to': datetime(2020, 1, 1)},
    {'object': FACT.object.model_copy(update={'name': ''})},
    {'object': FACT.object.model_copy(update={'type': 'invalid'})},
    {'source': FACT.source.model_copy(update={'turn_idx': -1})},
])
def test_copied_invalid_fact_never_returns_success(patch):
    damaged = FACT.model_copy(update=patch)
    with TestClient(create_app(pipeline=Pipeline(FixedExtractor(batch_for(damaged))))) as c:
        r = c.post('/extract', json=CALL.model_dump(mode='json'))
        assert r.status_code == 503
        assert 'meta' not in r.json() and 'prediction' not in r.json()
        assert r.json()['error']['module'] == 'extractor'
    with pytest.raises(PipelineError) as caught:
        validate_output(CALL, (damaged,))
    assert caught.value.code == 'INVALID_OUTPUT'

@pytest.mark.parametrize('field', ['valid_from', 'valid_to'])
@pytest.mark.parametrize('value', [0, 123.5, True, '2026-11-13', 'yesterday'])
def test_dates_require_iso_datetime(field, value):
    with pytest.raises(ValidationError):
        Fact.model_validate({**FACT.model_dump(), field: value})

@pytest.mark.parametrize('confirmation,attribute,evidence,expected', [
    ('rejected','service_state',(),0),
    ('unknown','service_state',(),1),
    ('confirmed','service_state',(),1),
    ('rejected','agreement',(12,),0),
    ('unknown','agreement',(12,),0),
    ('confirmed','agreement',(),0),
    ('confirmed','agreement',(7,),0),
    ('confirmed','agreement',(999,),0),
    ('confirmed','agreement',(12,),1),
])
def test_confirmation_export_policy(confirmation, attribute, evidence, expected):
    fact = FACT.model_copy(update={'attribute': attribute})
    result, warnings = Pipeline(FixedExtractor(batch_for(fact, confirmation=confirmation, evidence_turns=evidence))).run(CALL)
    assert len(result.facts) == expected
    assert any('Filtered' in w for w in warnings) == (expected == 0)


def test_normalizers_transform_in_order_and_preserve_original():
    seen = []
    class Change:
        def __init__(self, before, after): self.before, self.after = before, after
        def normalize(self, batch, call):
            assert call is CALL
            assert batch.candidates[0].fact.value == self.before
            seen.append(self.before)
            fact = batch.candidates[0].fact.model_copy(update={'value': self.after})
            return batch.model_copy(update={'candidates': (batch.candidates[0].model_copy(update={'fact': fact}),)})
    result, _ = Pipeline(FixedExtractor(batch_for()), (Change('не работает','обрывы'), Change('обрывы','периодические обрывы'))).run(CALL)
    assert seen == ['не работает','обрывы']
    assert result.facts[0].value == 'периодические обрывы'
    assert result.turns == CALL.turns and FACT.value == 'не работает'


def test_invalid_normalizer_output_and_exception_are_diagnostic(caplog):
    class BrokenNormalizer:
        def normalize(self, batch, call): raise RuntimeError('SECRET TRANSCRIPT')
    with caplog.at_level(logging.WARNING), TestClient(create_app(pipeline=Pipeline(FixedExtractor(batch_for()),(BrokenNormalizer(),)))) as c:
        r = c.post('/extract', json=CALL.model_dump(mode='json'))
    assert r.status_code == 503
    assert r.json()['error']['module'] == 'normalizer.BrokenNormalizer'
    assert 'RuntimeError' in caplog.text
    assert 'SECRET TRANSCRIPT' not in caplog.text + r.text


def test_normalizer_cannot_bypass_validation():
    class Corrupt:
        def normalize(self, batch, call): return batch_for(FACT.model_copy(update={'polarity': 0}))
    with pytest.raises(PipelineError) as e:
        Pipeline(FixedExtractor(batch_for()), (Corrupt(),)).run(CALL)
    assert e.value.module == 'normalizer.Corrupt'
    assert e.value.error_type == 'ValidationError'


def test_empty_batch_is_success():
    with TestClient(create_app(pipeline=Pipeline(FixedExtractor(CandidateBatch(candidates=()))))) as c:
        r = c.post('/extract', json=CALL.model_dump(mode='json'))
        assert r.status_code == 200 and r.json()['prediction']['facts'] == []
        assert r.json()['meta']['schema_valid']


def test_mode_disagreement_rejected():
    with pytest.raises(ValueError, match='disagrees'):
        create_app(Settings(mode='real'), Pipeline(MockExtractor()))
    with pytest.raises(ValueError, match='requires ModelDescriptor'):
        create_app(Settings(mode='real'), Pipeline(FixedExtractor(batch_for())))


def test_real_lifecycle_and_metadata():
    resources = ModelResources('/models/local-private-path', object(), {'max_new_tokens':32})
    def bridge(call, supplied):
        assert supplied is resources
        return batch_for()
    adapter = InjectedAdapter(resources, bridge, descriptor=ModelDescriptor('qwen-export-A','real','test-engine'))
    with TestClient(create_app(Settings(mode='real', model_id='ignored-label'), Pipeline(adapter))) as c:
        for state in ['unloaded','loading','failed']:
            adapter.set_state(state)
            assert c.get('/health').status_code == 503
            assert c.get('/health').json()['model_state'] == state
            assert c.post('/extract',json=CALL.model_dump(mode='json')).status_code == 503
        adapter.set_state('ready')
        h, info = c.get('/health').json(), c.get('/model/info').json()
        r = c.post('/extract', json=CALL.model_dump(mode='json')).json()
        assert h['ready'] and not h['is_mock']
        assert info['mode'] == r['meta']['mode'] == 'real'
        assert info['model_id'] == r['meta']['model_id'] == 'qwen-export-A'
        assert info['engine'] == 'test-engine' and not r['meta']['is_mock']
        assert '/models/local' not in str(info)


def test_pending_injected_pipeline_is_not_ready():
    with TestClient(create_app(Settings(mode='real'), Pipeline(PendingRealAdapter()))) as c:
        assert c.get('/health').status_code == 503


def test_invalid_adapter_result():
    adapter = InjectedAdapter(ModelResources('unused',None),lambda c,r:'{invalid JSON')
    with pytest.raises(PipelineError) as e:
        Pipeline(adapter).run(CALL)
    assert e.value.module == 'extractor' and e.value.error_type == 'ValidationError'


def test_unicode_nonzero_source_and_missing_reference():
    f = FACT.model_copy(update={'speaker':'client','source':{'turn_idx':12,'span':(2,4),'quote':'Да'}})
    result = validate_output(CALL,(f,))
    assert result.facts[0].source.quote == CALL.turns[1].text[2:4]
    for source, code in [({'turn_idx':99},'INVALID_SOURCE'), ({'turn_idx':12,'span':(0,999),'quote':'x'},'INVALID_SPAN')]:
        with pytest.raises(PipelineError) as e:
            validate_output(CALL,(f.model_copy(update={'source':source}),))
        assert e.value.code == code


def test_cancellation_keeps_physical_slot_and_shutdown_drains():
    async def scenario():
        entered = threading.Event()
        release = threading.Event()
        active = 0
        peak = 0
        calls = 0
        class Blocking:
            def extract(self, call):
                nonlocal active, peak, calls
                active += 1; peak = max(peak,active); calls += 1
                entered.set()
                try:
                    assert release.wait(5)
                    return CandidateBatch(candidates=())
                finally: active -= 1
        worker = InferenceWorker(Pipeline(Blocking()))
        first = asyncio.create_task(worker.run(CALL))
        try:
            assert await asyncio.to_thread(entered.wait,2)
            first.cancel()
            with pytest.raises(asyncio.CancelledError): await first
            for _ in range(2):
                with pytest.raises(PipelineError) as e: await worker.run(CALL)
                assert e.value.code == 'WORKER_BUSY'
            assert calls == 1 and active == 1
            release.set()
            await asyncio.shield(worker.task)
            await worker.run(CALL)
            assert calls == 2 and peak == 1
            # A shutdown must wait for outstanding physical work and reject new work.
            release.clear(); entered.clear()
            third = asyncio.create_task(worker.run(CALL))
            assert await asyncio.to_thread(entered.wait,2)
            closing = asyncio.create_task(worker.close())
            await asyncio.sleep(0)
            assert not closing.done()
            with pytest.raises(PipelineError) as e: await worker.run(CALL)
            assert e.value.code == 'WORKER_SHUTTING_DOWN'
            release.set()
            await third; await closing
            assert active == 0 and peak == 1
        finally:
            release.set()
            await worker.close()
    asyncio.run(scenario())


def test_http_endpoint_cancellation_uses_same_slot():
    from types import SimpleNamespace
    async def scenario():
        entered, release = threading.Event(), threading.Event()
        class Blocking:
            def extract(self, call):
                entered.set()
                assert release.wait(5)
                return CandidateBatch(candidates=())
        app = create_app(pipeline=Pipeline(Blocking()))
        endpoint = next(r.endpoint for r in app.routes if getattr(r,'path',None) == '/extract')
        request = SimpleNamespace(state=SimpleNamespace(request_id='cancellation-test'))
        first = asyncio.create_task(endpoint(CALL,request))
        try:
            assert await asyncio.to_thread(entered.wait,2)
            first.cancel()
            with pytest.raises(asyncio.CancelledError): await first
            with pytest.raises(PipelineError) as e: await endpoint(CALL,request)
            assert e.value.code == 'WORKER_BUSY'
            release.set()
            await app.state.worker.task
            result = await endpoint(CALL,request)
            assert result.meta.schema_valid and result.prediction.facts == ()
        finally:
            release.set()
            await app.state.worker.close()
    asyncio.run(scenario())


def test_mock_metadata_consistent():
    call = {'call_id':'mock','started_at':'2026-11-13T12:00:00','turns':[{'idx':1,'speaker':'client','text':'Интернет не работает.'}]}
    with TestClient(create_app()) as c:
        meta = c.post('/extract',json=call).json()['meta']
        health, info = c.get('/health').json(), c.get('/model/info').json()
        assert health['mode'] == info['mode'] == meta['mode'] == 'mock'
        assert health['is_mock'] and info['is_mock'] and meta['is_mock']
        assert meta['model_id'] == info['model_id'] == 'synthetic-fixtures-v1'
        assert 'model-banner' in c.get('/').text
