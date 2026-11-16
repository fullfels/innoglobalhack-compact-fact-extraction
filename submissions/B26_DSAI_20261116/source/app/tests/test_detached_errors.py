import asyncio
import json
import logging
import threading
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from src.api.app import create_app
from src.pipeline.service import Pipeline

CALL = {'call_id':'detached', 'started_at':'2026-11-13T12:00:00', 'turns':[{'idx':7,'speaker':'client','text':'PRIVATE TRANSCRIPT'}]}


def test_detached_failure_is_logged_once_and_keeps_slot(caplog):
    from src.contracts.models import CallInput
    async def scenario():
        entered, release = threading.Event(), threading.Event()
        class Broken:
            def extract(self, call):
                entered.set()
                assert release.wait(5)
                raise RuntimeError('PRIVATE TRANSCRIPT token=secret /private/model')
        app = create_app(pipeline=Pipeline(Broken()))
        endpoint = next(r.endpoint for r in app.routes if getattr(r,'path',None) == '/extract')
        request = SimpleNamespace(state=SimpleNamespace(request_id='request-correlation-123'))
        call = CallInput.model_validate(CALL)
        waiter = asyncio.create_task(endpoint(call, request))
        try:
            assert await asyncio.to_thread(entered.wait, 2)
            waiter.cancel()
            with pytest.raises(asyncio.CancelledError): await waiter
            from src.pipeline.adapters import PipelineError
            with pytest.raises(PipelineError) as caught: await endpoint(call, request)
            assert caught.value.code == 'WORKER_BUSY'
            release.set()
            await app.state.worker.close()
            await asyncio.sleep(0)  # done callback has completed
        finally:
            release.set()
            await app.state.worker.close()
    with caplog.at_level(logging.ERROR, logger='mws.worker'):
        asyncio.run(scenario())
    records = [json.loads(r.message) for r in caplog.records if r.name == 'mws.worker']
    assert records == [{'event':'detached_inference_failed','request_id':'request-correlation-123',
                        'module':'extractor','error_code':'STAGE_FAILED','error_type':'RuntimeError'}]
    assert 'PRIVATE TRANSCRIPT' not in caplog.text
    assert 'token=secret' not in caplog.text and '/private/model' not in caplog.text


def test_active_waiter_error_is_not_logged_twice(caplog):
    class Broken:
        def extract(self, call): raise ValueError('sensitive detail')
    with caplog.at_level(logging.WARNING), TestClient(create_app(pipeline=Pipeline(Broken()))) as client:
        r = client.post('/extract', json=CALL)
    assert r.status_code == 503
    assert r.json()['error']['module'] == 'extractor'
    events = [json.loads(rec.message) for rec in caplog.records if rec.name in {'mws','mws.worker'}]
    assert len(events) == 1 and events[0]['error_type'] == 'ValueError'
    assert events[0]['request_id'] == r.json()['error']['request_id']
    assert 'sensitive detail' not in caplog.text + r.text


def test_completion_before_waiter_cancellation_is_not_lost(caplog):
    from src.api.worker import InferenceWorker
    async def scenario():
        entered, release = threading.Event(), threading.Event()
        class Broken:
            def extract(self, call):
                entered.set()
                assert release.wait(5)
                raise RuntimeError('sensitive details')
        worker = InferenceWorker(Pipeline(Broken()))
        waiter = asyncio.create_task(worker.run(CALL, request_id='completion-race'))
        try:
            assert await asyncio.to_thread(entered.wait, 2)
            # Registered after worker's consume callback: it first sees an active
            # waiter, then cancellation must log the already completed failure.
            worker.task.add_done_callback(lambda _: waiter.cancel())
            release.set()
            with pytest.raises(asyncio.CancelledError): await waiter
            await worker.close()
        finally:
            release.set()
            await worker.close()
    with caplog.at_level(logging.ERROR, logger='mws.worker'):
        asyncio.run(scenario())
    events = [json.loads(r.message) for r in caplog.records if r.name == 'mws.worker']
    assert len(events) == 1 and events[0]['request_id'] == 'completion-race'
    assert events[0]['error_type'] == 'RuntimeError'
    assert 'sensitive details' not in caplog.text


def test_cancelled_shutdown_still_drains_physical_work():
    from src.api.worker import InferenceWorker
    async def scenario():
        entered,release=threading.Event(),threading.Event()
        class Slow:
            def run(self,call):entered.set();assert release.wait(5);return 'done'
        worker=InferenceWorker(Slow());waiter=asyncio.create_task(worker.run(CALL))
        try:
            assert await asyncio.to_thread(entered.wait,1)
            shutdown=asyncio.create_task(worker.close())
            await asyncio.sleep(0)
            shutdown.cancel();await asyncio.sleep(0);await asyncio.sleep(0)
            assert not shutdown.done(), 'cancelled drain returned while native inference is still running'
            shutdown.cancel();await asyncio.sleep(0)
            assert not shutdown.done()
            release.set()
            assert await waiter=='done'
            with pytest.raises(asyncio.CancelledError):await shutdown
        finally:
            release.set();await worker.close()
    asyncio.run(scenario())
