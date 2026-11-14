"""Actual ML bridge + FS lifecycle/API; controlled engines only, no Qwen accuracy claim."""
import json
import pytest
from fastapi.testclient import TestClient
from ml.qwen.bridge import QwenBridge
from scripts.real_smoke import offline_smoke
from src.api.app import create_app
from src.api.config import Settings
from src.pipeline.adapters import ModelDescriptor, ModelResources
from src.pipeline.lifecycle import ManagedAdapter, LoadedModel
from src.pipeline.service import Pipeline

CALL = {'call_id': 'independent-qwen-contract', 'started_at': '2026-11-14T10:00:00',
        'turns': [{'idx': 7, 'speaker': 'client', 'text': 'Интернет не работает 😞.'}]}
FACT = {'object': {'name': 'Интернет', 'id': 'fixture-only'}, 'attribute': 'service_state',
        'value': 'не работает', 'polarity': 1, 'speaker': 'client', 'valid_from': CALL['started_at'],
        'valid_to': None, 'source': {'turn_idx': 7, 'quote': 'не работает 😞'}}


def controlled_pipeline(raw):
    events = []
    class Engine:
        def generate(self, call):
            assert not hasattr(call, 'facts')
            events.append('inference')
            if isinstance(raw, Exception): raise raw
            return raw
    def load(resources):
        events.append('load')
        return LoadedModel(QwenBridge(Engine()), lambda: events.append('close'))
    adapter = ManagedAdapter(ModelResources('fixture-no-checkpoint', None), load,
        descriptor=ModelDescriptor('protocol-only', 'real', 'controlled-test'),
        artifact_sha256='a' * 64, manifest_id='protocol-only')
    return Pipeline(adapter), events


def test_offline_reload_actual_bridge_api_loads_once_and_releases(tmp_path, monkeypatch):
    pipeline, events = controlled_pipeline(json.dumps({'facts': [FACT, {**FACT, 'value': 'обрывы'}]}))
    import src.experiments.common
    monkeypatch.setattr(src.experiments.common, 'factory', lambda name: lambda: pipeline)
    source = tmp_path / 'input.json'; source.write_text(json.dumps(CALL))
    result = offline_smoke(pipeline_factory='fixture:only', input_path=source, out=tmp_path/'out',
        model_id='protocol-only', artifact_sha256='a' * 64, raw_model=True)
    assert len(result.facts) == 2 and result.facts[0].source.turn_idx == 7
    assert events == ['load', 'inference', 'close']
    manifest = json.loads((tmp_path/'out/manifest.json').read_text())
    assert manifest['transport'] == 'in-process-asgi' and manifest['experiment_type'] == 'raw-model'


@pytest.mark.parametrize('raw,code', [
    ('{"facts":[', 'QWEN_INVALID_JSON'),
    (json.dumps({'facts': [{**FACT, 'valid_from': None}]}), 'QWEN_UNRESOLVED_TIME'),
    (json.dumps({'facts': [{**FACT, 'speaker': 'operator'}]}), 'QWEN_SPEAKER_SOURCE_MISMATCH'),
    (json.dumps({'facts': [{**FACT, 'polarity': True}]}), 'QWEN_INVALID_FACT'),
    (RuntimeError('SECRET_GENERATION_DETAIL'), 'QWEN_GENERATION_FAILED'),
])
def test_actual_bridge_errors_remain_failed_http_without_fallback(raw, code, caplog):
    pipeline, events = controlled_pipeline(raw)
    with TestClient(create_app(Settings(mode='real'), pipeline)) as client:
        response = client.post('/extract', json=CALL)
        assert response.status_code == 503
        assert response.json()['error']['code'] == code
        assert 'prediction' not in response.json()
    assert events == ['load', 'inference', 'close']
    assert 'SECRET_GENERATION_DETAIL' not in caplog.text


@pytest.mark.parametrize('mismatch', ['model_id', 'artifact_sha256'])
def test_wrong_offline_identity_rejected_before_engine_load(tmp_path, monkeypatch, mismatch):
    from src.experiments.common import ExperimentError
    import src.experiments.common
    pipeline, events = controlled_pipeline('{"facts":[]}')
    monkeypatch.setattr(src.experiments.common, 'factory', lambda name: lambda: pipeline)
    options = dict(model_id='protocol-only', artifact_sha256='a' * 64,
                   input_path=tmp_path/'unused.json', out=tmp_path/'out', raw_model=True)
    options[mismatch] = 'foreign' if mismatch == 'model_id' else 'b' * 64
    with pytest.raises(ExperimentError):offline_smoke(pipeline_factory='fixture:only', **options)
    assert events == [] and not (tmp_path/'out').exists()
