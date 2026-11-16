"""HTTP protocol doubles only: these tests do not execute or measure Qwen."""
import io
import json
import pytest
from pydantic import ValidationError

from scripts.real_smoke import real_smoke
from src.experiments.common import ExperimentError, read_json
from src.pipeline.adapters import PipelineError


@pytest.fixture
def api(tmp_path, monkeypatch):
    call = {'call_id': 'independent-real-smoke-protocol-fixture', 'started_at': '2026-11-14T12:00:00',
            'turns': [{'idx': 7, 'speaker': 'client', 'text': 'Проверочный диалог.'}]}
    payload = {'prediction': {**call, 'facts': []}, 'meta': {'request_id': 'protocol-fixture',
               'model_id': 'fixture-engine', 'manifest_id': 'fixture-manifest', 'mode': 'real',
               'elapsed_ms': 1, 'warnings': [], 'schema_valid': True, 'is_mock': False}}
    info = {'model_id': 'fixture-engine', 'mode': 'real', 'is_mock': False, 'model_state': 'ready',
            'hash': 'a' * 64, 'manifest_id': 'fixture-manifest'}
    health = {'ready': True, 'mode': 'real', 'is_mock': False}
    requests = []

    def open_request(request, *, timeout):
        requests.append(request)
        if request.data:
            assert 'facts' not in json.loads(request.data)
        return io.BytesIO(json.dumps({'/health': health, '/model/info': info, '/extract': payload}[request.full_url.removeprefix('http://fixture')]).encode())

    monkeypatch.setattr('scripts.real_smoke.urlopen', open_request)
    source = tmp_path / 'input.json'
    source.write_text(json.dumps(call))
    options = dict(url='http://fixture', input_path=source, out=tmp_path / 'out',
                   model_id='fixture-engine', artifact_sha256='a' * 64)
    return options, payload, info, health, requests


def test_real_api_protocol_exports_validated_prediction_once(api):
    options, _, _, _, requests = api
    result = real_smoke(**options)
    assert result.call_id == 'independent-real-smoke-protocol-fixture'
    assert len([r for r in requests if r.data]) == 1
    assert read_json(options['out'] / 'manifest.json')['status'] == 'complete'
    assert read_json(options['out'] / 'prediction.json')['facts'] == []
    with pytest.raises(FileExistsError):
        real_smoke(**options)
    assert len(requests) == 3


@pytest.mark.parametrize('damage', ['mock', 'not_ready', 'hash', 'call_id', 'meta', 'manifest', 'gold', 'grounding'])
def test_invalid_or_mock_api_never_exports_real_prediction(api, damage):
    options, payload, info, health, _ = api
    if damage == 'mock': health.update(mode='mock', is_mock=True)
    elif damage == 'not_ready': health['ready'] = False
    elif damage == 'hash': info['hash'] = 'b' * 64
    elif damage == 'call_id': payload['prediction']['call_id'] = 'foreign'
    elif damage == 'meta': payload['meta']['is_mock'] = True
    elif damage == 'manifest': payload['meta']['manifest_id'] = 'foreign'
    elif damage == 'gold':
        value=read_json(options['input_path']);value['facts']=[]
        options['input_path'].write_text(json.dumps(value))
    elif damage == 'grounding':
        payload['prediction']['facts'] = [{'object': {'name': 'Интернет'}, 'attribute': 'service_state',
            'value': 'работает', 'polarity': 1, 'speaker': 'client', 'valid_from': '2026-11-14T12:00:00',
            'source': {'turn_idx': 99}}]
    with pytest.raises((ExperimentError, ValidationError, PipelineError)):
        real_smoke(**options)
    assert not (options['out'] / 'prediction.json').exists()
    assert read_json(options['out'] / 'manifest.json')['status'] == 'failed'


def test_missing_normalizers_requires_explicit_raw_experiment(api):
    options, payload, _, _, _ = api
    payload['meta']['warnings'] = ['DE normalizers not connected']
    with pytest.raises(ExperimentError):
        real_smoke(**options)
    options['out'] = options['out'].with_name('raw')
    real_smoke(**options, raw_model=True)
    manifest = read_json(options['out'] / 'manifest.json')
    assert manifest['experiment_type'] == 'raw-model'
    assert manifest['de_normalizers_connected'] is False
