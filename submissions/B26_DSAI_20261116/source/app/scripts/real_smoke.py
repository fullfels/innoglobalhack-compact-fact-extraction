"""Check a real API, or reload an approved ML factory with --factory.

Run with python -m scripts.real_smoke. Outputs contain private transcript data.
"""
import argparse
import json
import math
from pathlib import Path
import re
import sys
from urllib.request import Request, urlopen

from src.contracts.models import CallInput, ExtractResponse
from src.experiments.common import ExperimentError, identity, read_json, sha, write_json
from src.pipeline.service import validate_output


def real_smoke(*, url, input_path, out, model_id, artifact_sha256, raw_model=False, timeout_sec=120,
               request_json=None):
    if not re.fullmatch('[0-9a-fA-F]{64}', artifact_sha256):
        raise ExperimentError('expected artifact SHA256 required')
    if not math.isfinite(timeout_sec) or timeout_sec <= 0:
        raise ExperimentError('positive finite HTTP timeout required')
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    manifest = {**identity(), 'experiment_type': 'raw-model' if raw_model else 'pipeline',
                'model_id': model_id, 'mode': 'real', 'expected_artifact_sha256': artifact_sha256,
                'status': 'running', 'transport': 'in-process-asgi' if request_json else 'http'}

    def http(path, payload=None):
        if request_json is not None:
            return request_json(path, payload)
        request = Request(url.rstrip('/') + path,
                          data=json.dumps(payload).encode('utf-8') if payload is not None else None,
                          headers={'Content-Type': 'application/json'})
        with urlopen(request, timeout=timeout_sec) as response:
            return json.load(response)

    try:
        # Unlike offline gold loading, this CLI accepts features only, never labels.
        call = CallInput.model_validate(read_json(input_path))
        manifest['input_sha256'] = sha(input_path)
        health = http('/health')
        if health.get('ready') is not True or health.get('mode') != 'real' or health.get('is_mock') is not False:
            raise ExperimentError('real model is not ready')
        info = http('/model/info')
        if (info.get('mode') != 'real' or info.get('is_mock') is not False or
                info.get('model_state') != 'ready' or info.get('model_id') != model_id or
                str(info.get('hash', '')).lower() != artifact_sha256.lower()):
            raise ExperimentError('real model identity/artifact mismatch')
        response = ExtractResponse.model_validate(http('/extract', call.model_dump(mode='json')))
        if (response.meta.mode != 'real' or response.meta.is_mock is not False or
                response.meta.model_id != model_id or response.meta.schema_valid is not True):
            raise ExperimentError('extraction metadata mismatch')
        if response.meta.manifest_id != info.get('manifest_id'):
            raise ExperimentError('extraction manifest mismatch')
        prediction = response.prediction
        if CallInput.model_validate(prediction.model_dump(exclude={'facts'})) != call:
            raise ExperimentError('prediction changed input')
        validate_output(call, prediction.facts)
        missing_de = 'DE normalizers not connected' in response.meta.warnings
        if missing_de and not raw_model:
            raise ExperimentError('explicit raw-model experiment required without DE normalizers')
        write_json(out / 'prediction.json', prediction.model_dump(mode='json'))
        write_json(out / 'response.json', response.model_dump(mode='json'))
        write_json(out / 'model_info.json', info)
        write_json(out / 'validation_report.json', {'valid': True, 'schema_valid': True,
                   'source_grounding_valid': True, 'input_preserved': True,
                   'de_normalizers_connected': not missing_de, 'semantic_quality_evaluated': False})
        manifest.update(status='complete', artifact_sha256=info['hash'], manifest_id=info.get('manifest_id'),
                        de_normalizers_connected=not missing_de, prediction_sha256=sha(out / 'prediction.json'))
        return prediction
    except Exception as exc:
        manifest.update(status='failed', error_type=type(exc).__name__)
        write_json(out / 'validation_report.json', {'valid': False, 'error_type': type(exc).__name__})
        raise
    finally:
        write_json(out / 'manifest.json', manifest)


def offline_smoke(*, pipeline_factory, **options):
    """Reload the actual factory in one process and exercise the existing API.

    No replacement engine: ML owns the factory and its offline loading policy.
    Lifespan loads once and releases before this function returns, including on failure.
    """
    from fastapi.testclient import TestClient
    from src.api.app import create_app
    from src.api.config import Settings
    from src.experiments.common import factory
    pipeline = factory(pipeline_factory)()
    descriptor = pipeline.extractor.descriptor
    if (descriptor.mode != 'real' or descriptor.model_id != options['model_id'] or
            str(getattr(pipeline.extractor, 'public_info', {}).get('hash', '')).lower() != options['artifact_sha256'].lower()):
        raise ExperimentError('offline factory identity/artifact mismatch before load')
    settings = Settings(mode='real', inference_timeout_sec=options.get('timeout_sec', 120))
    with TestClient(create_app(settings, pipeline)) as client:
        def request_json(path, payload):
            response = client.get(path) if payload is None else client.post(path, json=payload)
            response.raise_for_status()
            return response.json()
        return real_smoke(url='http://in-process', request_json=request_json, **options)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    transport = parser.add_mutually_exclusive_group()
    transport.add_argument('--url', help='Running API; default http://127.0.0.1:8000')
    transport.add_argument('--factory', help='Trusted no-argument ML pipeline factory for offline reload/ASGI acceptance')
    for name in ['input', 'out', 'model-id', 'artifact-sha256']:
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--raw-model', action='store_true')
    parser.add_argument('--timeout-sec', type=float, default=120)
    args = parser.parse_args()
    try:
        options = dict(input_path=args.input, out=args.out, model_id=args.model_id,
                       artifact_sha256=args.artifact_sha256, raw_model=args.raw_model, timeout_sec=args.timeout_sec)
        if args.factory:
            offline_smoke(pipeline_factory=args.factory, **options)
        else:
            real_smoke(url=args.url or 'http://127.0.0.1:8000', **options)
        print('PASS: real API identity, schema and source checks; quality not evaluated')
        return 0
    except Exception as exc:
        print('Real API smoke failed: ' + type(exc).__name__, file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
