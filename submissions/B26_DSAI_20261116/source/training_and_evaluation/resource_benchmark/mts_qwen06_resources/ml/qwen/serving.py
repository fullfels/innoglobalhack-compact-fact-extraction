"""v23 export -> existing API/Pipeline; no new parser or extraction policy."""
from pathlib import Path
import gc
import json
import os
import re
import traceback

from ml.qwen.bridge import strict_json
from ml.qwen.final_runtime import MODEL, REVISION, Engine, load_model, sha
from ml.qwen.final_bridge import FinalBridge
from ml.qwen.postprocessing import TariffValueNormalizer, POLICY_ID
from src.pipeline.adapters import ModelResources, ModelDescriptor, PipelineError
from src.pipeline.lifecycle import ManagedAdapter, LoadedModel
from src.pipeline.service import Pipeline

POLICY_FILES = (
    'ml/qwen/bridge.py', 'ml/qwen/final_runtime.py', 'ml/qwen/final_bridge.py',
    'ml/qwen/calendar_v23.py', 'ml/qwen/date_anchors_v21.py',
    'ml/qwen/temporal_v21.py', 'src/contracts/models.py',
    'src/pipeline/adapters.py', 'src/pipeline/lifecycle.py', 'src/pipeline/service.py',
)
REQUIRED = {
    'adapter/adapter_model.safetensors', 'adapter/adapter_config.json',
    'tokenizer/tokenizer.json', 'tokenizer/tokenizer_config.json',
    'tokenizer/chat_template.jinja', 'prompt.txt', 'catalog.json', *POLICY_FILES,
}


def blocked(code):
    raise PipelineError(code, 503, 'qwen.artifact')


def verify_export(path, *, expected_hash=None, expected_model_id=None, app_root=None):
    root = Path(path).resolve()
    manifest_path = root/'model_manifest.json'
    if not manifest_path.is_file(): blocked('EXPORT_MANIFEST_MISSING')
    if expected_hash is not None and (not re.fullmatch(r'[0-9a-f]{64}', expected_hash)
                                     or sha(manifest_path) != expected_hash):
        blocked('EXPORT_MANIFEST_HASH_MISMATCH')
    try:
        m = strict_json(manifest_path.read_text(encoding='utf-8'))
    except (ValueError, TypeError): blocked('EXPORT_MANIFEST_INVALID')
    if not isinstance(m, dict): blocked('EXPORT_MANIFEST_INVALID')
    if m.get('base_model') != MODEL or m.get('base_revision') != REVISION:
        blocked('EXPORT_BASE_MISMATCH')
    if m.get('reload_verified') is not True: blocked('EXPORT_RELOAD_NOT_VERIFIED')
    reload = m.get('reload_test')
    if not isinstance(reload, dict) or reload.get('passed') is not True or reload.get('fresh_base_and_adapter') is not True:
        blocked('EXPORT_RELOAD_NOT_VERIFIED')
    if not isinstance(m.get('model_id'), str) or not m['model_id']:
        blocked('EXPORT_ID_MISSING')
    if expected_model_id is not None and expected_model_id != m['model_id']:
        blocked('EXPORT_ID_MISMATCH')
    if type(m.get('calendar_enabled')) is not bool: blocked('EXPORT_CALENDAR_INVALID')
    if m.get('context') != 24576 or m.get('max_new_tokens') != 2048:
        blocked('EXPORT_INFERENCE_POLICY_MISMATCH')
    if type(m.get('total_parameters')) is not int or m['total_parameters'] <= 0:
        blocked('EXPORT_PARAMETER_COUNT_INVALID')
    inventory = m.get('sha256')
    if not isinstance(inventory, dict) or not REQUIRED <= inventory.keys():
        blocked('EXPORT_INVENTORY_INCOMPLETE')
    for relative, expected in inventory.items():
        if not isinstance(relative, str) or not isinstance(expected, str) or not re.fullmatch(r'[0-9a-f]{64}', expected):
            blocked('EXPORT_INVENTORY_INVALID')
        p = (root/relative).resolve()
        if not p.is_relative_to(root) or not p.is_file() or sha(p) != expected:
            blocked('EXPORT_FILE_HASH_MISMATCH')
    if reload.get('adapter_sha256') != inventory['adapter/adapter_model.safetensors']:
        blocked('EXPORT_RELOAD_ADAPTER_MISMATCH')
    used_files = {p.relative_to(root).as_posix() for folder in ['adapter', 'tokenizer']
                  for p in (root/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    if not used_files <= inventory.keys(): blocked('EXPORT_UNLISTED_RUNTIME_FILE')
    installed = Path(app_root or Path(__file__).resolve().parents[2])
    for relative in POLICY_FILES:
        p = installed/relative
        if not p.is_file() or sha(p) != inventory[relative]: blocked('EXPORT_INSTALLED_POLICY_MISMATCH')
    return root, m


def verify_base(path):
    root = Path(path).resolve()
    try:
        identity = strict_json((root/'base_identity.json').read_text(encoding='utf-8'))
    except (OSError, ValueError, TypeError): blocked('BASE_IDENTITY_MISSING_OR_INVALID')
    if identity.get('repo_id') != MODEL or identity.get('revision') != REVISION:
        blocked('BASE_REVISION_MISMATCH')
    inventory = identity.get('sha256')
    if not isinstance(inventory, dict) or 'config.json' not in inventory or not any(x.endswith('.safetensors') for x in inventory):
        blocked('BASE_INVENTORY_INCOMPLETE')
    for relative, expected in inventory.items():
        p=(root/relative).resolve()
        if not p.is_relative_to(root) or not p.is_file() or sha(p) != expected:
            blocked('BASE_HASH_MISMATCH')
    if strict_json((root/'config.json').read_text(encoding='utf-8')).get('model_type') != 'qwen3':
        blocked('BASE_ARCHITECTURE_MISMATCH')
    return root


def configuration(config=None):
    allowed = {'model_dir', 'base_path', 'device', 'precision', 'manifest_sha256', 'model_id', 'max_call_seconds'}
    c = dict(config or {})
    if set(c) - allowed: blocked('SERVING_UNKNOWN_CONFIGURATION')
    values = {
        'model_dir': c.get('model_dir', os.getenv('MTS_MODEL_DIR')),
        'base_path': c.get('base_path', os.getenv('MTS_MODEL_BASE_PATH')),
        'device': c.get('device', os.getenv('MTS_MODEL_DEVICE', 'cpu')),
        'precision': c.get('precision', 'fp16'),
        'manifest_sha256': c.get('manifest_sha256', os.getenv('MWS_ARTIFACT_SHA256')),
        'model_id': c.get('model_id', os.getenv('MWS_MODEL_ID')),
    }
    if not values['model_dir']: blocked('SERVING_MODEL_DIR_REQUIRED')
    if values['device'] not in {'cpu', 'cuda'}: blocked('SERVING_DEVICE_UNSUPPORTED')
    if values['precision'] not in {'fp16', 'int8', 'int4'}: blocked('SERVING_PRECISION_UNSUPPORTED')
    if values['device'] == 'cpu' and values['precision'] != 'fp16': blocked('SERVING_CPU_QUANTIZATION_UNSUPPORTED')
    limit = c.get('max_call_seconds')
    if limit is None:
        raw = os.getenv('MWS_MODEL_MAX_CALL_SECONDS')
        if raw is None:
            limit = 600 if values['device'] == 'cpu' else 120
        else:
            try: limit = int(raw)
            except (TypeError, ValueError): blocked('SERVING_GENERATION_LIMIT_INVALID')
    if type(limit) is not int or not 1 <= limit <= 3600:
        blocked('SERVING_GENERATION_LIMIT_INVALID')
    values['max_call_seconds'] = limit
    return values


def build(config=None):
    c = configuration(config)
    root, m = verify_export(c['model_dir'], expected_hash=c['manifest_sha256'], expected_model_id=c['model_id'])
    base = verify_base(c['base_path']) if c['base_path'] else None
    # Defaults match the verified Colab pipeline. No hidden DE rules are attached.
    def loader(resources):
        model = tokenizer = engine = bridge = None
        previous = os.environ.get('MTS_MODEL_BASE_PATH')
        try:
            if base: os.environ['MTS_MODEL_BASE_PATH'] = str(base)
            model, tokenizer = load_model(root/'adapter', root/'tokenizer', precision=c['precision'],
                                           local_only=True, device=c['device'])
            model.eval()
            engine = Engine(model, tokenizer, (root/'prompt.txt').read_text(encoding='utf-8'),
                            strict_json((root/'catalog.json').read_text(encoding='utf-8')), device=c['device'],
                            max_call_seconds=c['max_call_seconds'])
            bridge = FinalBridge(engine, calendar=m['calendar_enabled'])
        except Exception as exc:
            error_type = type(exc).__name__
            traceback.clear_frames(exc.__traceback__)
            model = tokenizer = engine = bridge = None
            gc.collect()
            if c['device'] == 'cuda':
                import torch
                torch.cuda.empty_cache()
            raise PipelineError('QWEN_RELEASE_LOAD_FAILED', 503, 'qwen.loader', error_type) from None
        finally:
            if previous is None: os.environ.pop('MTS_MODEL_BASE_PATH', None)
            else: os.environ['MTS_MODEL_BASE_PATH'] = previous

        def close():
            nonlocal model, tokenizer, engine, bridge
            if engine is not None:
                engine.model = engine.tokenizer = None
            if bridge is not None: bridge.engine = None
            model = tokenizer = engine = bridge = None
            gc.collect()
            if c['device'] == 'cuda':
                import torch
                torch.cuda.empty_cache()
        return LoadedModel(bridge, close)

    dtype = 'float32' if c['device'] == 'cpu' else {'fp16':'float16', 'int8':'int8', 'int4':'nf4'}[c['precision']]
    adapter = ManagedAdapter(ModelResources(str(root), str(root/'tokenizer'), c), loader,
        descriptor=ModelDescriptor(m['model_id'], 'real', 'qwen3-v23-'+c['precision']),
        revision=REVISION, artifact_sha256=sha(root/'model_manifest.json'), dtype=dtype,
        context=m['context'], parameters=m['total_parameters'], manifest_id=m['model_id'])
    adapter.public_info['max_call_seconds'] = c['max_call_seconds']
    adapter.public_info['postprocessing_policy'] = POLICY_ID
    catalog = strict_json((root/'catalog.json').read_text(encoding='utf-8'))
    return Pipeline(adapter, normalizers=(TariffValueNormalizer(catalog),))


def factory():
    return build()


def evaluation_factory(config):
    return build(config)


def benchmark_factory(config):
    from src.experiments.benchmark import BenchmarkTarget
    c = configuration(config)
    if not c['base_path']: blocked('BENCHMARK_MATERIALIZED_BASE_REQUIRED')
    pipeline = build(c)
    pipeline.extractor.load()
    return BenchmarkTarget(pipeline, c['device'], synthetic=False,
                           parameters=pipeline.extractor.public_info['parameters'],
                           checkpoint_paths=(str(Path(c['base_path'])), str(Path(c['model_dir'])/'adapter')),
                           close=pipeline.extractor.close)
