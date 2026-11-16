"""Packaging/identity preflight; never imports Torch or loads a model.

This does not replace ML artifact validation or establish model readiness.
"""
import importlib.metadata
import json
import os
from pathlib import Path
import re
import sys

from src.experiments.common import read_json, sha


class ProfileBlocked(ValueError):
    pass


def check_profile(env, *, package_version=importlib.metadata.version, root=None):
    root = Path(root or Path(__file__).resolve().parents[1])
    def require(condition, code):
        if not condition:
            raise ProfileBlocked(code)
    require(sys.version_info[:2] == (3, 12), 'PYTHON_312_REQUIRED')
    require(env.get('MWS_MODE') == 'real', 'REAL_MODE_REQUIRED')
    require(env.get('MWS_PIPELINE_FACTORY') == 'ml.qwen.runtime:factory', 'UNAPPROVED_FACTORY')
    require(env.get('MTS_MODEL_DEVICE', 'cpu') in {'cpu', 'cuda'}, 'DEVICE_PROFILE_NOT_APPROVED')
    require(env.get('HF_HUB_OFFLINE') == '1', 'OFFLINE_MODE_REQUIRED')
    require(bool(env.get('MTS_MODEL_DIR')), 'MODEL_DIR_REQUIRED')
    export = Path(env['MTS_MODEL_DIR'])
    require(export.is_dir(), 'MODEL_EXPORT_MISSING')
    manifest_path = export / 'model_manifest.json'
    require(manifest_path.is_file(), 'MODEL_MANIFEST_MISSING')
    expected = env.get('MWS_ARTIFACT_SHA256', '')
    require(re.fullmatch('[0-9a-fA-F]{64}', expected) is not None, 'EXPECTED_HASH_REQUIRED')
    require(sha(manifest_path) == expected.lower(), 'MANIFEST_HASH_MISMATCH')
    try:
        manifest = read_json(manifest_path)
    except (ValueError, TypeError):
        raise ProfileBlocked('MODEL_MANIFEST_INVALID') from None
    require(isinstance(manifest, dict) and bool(env.get('MWS_MODEL_ID')) and
            manifest.get('model_id') == env['MWS_MODEL_ID'], 'MODEL_ID_MISMATCH')
    require(all((export / p).is_dir() for p in ['adapter', 'tokenizer']), 'EXPORT_LAYOUT_MISSING')
    require(all((export / p).is_file() for p in ['prompt.txt', 'catalog.json']), 'EXPORT_LAYOUT_MISSING')
    require(bool(env.get('HF_HOME')) and Path(env['HF_HOME']).is_dir(), 'OFFLINE_CACHE_MISSING')
    expected_torch = env.get('MWS_TORCH_VERSION', '')
    require(bool(expected_torch) and expected_torch != 'latest', 'TORCH_VERSION_REQUIRED')
    versions = {'torch': expected_torch}
    for path in [root / 'requirements.lock', root / 'ml/qwen/requirements.txt']:
        for raw in path.read_text().splitlines():
            line = raw.strip()
            if not line or line.startswith('#'):
                continue
            require(line.count('==') == 1, 'UNPINNED_REQUIREMENT')
            name, value = line.split('==')
            require(name not in versions or versions[name] == value, 'CONFLICTING_REQUIREMENT')
            versions[name] = value
    for name, expected_version in versions.items():
        try:
            actual = package_version(name)
        except importlib.metadata.PackageNotFoundError:
            raise ProfileBlocked('DEPENDENCY_MISSING') from None
        require(actual == expected_version, 'DEPENDENCY_VERSION_MISMATCH')
    return {'status': 'packaging_preflight_passed', 'model_id': manifest['model_id'],
            'manifest_sha256': expected.lower(), 'device': env.get('MTS_MODEL_DEVICE', 'cpu'),
            'torch_version': expected_torch, 'model_loaded': False, 'model_readiness_verified': False}


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    try:
        result = check_profile(os.environ)
        print(json.dumps(result), flush=True)
        if argv == ['--check-only']:
            return 0
        if not argv:
            raise ProfileBlocked('SERVER_COMMAND_REQUIRED')
        os.execvp(argv[0], argv)
    except Exception as exc:
        print(json.dumps({'status': 'blocked', 'code': str(exc) if isinstance(exc, ProfileBlocked)
                         else 'PREFLIGHT_FAILED', 'error_type': type(exc).__name__}), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
