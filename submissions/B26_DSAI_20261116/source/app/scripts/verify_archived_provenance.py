"""Verify PR #52 archive content without importing ML code or executing notebooks.

EOL equivalence is explicit and limited to the Python sources and notebook.
This checks consistency with supplied manifests, not authenticity of a GPU run.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re

from src.experiments.common import read_json


def verify_archive(root):
    root = Path(root)
    fingerprint = read_json(root / 'results/run_manifest.json')['fingerprint']
    provenance = read_json(root / 'provenance.json')['source_artifact_sha256']
    files = [
        ('source/predict.py', fingerprint['script_sha256'], True),
        ('source/common.py', fingerprint['common_sha256'], True),
        ('source/prompt.txt', fingerprint['prompt_sha256'], False),
        ('notebooks/MTS_02_Qwen35_Check.ipynb', provenance['MTS_02_Qwen35_Check.ipynb'], True),
    ]
    checks = []
    for name, expected, allow_eol in files:
        if not isinstance(expected, str) or not re.fullmatch('[0-9a-f]{64}', expected):
            raise ValueError('Invalid declared SHA256: ' + name)
        raw = (root / name).read_bytes()
        actual = hashlib.sha256(raw).hexdigest()
        policy = 'exact-bytes'
        if actual != expected:
            # Git LF text can reconstruct the archived CRLF bytes. Do not trim
            # whitespace, ignore comments or normalise the prompt's contents.
            if (not allow_eol or b'\r' in raw or
                    hashlib.sha256(raw.replace(b'\n', b'\r\n')).hexdigest() != expected):
                raise ValueError('Archive content hash mismatch: ' + name)
            policy = 'git-LF-to-executed-CRLF'
        checks.append({'path': name, 'git_bytes_sha256': actual,
                       'declared_executed_sha256': expected, 'match_policy': policy})
    return {'status': 'content_consistent', 'checks': checks,
            'gpu_run_authenticated': False, 'private_zip_hashes_verified': False,
            'note': 'Manifest-relative content check only; metrics require original evaluator separately.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', required=True)
    args = parser.parse_args()
    try:
        print(json.dumps(verify_archive(args.archive), indent=2))
        return 0
    except Exception as exc:
        print(json.dumps({'status': 'failed', 'error_type': type(exc).__name__}))
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
