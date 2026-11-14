"""Synthetic content only; no organizer data, weights or ML imports."""
import hashlib
import json
import pytest

from scripts.verify_archived_provenance import verify_archive


@pytest.fixture
def archive(tmp_path):
    paths = ['source/predict.py', 'source/common.py', 'source/prompt.txt',
             'notebooks/MTS_02_Qwen35_Check.ipynb']
    hashes = []
    for name in paths:
        p = tmp_path / name
        p.parent.mkdir(exist_ok=True)
        raw = b'Independent fixture\nSecond line\n'
        p.write_bytes(raw)
        executed = raw if name.endswith('prompt.txt') else raw.replace(b'\n', b'\r\n')
        hashes.append(hashlib.sha256(executed).hexdigest())
    (tmp_path / 'results').mkdir()
    (tmp_path / 'results/run_manifest.json').write_text(json.dumps({'fingerprint': {
        'script_sha256': hashes[0], 'common_sha256': hashes[1], 'prompt_sha256': hashes[2]}}))
    (tmp_path / 'provenance.json').write_text(json.dumps({'source_artifact_sha256': {
        'MTS_02_Qwen35_Check.ipynb': hashes[3]}}))
    return tmp_path, paths


def test_declared_crlf_and_git_lf_are_reported_separately(archive):
    root, _ = archive
    result = verify_archive(root)
    assert result['status'] == 'content_consistent'
    assert [r['match_policy'] for r in result['checks']].count('git-LF-to-executed-CRLF') == 3
    assert result['gpu_run_authenticated'] is False


@pytest.mark.parametrize('index', range(4))
def test_code_prompt_or_notebook_mutation_is_rejected(archive, index):
    root, paths = archive
    p = root / paths[index]
    p.write_bytes(p.read_bytes() + b'TAMPER SENTINEL\n')
    with pytest.raises(ValueError, match='hash mismatch'):
        verify_archive(root)


def test_exact_archived_bytes_also_pass(archive):
    root, paths = archive
    for name in paths:
        if not name.endswith('prompt.txt'):
            p = root / name
            p.write_bytes(p.read_bytes().replace(b'\n', b'\r\n'))
    assert all(r['match_policy'] == 'exact-bytes' for r in verify_archive(root)['checks'])


def test_prompt_line_endings_are_not_silently_changed(archive):
    root, _ = archive
    p = root / 'source/prompt.txt'
    p.write_bytes(p.read_bytes().replace(b'\n', b'\r\n'))
    with pytest.raises(ValueError):
        verify_archive(root)
