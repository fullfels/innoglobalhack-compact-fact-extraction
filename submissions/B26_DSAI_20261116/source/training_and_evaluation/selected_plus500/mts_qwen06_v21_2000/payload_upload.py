"""Notebook upload helper: content identity, repeatable verification, no overwrite."""
from pathlib import Path
import hashlib
import io
import json
import zipfile


def prepare_payload(root, uploaded, expected_zip_sha, expected_manifest_sha):
    root = Path(root).resolve()
    if not root.exists() or not any(root.iterdir()):
        matches = [b for b in (uploaded or {}).values()
                   if hashlib.sha256(b).hexdigest() == expected_zip_sha]
        if not matches:
            raise ValueError('Среди загруженных файлов нет выданного payload: контрольная сумма отличается')
        with zipfile.ZipFile(io.BytesIO(matches[0])) as z:
            if z.testzip() is not None or sum(i.file_size for i in z.infolist()) >= 256 * 1024**2:
                raise ValueError('Повреждённый или слишком большой payload')
            for item in z.infolist():
                path = (root.parent / item.filename).resolve()
                if not path.is_relative_to(root) or (item.external_attr >> 16) & 0o170000 == 0o120000:
                    raise ValueError('Недопустимый путь в payload')
            z.extractall(root.parent)
    manifest_path = root/'manifest.json'
    if not manifest_path.is_file() or hashlib.sha256(manifest_path.read_bytes()).hexdigest() != expected_manifest_sha:
        raise ValueError('Manifest отсутствует или изменён. Папка не перезаписана; сохраните результаты и восстановите payload в новой среде')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    for rel, expected in manifest['sha256'].items():
        path = (root/rel).resolve()
        if not path.is_relative_to(root) or not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError('Файл отсутствует или изменён, папка не перезаписана: '+rel)
    return manifest
