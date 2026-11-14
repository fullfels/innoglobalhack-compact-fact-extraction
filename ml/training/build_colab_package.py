"""Build the private Colab code transport; explicit code-only allowlist, no data/weights."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile


def build(root, output):
    paths = ["src/__init__.py", "src/contracts/__init__.py", "src/contracts/models.py",
             "src/pipeline/__init__.py", "src/pipeline/adapters.py", "src/pipeline/lifecycle.py", "src/pipeline/service.py",
             "ml/__init__.py", "ml/qwen/__init__.py", "ml/qwen/bridge.py", "ml/qwen/runtime.py",
             "ml/qwen/prompting.py", "ml/qwen/train.py", "ml/qwen/requirements.txt", "ml/qwen/INTEGRATION.md", "ml/qwen/README.md"]
    manifest = {"fs_contract_snapshot": "640dc29e6fc49869110fd291e3d445210d9d0aeb",
                "policy": "Explicit code-only transport; no corpus, organizer catalog/evaluator, weights or secrets.",
                "sha256": {p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in paths}}
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as z:
        for rel in paths: z.write(root / rel, "mts_ml_code/" + rel)
        z.writestr("mts_ml_code/code_package_manifest.json", json.dumps(manifest, indent=2) + "\n")
    with zipfile.ZipFile(output) as z: assert z.testzip() is None
    return {"archive": output.name, "code_files": len(paths), "sha256": hashlib.sha256(output.read_bytes()).hexdigest()}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    print(json.dumps(build(Path(__file__).resolve().parents[2], args.output), indent=2))
