"""Verify Git LF hashes, executed-byte fingerprints, and optional original artifacts.

No imports of archived ML source; no execution of notebooks or GPU assertions.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import zipfile


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def verify_archive(root, originals=None):
    root = Path(root)
    spec = json.loads((root / "provenance_eol.json").read_text(encoding="utf-8"))
    fp = json.loads((root / "results/run_manifest.json").read_text(encoding="utf-8"))["fingerprint"]
    provenance = json.loads((root / "provenance.json").read_text(encoding="utf-8"))
    expected = {"source/predict.py": fp["script_sha256"], "source/common.py": fp["common_sha256"],
                "source/prompt.txt": fp["prompt_sha256"],
                "notebooks/MTS_02_Qwen35_Check.ipynb": provenance["source_artifact_sha256"]["MTS_02_Qwen35_Check.ipynb"]}
    if set(spec["files"]) != set(expected):
        raise ValueError("Provenance file set differs")
    rows = []
    for name, row in spec["files"].items():
        raw = (root / name).read_bytes()
        # Worktrees may use CRLF, but only an explicit EOL transformation is allowed.
        lf = raw.replace(b"\r\n", b"\n")
        if b"\r" in lf:
            raise ValueError("Bare CR is not an allowed EOL transform: " + name)
        if row["executed_sha256"] != expected[name] or sha(lf) != row["git_lf_sha256"]:
            raise ValueError("Content hash mismatch: " + name)
        if row["executed_eol"] == "CRLF":
            reconstructed = lf.replace(b"\n", b"\r\n")
        elif row["executed_eol"] == "LF":
            if raw != lf:
                raise ValueError("Unexpected EOL change of exact-LF artifact: " + name)
            reconstructed = lf
        else:
            raise ValueError("Unsupported executed EOL policy")
        if sha(reconstructed) != expected[name]:
            raise ValueError("Executed-byte hash mismatch: " + name)
        rows.append({"path": name, "worktree_sha256": sha(raw), "git_lf_sha256": sha(lf),
                     "executed_sha256": expected[name], "executed_eol": row["executed_eol"],
                     "raw_worktree_equals_executed": raw == reconstructed})
    original_result = {"original_artifact_bytes_checked": False,
                       "note": "Without --originals this establishes manifest-relative EOL equivalence only."}
    if originals is not None:
        originals = Path(originals)
        for name, expected_sha in provenance["source_artifact_sha256"].items():
            if sha((originals / name).read_bytes()) != expected_sha:
                raise ValueError("Original artifact hash mismatch: " + name)
        with zipfile.ZipFile(originals / "mts_qwen35_probe.zip") as z:
            if z.testzip() is not None:
                raise ValueError("Input ZIP CRC failure")
            for name in ("predict.py", "common.py", "prompt.txt"):
                if sha(z.read("mts_qwen35_probe/" + name)) != expected["source/" + name]:
                    raise ValueError("Original executed-source mismatch: " + name)
        original_result = {"original_artifact_bytes_checked": True,
                           "note": "Original ZIPs/notebook hashed directly; source ZIP bytes match run fingerprints. This does not authenticate a GPU execution."}
    return {"status": "content_consistent", "checks": rows, **original_result,
            "gpu_run_authenticated": False, "metrics_modified": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--originals", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify_archive(args.archive, args.originals), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
