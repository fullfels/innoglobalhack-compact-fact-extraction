"""Recompute the archived smoke-test metrics from separately obtained official data."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

def read_jsonl(path):
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    result = {row["call_id"]: row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"Duplicate call_id in {path.name}")
    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, type=Path, help="Official Case data directory, obtained separately")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    hashes = json.loads((root / "source/official_hashes.json").read_text(encoding="utf-8"))
    for name in ("val.jsonl", "baseline_pred.jsonl", "case_metrics.py"):
        actual = hashlib.sha256((args.data / name).read_bytes()).hexdigest()
        if actual != hashes[name]:
            raise ValueError(f"Official file hash differs: {name}; verify scorer/data version before comparing")
    ids = json.loads((root / "selected_call_ids.json").read_text(encoding="utf-8"))
    if len(ids) != 10 or len(set(ids)) != 10:
        raise ValueError("Expected ten unique selected IDs")
    gold_all = read_jsonl(args.data / "val.jsonl")
    baseline_all = read_jsonl(args.data / "baseline_pred.jsonl")
    gold = {cid: gold_all[cid] for cid in ids}
    baseline = {cid: baseline_all[cid] for cid in ids}
    pred = read_jsonl(root / "results/pred_before.jsonl")
    if set(pred) != set(ids):
        raise ValueError("Prediction IDs differ from the selected input IDs")
    spec = importlib.util.spec_from_file_location("official_case_metrics", args.data / "case_metrics.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    own = module.evaluate(gold, pred)
    control = module.evaluate(gold, baseline)
    for name, actual in (("metrics_before.json", own), ("metrics_mts_selected.json", control)):
        expected = json.loads((root / "results" / name).read_text(encoding="utf-8"))
        if actual != expected:
            raise ValueError(f"Recomputed values differ from {name}")
    print(json.dumps({"verified": True, "n_calls": 10, "qwen": own, "mts_same_calls": control}, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
