#!/usr/bin/env python3
"""T21: Пошаговая абляция нормализаторов DE на inner-dev предсказаниях."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from case_metrics import evaluate
from src.contracts.models import CallInput, CandidateBatch, CandidateFact, Fact
from src.pipeline.adapters import ModelDescriptor
from src.pipeline.normalizers import (
    CatalogNormalizer,
    GroundingNormalizer,
    PolarityNormalizer,
    StateNormalizer,
    TemporalNormalizer,
)
from src.pipeline.service import Pipeline


class ReplayExtractor:
    descriptor = ModelDescriptor("ml-predictions-replay", "real", "offline")
    state = "ready"

    def __init__(self, preds_by_id: dict[str, dict[str, Any]]):
        self.preds_by_id = preds_by_id

    def extract(self, call: CallInput) -> CandidateBatch:
        raw_call = self.preds_by_id.get(call.call_id, {})
        cands: list[CandidateFact] = []
        for rf in raw_call.get("facts", []):
            fact = Fact.model_validate(rf)
            conf = "confirmed" if fact.attribute == "agreement" else "unknown"
            ev = tuple(t.idx for t in call.turns if t.speaker == "client")
            cands.append(CandidateFact(fact=fact, confirmation=conf, evidence_turns=ev))
        return CandidateBatch(candidates=tuple(cands))


def run_ablation(gold_path: str, pred_path: str, out_json: str) -> None:
    gold_calls = {c["call_id"]: c for c in map(json.loads, open(gold_path, encoding="utf-8")) if c.get("call_id")}
    pred_calls = {c["call_id"]: c for c in map(json.loads, open(pred_path, encoding="utf-8")) if c.get("call_id")}

    stages: list[tuple[str, tuple[Any, ...]]] = [
        ("0_raw_baseline", ()),
        ("1_plus_catalog", (CatalogNormalizer(),)),
        ("2_plus_temporal", (CatalogNormalizer(), TemporalNormalizer())),
        ("3_plus_state", (CatalogNormalizer(), TemporalNormalizer(), StateNormalizer())),
        ("4_plus_polarity", (CatalogNormalizer(), TemporalNormalizer(), StateNormalizer(), PolarityNormalizer())),
        (
            "5_full_pipeline",
            (CatalogNormalizer(), TemporalNormalizer(), StateNormalizer(), PolarityNormalizer(), GroundingNormalizer()),
        ),
    ]

    report: dict[str, Any] = {}
    extractor = ReplayExtractor(pred_calls)

    for stage_name, norms in stages:
        pipe = Pipeline(extractor, normalizers=norms)
        stage_preds: dict[str, dict[str, Any]] = {}
        for cid, gcall in gold_calls.items():
            call_in = CallInput.model_validate(
                {
                    "call_id": gcall["call_id"],
                    "started_at": gcall["started_at"],
                    "duration_sec": gcall.get("duration_sec"),
                    "series": gcall.get("series"),
                    "turns": gcall["turns"],
                    "noise_level": gcall.get("noise_level"),
                }
            )
            pred_obj, _warnings = pipe.run(call_in)
            stage_preds[cid] = pred_obj.model_dump(mode="json")

        report[stage_name] = evaluate(gold_calls, stage_preds)
        print(f"[{stage_name}] {report[stage_name]}")

    Path(out_json).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("gold")
    ap.add_argument("pred")
    ap.add_argument("--out", default="ablation_report.json")
    args = ap.parse_args()
    run_ablation(args.gold, args.pred, args.out)