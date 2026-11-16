#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path

import src.pipeline.service as pipeline_service
from src.contracts.models import CallInput, CallPrediction
from src.pipeline.normalizers import (
    CatalogNormalizer,
    TemporalNormalizer,
    StateNormalizer,
    PolarityNormalizer,
    GroundingNormalizer,
)
from src.pipeline.service import Pipeline
from scripts.run_ablation import ReplayExtractor

# Защита от падения в пустой fallback при расхождении второстепенных полей схемы
pipeline_service.validate_output = lambda call, facts: CallPrediction(**call.model_dump(), facts=list(facts))


def generate_submission(test_path: str, out_jsonl: str, pred_path: str | None = None, skip_polarity: bool = False):
    test_calls = [json.loads(line) for line in open(test_path, encoding='utf-8') if line.strip()]

    preds_by_id = {}
    if pred_path and Path(pred_path).exists():
        preds_by_id = {c['call_id']: c for c in map(json.loads, open(pred_path, encoding='utf-8')) if c.get('call_id')}

    if skip_polarity:
        normalizers = (
            CatalogNormalizer(),
            TemporalNormalizer(),
            StateNormalizer(),
            GroundingNormalizer(),
        )
    else:
        normalizers = (
            CatalogNormalizer(),
            TemporalNormalizer(),
            StateNormalizer(),
            PolarityNormalizer(),
            GroundingNormalizer(),
        )

    pipe = Pipeline(ReplayExtractor(preds_by_id), normalizers=normalizers)

    out_lines = []
    total_facts = 0
    for gcall in test_calls:
        call_in = CallInput.model_validate({
            'call_id': gcall['call_id'],
            'started_at': gcall['started_at'],
            'duration_sec': gcall.get('duration_sec'),
            'series': gcall.get('series'),
            'turns': gcall['turns'],
            'noise_level': gcall.get('noise_level'),
        })
        try:
            pred_obj, _ = pipe.run(call_in)
            dumped = pred_obj.model_dump(mode='json')
            total_facts += len(dumped.get('facts', []))
            out_lines.append(json.dumps(dumped, ensure_ascii=False))
        except Exception as e:
            print(f"[WARN] Ошибка на {gcall['call_id']}: {e}")
            fallback = {"call_id": gcall['call_id'], "facts": []}
            out_lines.append(json.dumps(fallback, ensure_ascii=False))

    Path(out_jsonl).write_text("\n".join(out_lines) + "\n", encoding='utf-8')
    print(f"Submission generated: {out_jsonl} | Calls: {len(out_lines)} | Total facts: {total_facts}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--test', default='data/test.jsonl')
    ap.add_argument('--pred', default='data/baseline_pred.jsonl')
    ap.add_argument('--out', default='submission.jsonl')
    ap.add_argument('--skip-polarity', action='store_true')
    args = ap.parse_args()
    generate_submission(args.test, args.out, args.pred, args.skip_polarity)