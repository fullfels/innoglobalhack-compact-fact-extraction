#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Any
from src.contracts.models import CallInput, CandidateBatch, CandidateFact, Fact, Source
from src.pipeline.adapters import ModelDescriptor
from src.pipeline.normalizers import (
    CatalogNormalizer, 
    TemporalNormalizer, 
    StateNormalizer, 
    PolarityNormalizer, 
    GroundingNormalizer
)
from src.pipeline.service import Pipeline

def generate_submission(test_path: str, out_jsonl: str, pred_path: str | None = None):
    test_calls = [json.loads(line) for line in open(test_path, encoding='utf-8') if line.strip()]
    
    preds_by_id = {}
    if pred_path and Path(pred_path).exists():
        preds_by_id = {c['call_id']: c for c in map(json.loads, open(pred_path, encoding='utf-8')) if c.get('call_id')}

    normalizers = (
        CatalogNormalizer(),
        TemporalNormalizer(),
        StateNormalizer(),
        PolarityNormalizer(),
        GroundingNormalizer(),
    )
    
    class TestExtractor:
        descriptor = ModelDescriptor('pipeline-extractor', 'real', 'offline')
        state = 'ready'
        def extract(self, call: CallInput) -> CandidateBatch:
            raw_call = preds_by_id.get(call.call_id, {})
            cands = []
            max_turn = len(call.turns) - 1 if call.turns else 0
            for rf in raw_call.get('facts', []):
                try:
                    fact = Fact.model_validate(rf)
                except Exception:
                    continue
                t_idx = fact.source.turn_idx if fact.source and fact.source.turn_idx is not None else 0
                if t_idx < 0 or t_idx > max_turn:
                    t_idx = 0
                conf = 'confirmed' if fact.attribute == 'agreement' else 'unknown'
                ev = tuple(t.idx for t in call.turns if t.speaker == 'client')
                cands.append(CandidateFact(fact=fact, confirmation=conf, evidence_turns=ev))
            return CandidateBatch(candidates=tuple(cands))

    pipe = Pipeline(TestExtractor(), normalizers=normalizers)
    
    out_lines = []
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
            out_lines.append(json.dumps(pred_obj.model_dump(mode='json'), ensure_ascii=False))
        except Exception:
            fallback = {"call_id": gcall['call_id'], "facts": []}
            out_lines.append(json.dumps(fallback, ensure_ascii=False))

    Path(out_jsonl).write_text("\n".join(out_lines), encoding='utf-8')
    print(f"Submission generated successfully: {out_jsonl}")

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--test', default='data/test.jsonl')
    ap.add_argument('--pred', default='data/baseline_pred.jsonl')
    ap.add_argument('--out', default='submission.jsonl')
    args = ap.parse_args()
    generate_submission(args.test, args.out, args.pred)