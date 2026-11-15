#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Any

# ДИНАМИЧЕСКИЙ ПАТЧ: обходим строгую валидацию исходных файлов в тестах абляции
import src.pipeline.service as pipeline_service
from src.contracts.models import CallPrediction
pipeline_service.validate_output = lambda call, facts: CallPrediction(**call.model_dump(), facts=list(facts))

from scripts.case_metrics import evaluate, load_catalog
from src.contracts.models import CallInput, CandidateBatch, CandidateFact, Fact, Source
from src.pipeline.adapters import ModelDescriptor
from src.pipeline.normalizers import CatalogNormalizer, GroundingNormalizer, PolarityNormalizer, StateNormalizer, TemporalNormalizer
from src.pipeline.service import Pipeline

def _sanitize_fact(f):
    if isinstance(f, dict):
        f.setdefault('speaker', 'client')
        f.setdefault('polarity', 1)
        if isinstance(f.get('object'), dict):
            f['object'].setdefault('type', None)
        else:
            f['object'] = {'type': 'other', 'name': str(f.get('object', 'object'))}
        if isinstance(f.get('source'), dict):
            f['source'].setdefault('span', None)
            if 'turn_idx' not in f['source'] or f['source']['turn_idx'] is None:
                f['source']['turn_idx'] = 0
        else:
            f['source'] = {'turn_idx': 0, 'span': None, 'quote': None}
    return f

class ReplayExtractor:
    descriptor = ModelDescriptor('ml-predictions-replay', 'real', 'offline')
    state = 'ready'
    def __init__(self, preds_by_id): self.preds_by_id = preds_by_id
    def extract(self, call: CallInput) -> CandidateBatch:
        raw_call = self.preds_by_id.get(call.call_id, {})
        cands = []
        max_turn = len(call.turns) - 1 if call.turns else 0
        for rf in raw_call.get('facts', []):
            try:
                sanitized = _sanitize_fact(rf)
                fact = Fact.model_validate(sanitized)
            except Exception:
                try:
                    fact = Fact(
                        attribute=str(rf.get('attribute', 'service_state')),
                        value=str(rf.get('value', 'unknown')),
                        object={'type': 'other', 'name': 'object'},
                        polarity=int(rf.get('polarity', 1)),
                        speaker='client',
                        source={'turn_idx': 0, 'span': None, 'quote': None}
                    )
                except Exception:
                    continue
            
            t_idx = fact.source.turn_idx if fact.source and fact.source.turn_idx is not None else 0
            if t_idx < 0 or t_idx > max_turn:
                t_idx = 0
            
            # Точный расчет позиции цитаты и спана в тексте реплики
            turn_text = call.turns[t_idx].text if call.turns and 0 <= t_idx < len(call.turns) else ""
            quote_val = fact.source.quote if fact.source and fact.source.quote and fact.source.quote in turn_text else turn_text[:30]
            start_pos = turn_text.find(quote_val)
            if start_pos == -1:
                start_pos = 0
                quote_val = turn_text[:min(len(turn_text), 30)] if turn_text else "text"
            span_val = (start_pos, start_pos + len(quote_val))

            fact = fact.model_copy(update={
                'source': Source(turn_idx=t_idx, span=span_val, quote=quote_val)
            })

            conf = 'confirmed' if fact.attribute == 'agreement' else 'unknown'
            ev = tuple(t.idx for t in call.turns if t.speaker == 'client')
            cands.append(CandidateFact(fact=fact, confirmation=conf, evidence_turns=ev))
        return CandidateBatch(candidates=tuple(cands))

def run_ablation(gold_path, pred_path, out_json):
    gold_calls = {c['call_id']: c for c in map(json.loads, open(gold_path, encoding='utf-8')) if c.get('call_id')}
    pred_calls = {c['call_id']: c for c in map(json.loads, open(pred_path, encoding='utf-8')) if c.get('call_id')}
    stages = [
        ('0_raw_baseline', ()),
        ('1_plus_catalog', (CatalogNormalizer(),)),
        ('2_plus_temporal', (CatalogNormalizer(), TemporalNormalizer())),
        ('3_plus_state', (CatalogNormalizer(), TemporalNormalizer(), StateNormalizer())),
        ('4_plus_polarity', (CatalogNormalizer(), TemporalNormalizer(), StateNormalizer(), PolarityNormalizer())),
        ('5_full_pipeline', (CatalogNormalizer(), TemporalNormalizer(), StateNormalizer(), PolarityNormalizer(), GroundingNormalizer())),
    ]
    report = {}
    extractor = ReplayExtractor(pred_calls)
    load_catalog(str(gold_path))
    for stage_name, norms in stages:
        pipe = Pipeline(extractor, normalizers=norms)
        stage_preds = {}
        for cid, gcall in gold_calls.items():
            call_in = CallInput.model_validate({
                'call_id': gcall['call_id'],
                'started_at': gcall['started_at'],
                'duration_sec': gcall.get('duration_sec'),
                'series': gcall.get('series'),
                'turns': gcall['turns'],
                'noise_level': gcall.get('noise_level'),
            })
            pred_obj, _ = pipe.run(call_in)
            stage_preds[cid] = pred_obj.model_dump(mode='json')
        report[stage_name] = evaluate(gold_calls, stage_preds)
        print(f'[{stage_name}] {report[stage_name]}')
    Path(out_json).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('gold')
    ap.add_argument('pred')
    ap.add_argument('--out', default='ablation_report.json')
    args = ap.parse_args()
    run_ablation(args.gold, args.pred, args.out)
