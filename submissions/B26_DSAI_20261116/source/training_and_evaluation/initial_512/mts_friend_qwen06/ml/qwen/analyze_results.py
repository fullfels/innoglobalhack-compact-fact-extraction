"""Diagnostic slices using the unchanged organizer matcher, not a replacement metric."""
from collections import Counter, defaultdict
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

STATE_ATTRIBUTES = {'service_state', 'tariff', 'ticket_status'}


def fact_slices(facts):
    groups = defaultdict(list)
    for fact in facts:
        obj = fact.get('object') or {}
        key = (obj.get('id') or obj.get('name'), fact.get('attribute'))
        groups[key].append(fact)
    result = {}
    for group in groups.values():
        values = {str(f.get('value')) for f in group}
        transition = len(values) > 1 or any(f.get('valid_to') is not None for f in group)
        for fact in group:
            category = ('state_transitions' if transition else 'single_value_attributes') if fact.get('attribute') in STATE_ATTRIBUTES else 'actions_and_agreements'
            result[id(fact)] = category
    return result


def analyze(gold, predictions, records, evaluator):
    if set(gold) != set(predictions):
        raise ValueError('Gold and prediction call IDs must agree; choose an explicit diagnostic subset')
    if set(records) != set(predictions):
        raise ValueError('Every call needs its real generation status')
    by_attribute, by_kind = defaultdict(Counter), defaultdict(Counter)
    errors = Counter()
    traces = []
    for cid, call in gold.items():
        gf, pf = call['facts'], predictions[cid]['facts']
        record = records[cid]
        if record['status'] not in ('ok', 'failed'):
            raise ValueError('Unsupported generation status')
        if record['status'] == 'ok' and record.get('facts') != pf:
            raise ValueError('Prediction facts disagree with the actual successful raw record')
        if record['status'] != 'ok':
            if record.get('facts') is not None or pf:
                raise ValueError('Failed generation cannot be converted to successful facts')
            errors[record.get('error_code') or 'UNKNOWN_FAILURE'] += 1
        gs, ps = fact_slices(gf), fact_slices(pf)
        pairs = evaluator.match_facts(gf, pf)
        matched_g, matched_p = {id(g) for g, _, _ in pairs}, {id(p) for _, p, _ in pairs}
        pair_by_g = {id(g): p for g, p, _ in pairs}
        pair_by_p = {id(p): g for g, p, _ in pairs}
        for g in gf:
            buckets = (by_attribute[g['attribute']], by_kind[gs[id(g)]])
            for bucket in buckets:
                bucket['gold'] += 1
                bucket['matched'] += id(g) in matched_g
            if id(g) in matched_g:
                p = pair_by_g[id(g)]
                date_ok = all((g.get(k) or '')[:10] == (p.get(k) or '')[:10] for k in ('valid_from', 'valid_to'))
                source_ok = (g.get('source') or {}).get('turn_idx') == (p.get('source') or {}).get('turn_idx')
                for bucket in buckets:
                    bucket['matched_date_correct'] += date_ok
                    bucket['matched_source_correct'] += source_ok
                if not date_ok or not source_ok:
                    traces.append({'call_id': cid, 'attribute': g['attribute'], 'kind': gs[id(g)],
                                   'date_correct': date_ok, 'source_correct': source_ok})
        for p in pf:
            # A matched prediction inherits the matched gold slice; unmatched
            # predictions use their own temporal grouping. This policy is explicit.
            kind = gs[id(pair_by_p[id(p)])] if id(p) in matched_p else ps[id(p)]
            by_attribute[p['attribute']]['predicted'] += 1
            by_kind[kind]['predicted'] += 1
    def finish(values):
        result = {}
        for key, count in sorted(values.items()):
            g, p, m = count['gold'], count['predicted'], count['matched']
            result[key] = {**dict(count), 'missed': g-m, 'unmatched_predictions': p-m,
                           'fact_recall': m/g if g else None, 'fact_precision': m/p if p else None,
                           'date_accuracy_among_matched': count['matched_date_correct']/m if m else None,
                           'source_accuracy_among_matched': count['matched_source_correct']/m if m else None}
        return result
    return {'official_metrics': evaluator.evaluate(gold, predictions),
            'generation_status_counts': dict(Counter(r['status'] for r in records.values())),
            'structural_failure_codes': dict(errors), 'by_attribute': finish(by_attribute),
            'by_kind': finish(by_kind), 'matched_error_traces': traces,
            'slice_policy': 'State/tariff/ticket_status: multiple values or a closed interval = transition. Otherwise single-value observation within this call, not an intrinsically immutable attribute. Actions/agreements are separate events. Matched predictions inherit gold slice; unmatched use prediction grouping.',
            'scoring_limit': 'Inherits the unchanged official matcher and its known limitations; does not establish semantic truth independently.'}


def read_jsonl(path):
    rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    indexed = {r['call_id']: r for r in rows}
    if len(indexed) != len(rows): raise ValueError('Duplicate call ID')
    return indexed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gold', type=Path, required=True)
    ap.add_argument('--prediction-dir', type=Path, required=True)
    ap.add_argument('--evaluator', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--select-prediction-ids', action='store_true', help='Explicitly analyze a diagnostic subset; report the full gold count')
    args = ap.parse_args()
    gold = read_jsonl(args.gold)
    pred = read_jsonl(args.prediction_dir/'predictions.jsonl')
    original_count = len(gold)
    if args.select_prediction_ids:
        if not set(pred) <= set(gold): raise ValueError('Unknown prediction IDs')
        gold = {cid: gold[cid] for cid in pred}
    rows = [json.loads(p.read_text(encoding='utf-8')) for p in sorted((args.prediction_dir/'records').glob('*.json'))]
    records = {r['call_id']: r for r in rows}
    if len(records) != len(rows): raise ValueError('Duplicate raw record ID')
    spec = importlib.util.spec_from_file_location('unchanged_organizer_metrics', args.evaluator)
    evaluator = importlib.util.module_from_spec(spec); spec.loader.exec_module(evaluator)
    report = analyze(gold, pred, records, evaluator)
    report.update(original_gold_calls=original_count, explicit_subset=args.select_prediction_ids,
                  evaluator_sha256=hashlib.sha256(args.evaluator.read_bytes()).hexdigest())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps({'analyzed_calls': len(gold), 'output': str(args.output)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
