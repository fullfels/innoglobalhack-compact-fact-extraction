"""Read-only audit of the ten supplied files; writes only audit JSON. No training."""
import argparse
import collections
import hashlib
import importlib.util
import json
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('--source', type=Path, required=True)
ap.add_argument('--out', type=Path, required=True)
a = ap.parse_args()
files = ['README.md', 'schema.md', 'catalog.json', 'train.jsonl', 'val.jsonl',
         'baseline_pred.jsonl', 'baseline_metrics.json', 'case_metrics.py',
         'Кейс_CompactFactExtraction.pdf', 'Критерии_оценивания_МТС_MWS_CompactFactExtraction.pdf']
manifest = {n: {'bytes': (a.source/n).stat().st_size,
                 'sha256': hashlib.sha256((a.source/n).read_bytes()).hexdigest()} for n in files}
rows = {n: [json.loads(s) for s in (a.source/n).read_text().splitlines() if s.strip()]
        for n in ['train.jsonl', 'val.jsonl', 'baseline_pred.jsonl']}
stats = {}
for n, rs in rows.items():
    fs = [f for r in rs for f in r['facts']]
    z = {'calls': len(rs), 'unique_call_ids': len({r['call_id'] for r in rs}),
         'facts': len(fs), 'attributes': dict(collections.Counter(f['attribute'] for f in fs)),
         'polarity': dict(collections.Counter(f['polarity'] for f in fs)),
         'zero_fact_calls': sum(not r['facts'] for r in rs)}
    if n != 'baseline_pred.jsonl':
        issues = collections.Counter()
        for r in rs:
            ts = {t['idx']: t for t in r['turns']}
            for f in r['facts']:
                s=f['source']; t=ts.get(s['turn_idx']); sp=s.get('span'); q=s.get('quote')
                if t is None: issues['missing_source_turn'] += 1
                if t and f['speaker'] != t['speaker']: issues['speaker_mismatch'] += 1
                if sp:
                    issues['present_spans'] += 1
                    if not t or not 0 <= sp[0] < sp[1] <= len(t['text']): issues['out_of_bounds'] += 1
                    if t and q and t['text'][sp[0]:sp[1]] != q: issues['span_quote_mismatch'] += 1
                if t and q and q not in t['text']: issues['quote_not_literal'] += 1
        lens=sorted(sum(len(t['text']) for t in r['turns']) for r in rs)
        z.update(source_checks=dict(issues), chars={'min':lens[0], 'median':lens[len(lens)//2],
                   'p95_nearest_rank_lower':lens[int(.95*(len(lens)-1))], 'max':lens[-1]},
                 noise=dict(collections.Counter(r['noise_level'] for r in rs)),
                 min_turns=min(len(r['turns']) for r in rs), max_turns=max(len(r['turns']) for r in rs))
    stats[n]=z
funcs={'call_id':lambda r:r['call_id'],
       'account':lambda r:(r.get('series') or {}).get('account_id') or r['call_id'].split(':')[0],
       'exact_turn_texts':lambda r:tuple(t['text'] for t in r['turns'])}
overlap={k:len({f(r) for r in rows['train.jsonl']} & {f(r) for r in rows['val.jsonl']}) for k,f in funcs.items()}
spec=importlib.util.spec_from_file_location('official_metrics', a.source/'case_metrics.py')
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
actual=m.evaluate({r['call_id']:r for r in rows['val.jsonl']},
                  {r['call_id']:r for r in rows['baseline_pred.jsonl']})
published=json.loads((a.source/'baseline_metrics.json').read_text())
report={'source_manifest':manifest, 'stats':stats, 'train_val_overlap':overlap,
        'baseline_id_set_matches_val':{r['call_id'] for r in rows['val.jsonl']} == {r['call_id'] for r in rows['baseline_pred.jsonl']},
        'baseline_published':published, 'baseline_recomputed':actual,
        'baseline_discrepancies':{k:{'published':published.get(k), 'recomputed':v} for k,v in actual.items() if published.get(k)!=v}}
a.out.parent.mkdir(parents=True, exist_ok=True)
a.out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'report':str(a.out),'recomputed':actual},ensure_ascii=False))
