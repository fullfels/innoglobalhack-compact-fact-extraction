from pathlib import Path
from collections import Counter
import json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from verify_data import verify,calendar_from_expression

def test_all1024_sources_dates_splits_and_augmentation_invariants():
    result=verify(ROOT);assert result['passed'] and result['training_rows']==1024
    assert result['real_rows']==768 and result['synthetic_families_checked']==128

def test_every_optimizer_window_has_fixed3_real_1_synthetic_mix():
    rows=[json.loads(s) for s in (ROOT/'sft/train.jsonl').read_text(encoding='utf-8').splitlines()]
    for i in range(0,1024,4):
        origins=[r['origin'] for r in rows[i:i+4]]
        assert sum(o['origin']=='synthetic' for o in origins)==1
        assert sum(o.get('selection')=='source_priority' for o in origins)==2
        assert sum(o.get('selection')=='general_replay' for o in origins)==1

def test_calendar_oracle_handles_leap_days_weeks_and_year_boundary():
    assert calendar_from_expression('через два дня','2024-02-28T10:00:00')=='2024-03-01'
    assert calendar_from_expression('через две недели','2026-12-25T10:00:00')=='2027-01-08'
    assert calendar_from_expression('вчера','2026-01-01T10:00:00')=='2025-12-31'

def test_synthetic_families_never_cross_train_selection_or_guard():
    groups=[]
    for p in ['train1024.jsonl','selection32.jsonl','audit24.jsonl']:
        r=[json.loads(s) for s in (ROOT/'data'/p).read_text(encoding='utf-8').splitlines()]
        groups.append({(c.get('series')or{}).get('account_id',c['call_id'].split(':')[0]) for c in r})
    assert all(not(groups[i]&groups[j]) for i in range(3) for j in range(i))

def test_questions_and_unconfirmed_offers_have_empty_gold_in_empty_scenario():
    rows=[json.loads(s) for s in (ROOT/'sft/train.jsonl').read_text(encoding='utf-8').splitlines()]
    empty=[r for r in rows if r['origin'].get('scenario')==7]
    assert len(empty)==32 and all(json.loads(r['completion'][0]['content'])=={'facts':[]} for r in empty)
