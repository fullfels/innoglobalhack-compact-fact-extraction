"""Conservative inner-dev model selection; never select by temporal ratio alone."""
import json
from datetime import datetime
from ml.qwen.replay_v22 import load_evaluator

def paired_dates(root,gold_rows,prediction_path):
    ev=load_evaluator(root/'evaluation/case_metrics.py');pred={r['call_id']:r for r in map(json.loads,prediction_path.read_text(encoding='utf-8').splitlines())}
    temporal=set();correct=set();matched=set()
    for call in gold_rows:
        cid=call['call_id'];facts=call['facts'];index={id(f):i for i,f in enumerate(facts)}
        days=set()
        for t in call['turns']:days|=ev.recover_dates(t['text'],datetime.fromisoformat(call['started_at']).date())
        recovered={d.isoformat() for d in days}
        for g,p,_ in ev.match_facts(facts,pred.get(cid,{}).get('facts')or[]):
            key=(cid,index[id(g)]);matched.add(key)
            if g['attribute'] not in ev.TEMPORAL_ATTRS:continue
            gold_days={ev._day(g.get('valid_from')),ev._day(g.get('valid_to'))}-{None}
            if not(gold_days&recovered):continue
            temporal.add(key)
            if all(ev._day(g.get(f))==ev._day(p.get(f)) for f in ['valid_from','valid_to']):correct.add(key)
    return {'matched':sorted(matched),'temporal':sorted(temporal),'correct_temporal':sorted(correct)}

def metric_key(metrics,failed=0):
    def value(k):return metrics[k] if metrics.get(k)is not None else -1.
    p,r=value('fact_precision'),value('fact_recall');f1=2*p*r/(p+r) if p+r>0 else 0.
    return (value('temporal_recall'),value('state_accuracy'),f1,r,p,-failed)

def allowed(candidate,baseline,paired_candidate,paired_baseline):
    reasons=[]
    # Recall and state may not fall on the fixed inner dev. Small precision/grounding tolerance.
    for key,tolerance in [('fact_recall',0.),('state_accuracy',0.),('temporal_recall',0.),('fact_precision',.02),('grounding_turn',.02)]:
        old,new=baseline.get(key),candidate.get(key)
        if old is not None and (new is None or new+1e-8<old-tolerance):reasons.append('metric_degraded:'+key)
    if candidate['n_grounded']<baseline['n_grounded']:reasons.append('fewer_matched_facts')
    old_temporal={tuple(x) for x in paired_baseline['temporal']};new_temporal={tuple(x) for x in paired_candidate['temporal']}
    old_correct={tuple(x) for x in paired_baseline['correct_temporal']};new_correct={tuple(x) for x in paired_candidate['correct_temporal']}
    if not old_temporal<=new_temporal:reasons.append('lost_previously_matched_temporal_facts')
    if not old_correct<=new_correct:reasons.append('lost_previously_correct_temporal_intervals')
    return not reasons,reasons

def primary_metrics(report,view):
    return report.get('metrics_by_slice',{}).get('official_real',{}).get(view) or report['metrics'][view]

def stress_guard(report,baseline,view,baseline_view):
    new=report.get('metrics_by_slice',{}).get('synthetic_stress',{}).get(view)
    old=baseline.get('metrics_by_slice',{}).get('synthetic_stress',{}).get(baseline_view)
    if not old:return True,[]
    reasons=[]
    for k in ['fact_recall','temporal_recall','state_accuracy']:
        if old.get(k)is not None and (not new or new.get(k)is None or new[k]<old[k]-.02):reasons.append('synthetic_stress_degraded:'+k)
    return not reasons,reasons
