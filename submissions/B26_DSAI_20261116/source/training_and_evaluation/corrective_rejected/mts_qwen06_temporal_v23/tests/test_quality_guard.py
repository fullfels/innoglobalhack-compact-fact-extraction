import pytest
from quality_guard import allowed,metric_key

def metric(**kw):
    return dict({'fact_precision':.6,'fact_recall':.5,'temporal_recall':.4,'state_accuracy':.5,'grounding_turn':.8,'n_grounded':10},**kw)
def pairs(temporal=(('x',0),('x',1)),correct=(('x',0),)):
    return {'temporal':temporal,'correct_temporal':correct,'matched':temporal}

def test_better_temporal_cannot_hide_lost_facts():
    ok,reasons=allowed(metric(temporal_recall=.9,fact_recall=.4),metric(),pairs(),pairs())
    assert not ok and 'metric_degraded:fact_recall' in reasons

def test_better_ratio_cannot_drop_hard_temporal_denominator():
    ok,reasons=allowed(metric(temporal_recall=1.),metric(),pairs(temporal=(('x',0),)),pairs())
    assert not ok and 'lost_previously_matched_temporal_facts' in reasons

def test_new_dates_cannot_break_previously_correct_intervals():
    ok,reasons=allowed(metric(temporal_recall=.6),metric(),pairs(correct=(('x',1),)),pairs())
    assert not ok and 'lost_previously_correct_temporal_intervals' in reasons

@pytest.mark.parametrize('field',['fact_recall','state_accuracy','temporal_recall'])
def test_no_degradation_in_core_metrics(field):
    m=metric();new={**m,field:m[field]-.01}
    assert allowed(new,m,pairs(),pairs())[0] is False

def test_unchanged_reference_allowed_but_does_not_win_tie():
    assert allowed(metric(),metric(),pairs(),pairs())[0]
    assert not(metric_key(metric())>metric_key(metric()))

def test_real_improvement_without_sacrifices_is_eligible():
    assert allowed(metric(temporal_recall=.6),metric(),pairs(correct=(('x',0),('x',1))),pairs())[0]

def test_no_temporal_denominator_is_not_perfect_accuracy():
    assert metric_key(metric(temporal_recall=None))<metric_key(metric())

def test_last_non_milestone_checkpoint_can_be_selected_with_the_same_guards(tmp_path,monkeypatch):
    import train_temporal
    base=metric();new=metric(temporal_recall=.6)
    report=lambda m:{'complete':True,'metric_errors':{},'metrics':{'partial_serving':m,'calendar_serving':m},
        'metrics_by_slice':{'official_real':{'partial_serving':m,'calendar_serving':m}},'statuses':{'partial':{'ok':32},'calendar':{'ok':32}}}
    initial={'step':0,'calendar':False,'metrics':base,'failed':0,'paired':pairs()}
    monkeypatch.setattr(train_temporal,'paired_dates',lambda *a,**kw:pairs(correct=(('x',0),('x',1))))
    chosen,options=train_temporal.assess_quality(report(new),tmp_path,tmp_path,[],report(base),initial,initial,137)
    assert chosen['step']==137 and chosen['allowed'] and len(options)==2
