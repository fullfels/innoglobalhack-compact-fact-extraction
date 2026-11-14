from copy import deepcopy
import pytest
from ml.qwen.analyze_results import analyze, fact_slices


def fact(attribute='service_state', value='работает', end=None):
    return {'object': {'id': 'o', 'name': 'связь'}, 'attribute': attribute, 'value': value,
            'valid_from': '2025-11-14T10:00:00', 'valid_to': end,
            'source': {'turn_idx': 0}, 'polarity': 1}


class ExplicitPairs:
    """Test-only supplied pairs; does not simulate official model quality."""
    def __init__(self, pairs): self.pairs = pairs
    def match_facts(self, gold, pred): return self.pairs
    def evaluate(self, gold, pred): return {'official_result_preserved': True}


def test_transition_single_value_and_event_errors_are_separate():
    gf = [fact(value='не работает', end='2025-11-14T12:00:00'), fact(), fact('tariff', 'Базовый'), fact('action', 'перезагрузка')]
    pf = [deepcopy(gf[0]), deepcopy(gf[1]), fact('action', 'другое действие')]
    pf[1]['valid_from'] = '2025-11-15T10:00:00'
    e = ExplicitPairs([(gf[0], pf[0], 1.), (gf[1], pf[1], 1.)])
    report = analyze({'c': {'facts': gf}}, {'c': {'facts': pf}}, {'c': {'status': 'ok', 'facts': pf}}, e)
    assert report['official_metrics'] == {'official_result_preserved': True}
    assert report['by_kind']['state_transitions']['date_accuracy_among_matched'] == .5
    assert report['by_kind']['state_transitions']['matched'] == 2
    assert report['by_kind']['single_value_attributes']['missed'] == 1
    assert report['by_kind']['actions_and_agreements']['unmatched_predictions'] == 1
    assert len(report['matched_error_traces']) == 1


def test_failed_generation_remains_a_failure_with_missing_gold_facts():
    g = {'c': {'facts': [fact()]}}; p = {'c': {'facts': []}}
    records = {'c': {'status': 'failed', 'facts': None, 'error_code': 'QWEN_OUTPUT_LIMIT'}}
    report = analyze(g, p, records, ExplicitPairs([]))
    assert report['structural_failure_codes'] == {'QWEN_OUTPUT_LIMIT': 1}
    assert report['by_attribute']['service_state']['missed'] == 1
    records['c']['facts'] = []
    with pytest.raises(ValueError, match='Failed generation'):
        analyze(g, p, records, ExplicitPairs([]))


def test_subset_and_record_mismatch_are_rejected():
    with pytest.raises(ValueError, match='call IDs'):
        analyze({'c': {'facts': []}}, {}, {}, ExplicitPairs([]))
    with pytest.raises(ValueError, match='raw record'):
        analyze({'c': {'facts': []}}, {'c': {'facts': []}}, {'c': {'status': 'ok', 'facts': [fact()]}}, ExplicitPairs([]))


def test_equal_duplicate_gold_facts_are_tracked_by_identity():
    gf = [fact(), fact()]; pf = [fact()]
    report = analyze({'c': {'facts': gf}}, {'c': {'facts': pf}}, {'c': {'status': 'ok', 'facts': pf}}, ExplicitPairs([(gf[1], pf[0], 1.)]))
    assert len(fact_slices(gf)) == 2
    assert report['by_attribute']['service_state']['matched'] == 1
    assert report['by_attribute']['service_state']['fact_recall'] == .5
