import json
import pytest
from ml.qwen.final_bridge import process_output
from ml.qwen.calendar_v23 import calendar_anchors
from src.contracts.models import CallInput
from src.pipeline.adapters import PipelineError

def fact(**kw):
    return {**{'object':{'name':'Анна','type':'client'},'attribute':'agreement','value':'перезвонок специалиста','polarity':1,'speaker':'operator',
        'valid_from':'2026-07-32T00:00:00','valid_to':None,'source':{'turn_idx':50,'span':None,'quote':None}},**kw}
def dialog(reply='Да, подходит.',proposal='Через шесть дней специалист перезвонит.',summary='Зафиксировала: специалист перезвонит.'):
    return CallInput.model_validate({'call_id':'unit','started_at':'2026-07-28T12:00:00','turns':[{'idx':4,'speaker':'operator','text':proposal},
        {'idx':9,'speaker':'client','text':reply},{'idx':50,'speaker':'operator','text':summary}]})
def parse(fs,c):return process_output(json.dumps({'facts':fs},ensure_ascii=False),c,calendar=True)

def test_later_summary_can_link_to_unique_prior_proposal_and_acceptance_without_moving_source():
    r=parse([fact()],dialog());c=r.batch.candidates[0]
    assert c.fact.valid_from.isoformat()=='2026-08-03T00:00:00' and c.fact.source.turn_idx==50 and c.fact.speaker=='operator'
    assert c.confirmation=='confirmed' and c.evidence_turns==(4,9,50)

@pytest.mark.parametrize('reply',['Спасибо.','Нет, не нужно.','Да, если получится.','Да, но только позже.'])
def test_ambiguous_or_rejected_client_reply_cannot_repair_invalid_date(reply):
    with pytest.raises(PipelineError):parse([fact()],dialog(reply))

def test_explicit_clock_is_computed_with_date_and_vague_bound_is_not_invented_as_hour():
    r=parse([fact()],dialog(proposal='Через шесть дней специалист перезвонит в 14:30.'))
    assert r.batch.candidates[0].fact.valid_from.isoformat()=='2026-08-03T14:30:00'
    r=parse([fact()],dialog(proposal='Через шесть дней специалист перезвонит до обеда.'))
    assert r.batch.candidates[0].fact.valid_from.isoformat()=='2026-08-03T00:00:00'
    assert r.changes[0]['proof']['precision']=='day' and r.changes[0]['proof']['clock_note']
    assert r.batch.candidates[0].warnings and r.batch.candidates[0].temporal_expression

def test_ambiguous_meridiem_is_not_an_invented_exact_hour():
    r=parse([fact()],dialog(proposal='Через шесть дней специалист перезвонит в 4:00 вечера.'))
    assert r.batch.candidates[0].fact.valid_from.isoformat()=='2026-08-03T00:00:00' and r.changes[0]['proof']['precision']=='day'

def test_proposal_about_different_event_is_not_borrowed_for_a_summary():
    with pytest.raises(PipelineError):parse([fact()],dialog(proposal='Через шесть дней мастер приедет.',summary='Специалист перезвонит.'))

def test_partial_retains_good_fact_and_all_bad_is_not_successful_empty():
    good=fact(valid_from='2026-07-29T00:00:00');bad=fact(speaker='client')
    r=parse([good,bad],dialog());assert r.status=='partial' and len(r.batch.candidates)==1 and r.rejected
    with pytest.raises(PipelineError) as e:parse([bad],dialog())
    assert e.value.code=='QWEN_ALL_FACTS_REJECTED'

def test_calendar_handles_leap_and_explicit_dates_and_skips_numbered_identifier():
    assert calendar_anchors('через два дня','2024-02-28T12:00:00')[0]['datetime']=='2024-03-01T00:00:00'
    assert calendar_anchors('С 31.12.2025','2026-01-01T12:00:00')[0]['datetime']=='2025-12-31T00:00:00'
    assert calendar_anchors('заявка 20.04.2026','2026-01-01T12:00:00')==[]

def test_asr_time_alias_preserves_original_offsets_and_negation():
    text='Вчира интернет перестал работать.';a=calendar_anchors(text,'2026-01-01T12:00:00')[0]
    assert a['datetime']=='2025-12-31T00:00:00' and text[slice(*a['span'])]==a['expression']=='Вчира'
    r=parse([fact()],dialog(proposal='Щерез шесть дней специалист перезвонит.'))
    assert r.batch.candidates[0].fact.valid_from.isoformat()=='2026-08-03T00:00:00'
    with pytest.raises(PipelineError):parse([fact()],dialog(proposal='Не зафтра специалист перезвонит.'))
