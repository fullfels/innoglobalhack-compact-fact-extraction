import pytest
from src.contracts.models import CandidateBatch, CandidateFact, Fact
from ml.qwen.postprocessing import TariffValueNormalizer

CATALOG={'tariffs':['Хайп','Хайп+','Сотка','Оптимальный','Новый оптимальный']}

@pytest.mark.parametrize('value,expected',[
    ('тариф Хайп+','Хайп+'),('клиент подтвердила тариф «Сотка»','Сотка'),
    ('тариф «новый оптимальный»','Новый оптимальный'),
    ('не Хайп+',None),('Хайп+ или Сотка',None),('Хайп+ → Сотка',None),
    ('неизвестный тариф',None),('Сотка и Хайп+',None)])
def test_safe_canonicalization(value,expected):
    assert TariffValueNormalizer(CATALOG).canonical(value)==expected

def test_only_tariff_value_changes_without_mutating_input():
    fact=Fact.model_validate({'object':{'name':'клиент','type':'client'},'attribute':'tariff',
        'value':'тариф Хайп+','polarity':-1,'speaker':'client','valid_from':'2026-11-15T12:00:00',
        'valid_to':None,'source':{'turn_idx':9,'quote':None,'span':None}})
    batch=CandidateBatch(candidates=(CandidateFact(fact=fact),))
    before=batch.model_dump_json();after=TariffValueNormalizer(CATALOG).normalize(batch,None)
    assert batch.model_dump_json()==before
    old=fact.model_dump();new=after.candidates[0].fact.model_dump()
    assert new=={**old,'value':'Хайп+'}

def test_other_attribute_and_unknown_object_stay_untouched():
    fact=Fact.model_validate({'object':{'name':'роутер','type':'device'},'attribute':'service_state',
        'value':'тариф Хайп+','polarity':1,'speaker':'client','valid_from':'2026-11-15T12:00:00',
        'valid_to':None,'source':{'turn_idx':9}})
    batch=CandidateBatch(candidates=(CandidateFact(fact=fact),))
    assert TariffValueNormalizer(CATALOG).normalize(batch,None)==batch
