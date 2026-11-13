import json
import os
from pathlib import Path
import pytest
from src.experiments.compare import pair_delta, compare
from src.experiments.common import ExperimentError
from src.experiments.evaluate import evaluate
from src.experiments.benchmark import benchmark, BenchmarkTarget
from src.pipeline.service import Pipeline
from src.pipeline.adapters import MockExtractor
from src.contracts.models import CallInput


def test_delta_pp_speedup_and_unknowns():
    result=pair_delta({'fact_recall':.7,'median_latency_sec':2,'checkpoint_bytes':100},
        {'fact_recall':.8,'median_latency_sec':1,'checkpoint_bytes':60})
    assert result['fact_recall_delta_pp']==pytest.approx(10)
    assert result['speedup']==2 and result['checkpoint_bytes_reduction_percent']==40
    assert result['grounding_span_delta_pp'] is None


def test_no_fake_reports_without_data(tmp_path):
    with pytest.raises(ExperimentError):compare({}, {},gold_path='unused',evaluator='unused',out=tmp_path/'out')
    assert not (tmp_path/'out').exists()

@pytest.mark.skipif(not os.getenv('OFFICIAL_EVALUATOR'),reason='Official evaluator supplied separately')
def test_actual_three_run_comparison_and_plots(tmp_path):
    pytest.importorskip('matplotlib')
    call=CallInput(call_id='comparison-fixture',started_at='2026-11-13T12:00:00',turns=[{'idx':1,'speaker':'client','text':'Интернет не работает.'},{'idx':2,'speaker':'client','text':'Я не использую роутер.'}])
    row=Pipeline(MockExtractor()).run(call)[0].model_dump(mode='json');row['facts'][0]['object']['id']='fixture:svc'
    gold=tmp_path/'gold.jsonl';gold.write_text(json.dumps(row)+'\n')
    runs={};benches={}
    for label in ['A','B','C']:
        pred=tmp_path/(label+'.jsonl');altered=json.loads(json.dumps(row))
        if label=='B':
            altered['facts'][0]['valid_from']='2020-01-01T00:00:00'
            altered['facts'][0]['source']={'turn_idx':2}
        if label=='C':
            altered['facts'][0]['polarity']=-1
            altered['facts'][0]['value']='работает нормально'
        pred.write_text(json.dumps(altered)+'\n')
        runs[label]=tmp_path/(label+'-eval')
        evaluate(gold_path=gold,predictions=pred,evaluator=os.environ['OFFICIAL_EVALUATOR'],out=runs[label],model_id='synthetic-fixtures-v1',experiment_id=label,mode='synthetic')
        benches[label]=tmp_path/(label+'-bench')
        benchmark(lambda c:BenchmarkTarget(Pipeline(MockExtractor()),'cpu',synthetic=True),[call],out=benches[label],config={},cold_trials=1,repeats=1)
    out=tmp_path/'compare'
    table=compare(runs,benches,gold_path=gold,evaluator=os.environ['OFFICIAL_EVALUATOR'],out=out)
    assert len(table)==3
    assert len(json.loads((out/'pairwise.json').read_text()))==3
    assert len(list(out.glob('*.png')))==5
    errors=[json.loads(x) for x in (out/'fact_errors.jsonl').read_text().splitlines()]
    assert any('polarity_error' in x['categories'] for x in errors)
    assert any('missing' in x['categories'] for x in errors)
    for category in ['value_error','temporal_error','grounding_turn_error','state_error']:
        assert any(category in x['categories'] for x in errors)
    summary=benches['A']/'benchmark_summary.json';data=json.loads(summary.read_text());data['median_latency_sec']=0
    summary.write_text(json.dumps(data))
    with pytest.raises(ExperimentError,match='hash mismatch'):
        compare(runs,benches,gold_path=gold,evaluator=os.environ['OFFICIAL_EVALUATOR'],out=tmp_path/'tampered')
