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

@pytest.fixture
def audit_bundles(tmp_path):
    if not os.getenv('OFFICIAL_EVALUATOR'):pytest.skip('Official evaluator supplied separately')
    call=CallInput(call_id='audit-comparison',started_at='2026-11-13T12:00:00',turns=[{'idx':1,'speaker':'client','text':'Интернет не работает.'}])
    row=Pipeline(MockExtractor()).run(call)[0].model_dump(mode='json');row['facts'][0]['object']['id']='fixture:svc'
    gold=tmp_path/'gold.jsonl';gold.write_text(json.dumps(row)+'\n')
    runs={};benches={}
    for label in ['A','B']:
        runs[label]=tmp_path/(label+'-eval');benches[label]=tmp_path/(label+'-bench')
        evaluate(gold_path=gold,predictions=gold,evaluator=os.environ['OFFICIAL_EVALUATOR'],out=runs[label],model_id='synthetic-fixtures-v1',experiment_id=label,mode='mock',model_config={})
        benchmark(lambda c:BenchmarkTarget(Pipeline(MockExtractor()),'cpu',synthetic=True),[call],out=benches[label],config={},cold_trials=1,repeats=1)
    return gold,runs,benches,call

@pytest.mark.parametrize('damage',['configuration','protocol','resource_peak','mixed_device','benchmark_mode','failed_samples'])
def test_incompatible_quality_resource_reports_rejected(audit_bundles,tmp_path,damage):
    from src.experiments.common import sha
    gold,runs,benches,call=audit_bundles
    def change(path,fn):
        data=json.loads(path.read_text());fn(data);path.write_text(json.dumps(data));return data
    if damage=='configuration':change(benches['B']/'benchmark_manifest.json',lambda m:m.update(config={'decode_budget':1}))
    elif damage=='protocol':change(runs['B']/'manifest.json',lambda m:m.update(evaluation_config={'comparison_protocol':{'postprocessing':'different'}}))
    elif damage=='resource_peak':
        change(benches['B']/'benchmark_summary.json',lambda s:s.update(peak_ram_bytes=0))
        change(benches['B']/'benchmark_manifest.json',lambda m:m.update(summary_sha256=sha(benches['B']/'benchmark_summary.json')))
    elif damage=='mixed_device':
        for b in benches.values():change(b/'benchmark_manifest.json',lambda m:m.update(device='cuda:0'))
    elif damage=='benchmark_mode':change(benches['B']/'benchmark_manifest.json',lambda m:m.update(mode='real'))
    elif damage=='failed_samples':
        class FailingWarm:
            extractor=MockExtractor()
            count=0
            def run(self,c):
                self.count+=1
                if self.count>1:raise RuntimeError('warm failure')
                return Pipeline(self.extractor).run(c)
        benches['B']=tmp_path/'failed-resource'
        benchmark(lambda c:BenchmarkTarget(FailingWarm(),'cpu',synthetic=True),[call],out=benches['B'],config={},cold_trials=1,repeats=1)
    with pytest.raises(ExperimentError):compare(runs,benches,gold_path=gold,evaluator=os.environ['OFFICIAL_EVALUATOR'],out=tmp_path/'report',plots=False)


def test_official_fact_counts_are_order_independent(audit_bundles,tmp_path):
    gold,runs,benches,_=audit_bundles
    out=tmp_path/'counts'
    compare(runs,benches,gold_path=gold,evaluator=os.environ['OFFICIAL_EVALUATOR'],out=out,plots=False)
    counts=json.loads((out/'fact_counts.json').read_text())
    assert counts['A']['audit-comparison']['service_state']=={'tp':1,'fp':0,'fn':0}
    records=[json.loads(s) for s in (out/'fact_errors.jsonl').read_text().splitlines()]
    assert any(r.get('official_fact_match') is True for r in records)


def test_real_comparison_requires_provenance_not_only_label(audit_bundles,tmp_path):
    gold,runs,_,_=audit_bundles
    for root in runs.values():
        path=root/'manifest.json';data=json.loads(path.read_text());data['mode']='real';path.write_text(json.dumps(data))
    with pytest.raises(ExperimentError,match='provenance'):
        compare(runs,{},gold_path=gold,evaluator=os.environ['OFFICIAL_EVALUATOR'],out=tmp_path/'unknown-real',plots=False)
