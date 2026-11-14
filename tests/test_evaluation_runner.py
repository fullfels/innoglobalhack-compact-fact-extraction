import json
import os
from pathlib import Path
import pytest
from src.contracts.models import CallInput
from src.pipeline.adapters import MockExtractor
from src.pipeline.service import Pipeline
from src.experiments.common import ExperimentError, rows
from src.experiments.evaluate import evaluate, features, generate, preflight, main

@pytest.fixture
def bundle(tmp_path):
    call={'call_id':'synthetic-eval','started_at':'2026-11-13T12:00:00','turns':[{'idx':7,'speaker':'client','text':'Интернет не работает.'}]}
    pred,_=Pipeline(MockExtractor()).run(CallInput.model_validate(call))
    row=pred.model_dump(mode='json');row['facts'][0]['object']['id']='fixture:service'
    g=tmp_path/'gold.jsonl';p=tmp_path/'pred.jsonl'
    g.write_text(json.dumps(row)+'\n');p.write_text(json.dumps(row)+'\n')
    return g,p,row

@pytest.mark.parametrize('case',['invalid_json','duplicate','missing_field','missing_id','extra_id','polarity','date','source','attribute','span','quote'])
def test_invalid_predictions(bundle,case):
    g,p,row=bundle
    if case=='invalid_json':p.write_text('{bad\n')
    elif case=='duplicate':p.write_text(p.read_text()*2)
    else:
        if case=='missing_field':del row['facts'][0]['valid_from']
        if case=='missing_id':del row['call_id']
        if case=='extra_id':row['call_id']='extra'
        if case=='polarity':row['facts'][0]['polarity']=0
        if case=='date':row['facts'][0]['valid_from']=0
        if case=='source':row['facts'][0]['source']['turn_idx']=99
        if case=='attribute':row['facts'][0]['attribute']='invalid'
        if case=='span':row['facts'][0]['source']['span']=[0,999]
        if case=='quote':row['facts'][0]['source']['quote']='not literal'
        p.write_text(json.dumps(row)+'\n')
    if case=='extra_id':
        report=preflight(g,p)[2];assert not report['valid'] and report['missing_call_ids'] and report['extra_call_ids']
    else:
        with pytest.raises((ValueError,TypeError)):preflight(g,p)


def test_minimum_official_prediction(bundle):
    g,p,row=bundle
    f=row['facts'][0];del f['speaker'];f['source']={'turn_idx':7};f['object']={'name':'Интернет'}
    p.write_text(json.dumps({'call_id':row['call_id'],'facts':[f]})+'\n')
    assert preflight(g,p)[2]['valid']


def test_features_do_not_leak_gold(bundle,tmp_path):
    g,_,row=bundle
    row['secret_target']='no leakage';row['facts'][0]['confirmed']=True
    g.write_text(json.dumps(row)+'\n')
    class Spy:
        def run(self,call):
            assert 'facts' not in call.model_dump() and 'secret_target' not in call.model_dump()
            return Pipeline(MockExtractor()).run(call)
    output=tmp_path/'generated.jsonl';generate(g,Spy(),output)
    assert rows(output)[row['call_id']]['facts']
    assert 'secret_target' not in features(row).model_dump()


def test_evaluator_invocation_and_artifacts(bundle,tmp_path):
    g,p,_=bundle
    script=tmp_path/'evaluator_spy.py'
    script.write_text("import json,sys\nfrom pathlib import Path\na=json.loads(Path(sys.argv[1]).read_text());b=json.loads(Path(sys.argv[2]).read_text())\nassert a['call_id']==b['call_id']\njson.dump(dict.fromkeys(['fact_precision','fact_recall','temporal_recall','state_accuracy','grounding_turn','grounding_span'],1.0),open(sys.argv[4],'w'))\n")
    out=tmp_path/'run'
    metrics=evaluate(gold_path=g,predictions=p,evaluator=script,out=out,model_id='synthetic',experiment_id='fixture',mode='synthetic')
    assert metrics['fact_precision']==1
    assert {x.name for x in out.iterdir()} >= {'predictions.jsonl','metrics.json','manifest.json','evaluation.log','validation_report.json'}
    assert json.loads((out/'manifest.json').read_text())['status']=='complete'
    with pytest.raises(FileExistsError):evaluate(gold_path=g,predictions=p,evaluator=script,out=out,model_id='x',experiment_id='x')


def test_cli_failure_is_nonzero(bundle,tmp_path,monkeypatch):
    g,p,_=bundle;p.write_text('{}\n')
    script=tmp_path/'unused.py';script.write_text("raise RuntimeError('must not run')")
    out=tmp_path/'failed'
    monkeypatch.setattr('sys.argv',['evaluate','--gold',str(g),'--predictions',str(p),'--evaluator',str(script),'--out',str(out),'--model-id','x','--experiment-id','x','--dataset-version','fixture'])
    assert main()==2
    assert json.loads((out/'manifest.json').read_text())['status']=='failed'
    assert not json.loads((out/'validation_report.json').read_text())['valid']

@pytest.mark.skipif(not os.getenv('OFFICIAL_EVALUATOR'),reason='Official evaluator supplied separately')
def test_actual_official_evaluator(bundle,tmp_path):
    g,p,_=bundle
    result=evaluate(gold_path=g,predictions=p,evaluator=os.environ['OFFICIAL_EVALUATOR'],out=tmp_path/'official',model_id='fixture',experiment_id='official-format-test',mode='synthetic')
    assert result['n_calls']==1
    assert all(result[k]==1 for k in ['fact_precision','fact_recall','temporal_recall','state_accuracy','grounding_turn','grounding_span'])

@pytest.mark.parametrize('fail',[False,True])
def test_factory_load_once_and_release_without_gold(bundle,tmp_path,monkeypatch,fail):
    from types import SimpleNamespace
    g,_,_=bundle;events=[]
    class Resource:
        descriptor=SimpleNamespace(model_id='loaded-fixture',mode='synthetic')
        state='unloaded'
        def load(self):events.append('load');self.state='ready'
        def close(self):events.append('close');self.state='unloaded'
    adapter=Resource()
    class Target:
        extractor=adapter
        def run(self,call):
            assert adapter.state=='ready' and 'facts' not in call.model_dump()
            if fail:raise RuntimeError('controlled failure')
            return Pipeline(MockExtractor()).run(call)
    monkeypatch.setattr('src.experiments.evaluate.factory',lambda spec:lambda config:Target())
    script=tmp_path/'protocol_spy.py'
    script.write_text("import json,sys\njson.dump(dict.fromkeys(['fact_precision','fact_recall','temporal_recall','state_accuracy','grounding_turn','grounding_span'],1),open(sys.argv[4],'w'))\n")
    kwargs=dict(gold_path=g,predictions=None,evaluator=script,out=tmp_path/'factory',model_id='loaded-fixture',experiment_id='factory',pipeline_factory='fixture:factory')
    if fail:
        with pytest.raises(RuntimeError):evaluate(**kwargs)
    else:evaluate(**kwargs)
    assert events==['load','close'] and adapter.state=='unloaded'


def test_empty_asr_turn_is_preserved_not_dropped_or_renumbered(bundle):
    _,_,row=bundle
    row['turns'].insert(0,{'idx':3,'speaker':'operator','text':''})
    call=features(row)
    assert [t.idx for t in call.turns]==[3,7]
    assert call.turns[0].text==''
    assert json.loads(call.model_dump_json())['turns'][0]['text']==''


def test_gold_optional_type_anomaly_is_reported_without_relaxing_predictions(bundle):
    g,p,row=bundle
    row['facts'][0]['object']['type']='other'
    g.write_text(json.dumps(row)+'\n')
    report=preflight(g,p)[2]
    assert report['valid'] and report['gold_annotation_warnings'][0]['warnings'][0]['code']=='GOLD_OPTIONAL_OBJECT_TYPE_OUTSIDE_SCHEMA'
    assert json.loads(g.read_text())['facts'][0]['object']['type']=='other'
    p.write_text(json.dumps(row)+'\n')
    with pytest.raises(ExperimentError):preflight(g,p)
