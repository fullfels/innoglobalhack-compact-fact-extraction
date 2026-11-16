from pathlib import Path
import importlib.util,json,time,types,sys
import pytest
ROOT=Path(__file__).resolve().parents[1]
from final_eval import evaluate
from quality_guard import primary_metrics,stress_guard
from ml.qwen.replay_v22 import load_evaluator

def test_unchanged_official_evaluator_and_baseline():
    ev=load_evaluator(ROOT/'evaluation/case_metrics.py')
    read=lambda n:{c['call_id']:c for c in map(json.loads,(ROOT/'evaluation'/n).read_text(encoding='utf-8').splitlines())}
    assert ev.evaluate(read('val.jsonl'),read('baseline_pred.jsonl'))==json.loads((ROOT/'evaluation/baseline_metrics.json').read_text())

def test_real_quality_is_not_replaced_by_better_synthetic_quality():
    r={'metrics':{'partial_serving':{'fact_recall':.9}},'metrics_by_slice':{'official_real':{'partial_serving':{'fact_recall':.2}}}}
    assert primary_metrics(r,'partial_serving')['fact_recall']==.2
    before={'metrics_by_slice':{'synthetic_stress':{'partial_serving':{'fact_recall':.8,'temporal_recall':.5,'state_accuracy':.5}}}}
    after={'metrics_by_slice':{'synthetic_stress':{'partial_serving':{'fact_recall':.5,'temporal_recall':.8,'state_accuracy':.5}}}}
    assert stress_guard(after,before,'partial_serving','partial_serving')[0] is False

def test_actual_generation_cache_failures_and_real_synthetic_slices(tmp_path,monkeypatch):
    class Cuda:
        def reset_peak_memory_stats(self):pass
        def synchronize(self):pass
        def max_memory_allocated(self):return 100
        def max_memory_reserved(self):return 200
    monkeypatch.setitem(sys.modules,'torch',types.SimpleNamespace(cuda=Cuda()))
    root=tmp_path/'root';root.mkdir();(root/'evaluation').mkdir()
    for n in ['catalog.json','case_metrics.py']:(root/'evaluation'/n).write_bytes((ROOT/'evaluation'/n).read_bytes())
    (root/'package_manifest.json').write_text('{}');(root/'prompt.txt').write_text('Instruction')
    f={'object':{'name':'Интернет','type':'service'},'attribute':'service_state','value':'не работает','polarity':1,'speaker':'client',
       'valid_from':'2026-07-28T12:00:00','valid_to':None,'source':{'turn_idx':2,'span':None,'quote':None}}
    calls=[{'call_id':cid,'started_at':'2026-07-28T12:00:00','turns':[{'idx':2,'speaker':'client','text':'Интернет не работает.'}],
            'facts':[{**f,'object':{**f['object'],'id':cid+':service'}}]} for cid in ['real1','real2','synthetic-v23-unit']]
    class Engine:
        count=0;last_raw=None;last_diagnostics=None
        def generate(self,call):
            self.count+=1;assert 'facts' not in call.model_dump()
            if call.call_id=='real2':
                from src.pipeline.adapters import PipelineError
                self.last_raw='{"facts":[';raise PipelineError('QWEN_OUTPUT_LIMIT')
            self.last_raw=json.dumps({'facts':[f]},ensure_ascii=False);return self.last_raw
    engine=Engine();out=tmp_path/'out'
    first=evaluate(engine,calls,root,out,'adapter','fp16',time.time()+1000)
    again=evaluate(engine,calls,root,out,'adapter','fp16',time.time()+1000)
    assert engine.count==3 and first['metrics']==again['metrics']
    assert first['metrics']['partial_serving']['fact_recall']==.6667
    assert first['metrics_by_slice']['official_real']['partial_serving']['fact_recall']==.5
    assert first['metrics_by_slice']['synthetic_stress']['partial_serving']['fact_recall']==1.
    assert first['statuses']['partial']=={'ok':2,'failed':1}
    with pytest.raises(ValueError):evaluate(engine,calls,root,out,'different_adapter','fp16',time.time()+1000)
