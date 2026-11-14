"""Official evaluator wrapper. python -m src.experiments.evaluate --help"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
from pydantic import ValidationError
from src.contracts.models import CallInput, Fact
from .common import ExperimentError, factory, identity, read_json, rows, sha, write_json

METRICS = ('fact_precision','fact_recall','temporal_recall','state_accuracy','grounding_turn','grounding_span')

def features(row):
    # Explicit allowlist: facts/confirmed and all labels never reach ML or DE.
    return CallInput.model_validate({k:row[k] for k in CallInput.model_fields if k in row})

def validate_record(row, call, *, gold=False):
    if not gold and set(row)-set(CallInput.model_fields)-{'facts'}:raise ExperimentError('unknown prediction fields')
    if not isinstance(row.get('facts'),list): raise ExperimentError('facts must be an array')
    turns = {t.idx:t for t in call.turns}
    warnings=[]
    for fact_index,f in enumerate(row['facts']):
        try:
            if not {'object','attribute','value','polarity','valid_from','valid_to','source'} <= f.keys(): raise ValueError()
            idx = f['source']['turn_idx']
            if type(idx) is not int or idx not in turns: raise ValueError()
            turn = turns[idx]
            data = dict(f)
            if gold:
                data.pop('confirmed',None)
                data['object']=dict(data['object'])
                if data['object'].get('type') not in {None,'service','device','ticket','client'}:
                    warnings.append({'fact_index':fact_index,'code':'GOLD_OPTIONAL_OBJECT_TYPE_OUTSIDE_SCHEMA'})
                    data['object'].pop('type')
            data.setdefault('speaker', turn.speaker)
            fact = Fact.model_validate(data)
            if fact.speaker != turn.speaker: raise ValueError()
            if not gold:
                s = fact.source
                if s.span is not None:
                    start,end = s.span
                    if not 0 <= start < end <= len(turn.text): raise ValueError()
                    if s.quote is not None and s.quote != turn.text[start:end]: raise ValueError()
                elif s.quote is not None and s.quote not in turn.text: raise ValueError()
        except (ValidationError,ValueError,TypeError,KeyError,AttributeError):
            raise ExperimentError('invalid fact schema/source') from None
    # Full prediction envelopes are never accepted as records.
    if row['call_id'] != call.call_id: raise ExperimentError('call_id mismatch')
    if not gold:
        for key in CallInput.model_fields:
            if key in row and key != 'call_id' and row[key] != call.model_dump(mode='json')[key]:
                # datetime strings may have equivalent spelling; validate structured metadata.
                merged = call.model_dump(mode='json');merged[key]=row[key]
                if CallInput.model_validate(merged) != call: raise ExperimentError('prediction changed input metadata')
    return warnings

def preflight(gold_path, pred_path):
    gold,pred = rows(gold_path),rows(pred_path)
    missing,extra = sorted(gold.keys()-pred.keys()),sorted(pred.keys()-gold.keys())
    report={'valid':False,'expected_calls':len(gold),'prediction_calls':len(pred),'missing_call_ids':missing,'extra_call_ids':extra,
        'gold_grounding_policy':'structural references only; original noisy gold spans/quotes unchanged'}
    if missing or extra: return gold,pred,report
    for cid,g in gold.items():
        call=features(g)
        warnings=validate_record(g,call,gold=True)
        if warnings:report.setdefault('gold_annotation_warnings',[]).append({'call_id':cid,'warnings':warnings})
        validate_record(pred[cid],call)
    report['valid']=True
    return gold,pred,report

def generate(gold_path, pipeline, output):
    with Path(output).open('w',encoding='utf-8') as f:
        for row in rows(gold_path).values():
            prediction,_ = pipeline.run(features(row))
            f.write(prediction.model_dump_json()+'\n')

def evaluate(*, gold_path, predictions, evaluator, out, model_id, experiment_id,
             model_config=None, evaluation_config=None, dataset_version='unknown', mode='unknown',
             pipeline_factory=None, published_baseline=None, recomputed_baseline=None):
    model_config=deepcopy(model_config or {});evaluation_config=deepcopy(evaluation_config or {})
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    manifest={**identity(),'schema_version':'1','experiment_id':experiment_id,'model_id':model_id,'mode':mode,
        'dataset':{'sha256':None,'version':dataset_version},'model_config':model_config or {},
        'evaluation_config':evaluation_config or {},'evaluator_sha256':None,'status':'running'}
    write_json(out/'manifest.json',manifest)
    (out/'evaluation.log').write_text('',encoding='utf-8')
    try:
        manifest['dataset']['sha256']=sha(gold_path)
        manifest['evaluator_sha256']=sha(evaluator)
        expected=evaluation_config.get('expected_evaluator_sha256')
        if expected and expected!=manifest['evaluator_sha256']:raise ExperimentError('evaluator hash mismatch')
        if pipeline_factory:
            pipeline=factory(pipeline_factory)(deepcopy(model_config))
            adapter=pipeline.extractor
            try:
                if adapter.descriptor.model_id != model_id: raise ExperimentError('model_id does not match adapter')
                manifest['mode']=adapter.descriptor.mode
                if callable(getattr(adapter,'load',None)):adapter.load()
                if getattr(adapter,'state','unloaded')!='ready':raise ExperimentError('model not ready')
                generate(gold_path,pipeline,out/'predictions.jsonl')
            finally:
                if callable(getattr(adapter,'close',None)):adapter.close()
        else:
            # Copy bytes, not a dict keyed by call_id: duplicate IDs must remain detectable.
            (out/'predictions.jsonl').write_bytes(Path(predictions).read_bytes())
        validated_predictions_hash=sha(out/'predictions.jsonl')
        gold_records,pred_records,report=preflight(gold_path,out/'predictions.jsonl')
        if sha(gold_path)!=manifest['dataset']['sha256']:raise ExperimentError('gold changed during run')
        write_json(out/'validation_report.json',report)
        if not report['valid']: raise ExperimentError('call_id coverage mismatch')
        config=evaluation_config or {}
        result=subprocess.run([sys.executable,str(Path(evaluator).resolve()),str(Path(gold_path).resolve()),
            str((out/'predictions.jsonl').resolve()),'--json',str((out/'metrics.json').resolve())],
            capture_output=True,text=True,timeout=config.get('timeout_sec',300))
        # Successful official stdout contains aggregate metrics only. Do not expose failing script traceback/input.
        (out/'evaluation.log').write_text(result.stdout if result.returncode==0 else 'Official evaluator failed\n',encoding='utf-8')
        if result.returncode: raise ExperimentError('official evaluator failed')
        if (sha(gold_path)!=manifest['dataset']['sha256'] or sha(evaluator)!=manifest['evaluator_sha256'] or sha(out/'predictions.jsonl')!=validated_predictions_hash):
            raise ExperimentError('evaluation inputs changed after validation')
        metrics=read_json(out/'metrics.json')
        for key in METRICS:
            v=metrics[key]
            if v is not None and (type(v) not in (int,float) or not 0<=v<=1): raise ExperimentError('invalid evaluator metric')
        for key,expected in {'n_calls':len(gold_records),'n_gold_facts':sum(len(r['facts']) for r in gold_records.values()),'n_pred_facts':sum(len(r['facts']) for r in pred_records.values())}.items():
            if key in metrics and (type(metrics[key]) is not int or metrics[key]!=expected):raise ExperimentError('metric count mismatch')
        if published_baseline:
            write_json(out/'published_baseline.json',{'source':'organizer published; NOT local recomputation','metrics':read_json(published_baseline),'sha256':sha(published_baseline)})
        if recomputed_baseline:
            write_json(out/'recomputed_baseline.json',{'source':'local official-script recomputation; NOT organizer published','metrics':read_json(recomputed_baseline),'sha256':sha(recomputed_baseline)})
        manifest.update(status='complete',predictions_sha256=sha(out/'predictions.jsonl'),metrics_sha256=sha(out/'metrics.json'))
        return metrics
    except Exception as exc:
        manifest.update(status='failed',error_type=type(exc).__name__)
        if not (out/'validation_report.json').exists(): write_json(out/'validation_report.json',{'valid':False,'error_type':type(exc).__name__})
        write_json(out/'metrics.json',{'status':'not_evaluated','error_type':type(exc).__name__})
        with (out/'evaluation.log').open('a') as f:f.write('Experiment failed: '+type(exc).__name__+'\n')
        raise
    finally: write_json(out/'manifest.json',manifest)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gold',required=True);p.add_argument('--evaluator',required=True);p.add_argument('--out',required=True)
    source=p.add_mutually_exclusive_group(required=True);source.add_argument('--predictions');source.add_argument('--factory')
    p.add_argument('--model-id',required=True);p.add_argument('--experiment-id',required=True)
    p.add_argument('--dataset-version',required=True);p.add_argument('--model-config');p.add_argument('--evaluation-config')
    p.add_argument('--mode',choices=['real','mock','synthetic','unknown'],default='unknown');p.add_argument('--published-baseline');p.add_argument('--recomputed-baseline')
    a=p.parse_args()
    try:
        evaluate(gold_path=a.gold,predictions=a.predictions,evaluator=a.evaluator,out=a.out,model_id=a.model_id,
            experiment_id=a.experiment_id,dataset_version=a.dataset_version,mode=a.mode,
            model_config=read_json(a.model_config) if a.model_config else {},evaluation_config=read_json(a.evaluation_config) if a.evaluation_config else {},
            pipeline_factory=a.factory,published_baseline=a.published_baseline,recomputed_baseline=a.recomputed_baseline)
        return 0
    except Exception as exc:
        print('Evaluation failed: '+type(exc).__name__,file=sys.stderr);return 2

if __name__=='__main__':sys.exit(main())
