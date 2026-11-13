"""Compare completed evaluation/benchmark bundles; never invent absent measurements."""
import argparse
import csv
import importlib.util
import itertools
import hashlib
import json
from pathlib import Path
import sys
from .common import ExperimentError, read_json, rows, sha, write_json
from .evaluate import METRICS, preflight, features
from .benchmark import summarize

RESOURCE_FIELDS=('parameters','checkpoint_bytes','peak_ram_bytes','peak_vram_bytes','mean_latency_sec','median_latency_sec','p95_latency_sec','calls_per_sec','calls_per_hour','gpu_seconds_per_1000')

def evaluator_module(path):
    spec=importlib.util.spec_from_file_location('official_case_metrics',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def fact_analysis(official,gold,pred):
    records=[];statuses={}
    for cid,g in gold.items():
        p=pred[cid];gf=g['facts'];pf=p['facts']
        gi={id(f):i for i,f in enumerate(gf)};pi={id(f):i for i,f in enumerate(pf)}
        pairs=official.match_facts(gf,pf);usedg=set();usedp=set()
        for a,b,_ in pairs:
            i,j=gi[id(a)],pi[id(b)];usedg.add(i);usedp.add(j)
            issues=[]
            if any((a.get(k) or '')[:10]!=(b.get(k) or '')[:10] for k in ['valid_from','valid_to']):issues.append('temporal_error')
            sa,sb=a['source'],b['source']
            if sa['turn_idx']!=sb['turn_idx']:issues.append('grounding_turn_error')
            elif sa.get('span') and sb.get('span') and max(sa['span'][0],sb['span'][0])>=min(sa['span'][1],sb['span'][1]):issues.append('grounding_span_error')
            label=issues or ['correct']
            statuses[(cid,i)]=label
            records.append({'call_id':cid,'attribute':a['attribute'],'gold_index':i,'prediction_index':j,'categories':label})
        for i,a in enumerate(gf):
            if i not in usedg:
                statuses[(cid,i)]=['missing']
                records.append({'call_id':cid,'attribute':a['attribute'],'gold_index':i,'prediction_index':None,'categories':['missing']})
        for j,b in enumerate(pf):
            if j not in usedp:records.append({'call_id':cid,'attribute':b['attribute'],'gold_index':None,'prediction_index':j,'categories':['extra']})
        # Auxiliary near-miss labels do NOT replace official missing/extra counts.
        available=set(range(len(pf)))-usedp
        for i in sorted(set(range(len(gf)))-usedg):
            a=gf[i];candidates=[]
            for j in available:
                b=pf[j]
                probe={**b,'value':a['value'],'polarity':a['polarity']}
                if official.match_facts([a],[probe]):candidates.append((official._value_sim(a['value'],b['value']),j))
            if candidates:
                _,j=max(candidates,key=lambda x:(x[0],-x[1]));available.remove(j);b=pf[j];issues=[]
                if (a['polarity']>0)!=(b['polarity']>0):issues.append('polarity_error')
                if official._value_sim(a['value'],b['value'])<official.VALUE_SIM:issues.append('value_error')
                records.append({'call_id':cid,'attribute':a['attribute'],'gold_index':i,'prediction_index':j,'categories':issues,'auxiliary':True})
        for attr in official.SINGLE_VALUED:
            a={**g,'facts':[f for f in gf if f['attribute']==attr]};b={**p,'facts':[f for f in pf if f['attribute']==attr]}
            score=official.evaluate({cid:a},{cid:b})
            if score['n_state'] and score['state_accuracy']<1:
                records.append({'call_id':cid,'attribute':attr,'categories':['state_error'],'official_state_accuracy':score['state_accuracy']})
    return records,statuses

def pair_delta(a,b):
    result={f'{k}_delta_pp':100*(b[k]-a[k]) if a.get(k) is not None and b.get(k) is not None else None for k in METRICS}
    for key in ['checkpoint_bytes','peak_ram_bytes','peak_vram_bytes']:
        x,y=a.get(key),b.get(key)
        result[key+'_delta_bytes']=y-x if x is not None and y is not None else None
        result[key+'_reduction_percent']=100*(1-y/x) if x and y is not None else None
    result['speedup']=a.get('median_latency_sec')/b['median_latency_sec'] if a.get('median_latency_sec') is not None and b.get('median_latency_sec') else None
    return result

def compare(runs,benchmarks,*,gold_path,evaluator,out,plots=True):
    if len(runs)<2:raise ExperimentError('at least two completed runs required')
    gold=rows(gold_path);official=evaluator_module(evaluator);table=[];analyses={};states={};reference=None;hw_reference=None;protocol=None
    for label,folder in runs.items():
        folder=Path(folder);manifest=read_json(folder/'manifest.json')
        if manifest['status']!='complete' or manifest['dataset']['sha256']!=sha(gold_path) or manifest['evaluator_sha256']!=sha(evaluator):
            raise ExperimentError('incompatible/incomplete evaluation bundle')
        if manifest['predictions_sha256']!=sha(folder/'predictions.jsonl') or manifest['metrics_sha256']!=sha(folder/'metrics.json'):
            raise ExperimentError('evaluation artifact hash mismatch')
        _,pred,report=preflight(gold_path,folder/'predictions.jsonl')
        if not report['valid']:raise ExperimentError('invalid prediction coverage')
        metrics=read_json(folder/'metrics.json')
        # Guard against manually edited metrics: compute again with the supplied unchanged evaluator.
        if official.evaluate(gold,pred)!=metrics:raise ExperimentError('metrics do not match official recomputation')
        row={'label':label,'model_id':manifest['model_id'],'mode':manifest['mode'],**{k:metrics[k] for k in METRICS}}
        if reference is None:reference=manifest['mode']
        elif manifest['mode']!=reference:raise ExperimentError('cannot compare mixed synthetic/mock/real modes')
        if label in benchmarks:
            root=Path(benchmarks[label]);bm=read_json(root/'benchmark_manifest.json');summary=read_json(root/'benchmark_summary.json');hw=read_json(root/'hardware.json')
            if bm['status']!='complete' or bm['model_id']!=row['model_id']:raise ExperimentError('incomplete/model-mismatched benchmark')
            if bm['summary_sha256']!=sha(root/'benchmark_summary.json'):raise ExperimentError('benchmark summary hash mismatch')
            raw=[json.loads(line) for line in (root/'benchmark_raw.jsonl').read_text().splitlines()]
            if any(summary.get(k)!=v for k,v in summarize(raw).items()):raise ExperimentError('summary does not match raw samples')
            digest=hashlib.sha256(json.dumps([features(gold[cid]).model_dump(mode='json') for cid in bm['call_ids']],sort_keys=True).encode()).hexdigest()
            if digest!=bm['input_features_sha256']:raise ExperimentError('benchmark inputs differ from evaluation dataset')
            if bm['raw_sha256']!=sha(root/'benchmark_raw.jsonl'):raise ExperimentError('benchmark raw hash mismatch')
            fingerprint={k:hw.get(k) for k in ['os','machine','cpu','logical_cores','physical_ram_bytes','device','cuda']}
            current={k:bm.get(k) for k in ['device','batch','concurrency','repeats','cold_trials','warmup_calls','call_ids','warmup_ids','input_features_sha256','warmup_features_sha256','gpu_allocation']}
            if hw_reference is None:hw_reference=fingerprint;protocol=current
            elif fingerprint!=hw_reference or current!=protocol:raise ExperimentError('hardware/protocol mismatch; split comparisons by device')
            row.update({k:summary.get(k) for k in RESOURCE_FIELDS});row['device']=hw['device'];row['synthetic_resources']=bm['synthetic']
        else:row.update({k:None for k in RESOURCE_FIELDS});row['device']='not_measured'
        table.append(row);analyses[label],states[label]=fact_analysis(official,gold,pred)
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    write_json(out/'comparison.json',table)
    with (out/'comparison.csv').open('w',newline='') as f:
        keys=list(dict.fromkeys(k for r in table for k in r));writer=csv.DictWriter(f,keys);writer.writeheader();writer.writerows(table)
    write_json(out/'pairwise.json',[{'a':a['label'],'b':b['label'],**pair_delta(a,b)} for a,b in itertools.combinations(table,2)])
    with (out/'fact_errors.jsonl').open('w') as f:
        for label,records in analyses.items():
            for r in records:f.write(json.dumps({'configuration':label,**r},ensure_ascii=False)+'\n')
    grouped={}
    for label,records in analyses.items():
        grouped[label]={}
        for r in records:
            for category in r['categories']:
                key=r['attribute']+':'+category
                grouped[label][key]=grouped[label].get(key,0)+1
    write_json(out/'errors_by_attribute.json',grouped)
    with (out/'pairwise_facts.jsonl').open('w') as f:
        for a,b in itertools.combinations(runs,2):
            for cid,call in gold.items():
                for idx,fact in enumerate(call['facts']):f.write(json.dumps({'a':a,'b':b,'call_id':cid,'gold_index':idx,'attribute':fact['attribute'],'a_categories':states[a][(cid,idx)],'b_categories':states[b][(cid,idx)]})+'\n')
    if plots:plot(table,out)
    write_json(out/'comparison_manifest.json',{'dataset_sha256':sha(gold_path),'evaluator_sha256':sha(evaluator),'labels':list(runs),'evaluation_manifests':{label:sha(Path(folder)/'manifest.json') for label,folder in runs.items()},'benchmark_manifests':{label:sha(Path(folder)/'benchmark_manifest.json') for label,folder in benchmarks.items()},
        'quality_axis':'fact_recall','resource_policy':'missing values omitted; device/hardware protocols checked',
        'error_matching':'official match_facts; auxiliary value/polarity near-misses are diagnostic only'})
    return table

def plot(table,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    panels=[('quality_latency','median_latency_sec','fact_recall'),('quality_memory','peak_ram_bytes','fact_recall'),
        ('temporal_resources','peak_ram_bytes','temporal_recall'),('state_resources','peak_ram_bytes','state_accuracy'),
        ('pareto','median_latency_sec','fact_recall')]
    for name,xkey,ykey in panels:
        fig,ax=plt.subplots(figsize=(7,4));points=[]
        for r in table:
            x,y=r.get(xkey),r.get(ykey)
            if x is None or y is None:continue
            points.append((x,y,r['label']));ax.scatter(x,y);ax.annotate(r['label'],(x,y))
        if name=='pareto':
            frontier=sorted((x,y) for x,y,_ in points if not any(xx<=x and yy>=y and (xx<x or yy>y) for xx,yy,_ in points))
            if frontier:ax.plot(*zip(*frontier),linestyle='--')
        ax.set_xlabel(xkey);ax.set_ylabel(ykey)
        synthetic=any(r['mode']!='real' or r.get('synthetic_resources') for r in table)
        ax.set_title(('SYNTHETIC / MOCK · ' if synthetic else '')+name)
        if not points:ax.text(.5,.5,'Resources not measured',ha='center',transform=ax.transAxes)
        fig.tight_layout();fig.savefig(out/(name+'.png'),dpi=150);plt.close(fig)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',action='append',required=True,help='label=completed evaluation directory')
    p.add_argument('--benchmark',action='append',default=[],help='label=completed benchmark directory')
    p.add_argument('--gold',required=True);p.add_argument('--evaluator',required=True);p.add_argument('--out',required=True);p.add_argument('--no-plots',action='store_true')
    a=p.parse_args()
    try:
        def mapping(items):
            result={}
            for item in items:
                key,sep,value=item.partition('=')
                if not sep or not key or key in result:raise ExperimentError('invalid/duplicate label')
                result[key]=value
            return result
        compare(mapping(a.run),mapping(a.benchmark),gold_path=a.gold,evaluator=a.evaluator,out=a.out,plots=not a.no_plots)
        return 0
    except Exception as exc:print('Comparison failed: '+type(exc).__name__,file=sys.stderr);return 2

if __name__=='__main__':sys.exit(main())
