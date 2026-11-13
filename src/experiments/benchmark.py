"""Sequential batch=1 pipeline benchmark, raw samples are the source of summaries."""
import argparse
from dataclasses import dataclass, field
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import threading
import time
from typing import Callable
from src.contracts.models import CallPrediction
from .common import factory, identity, read_json, rows, sha, write_json
from .evaluate import features

@dataclass
class BenchmarkTarget:
    pipeline: object
    device: str
    synthetic: bool = False
    parameters: int | None = None
    checkpoint_paths: tuple[str,...] = ()
    close: Callable = field(default=lambda:None)

def rss_bytes():
    try:
        if sys.platform.startswith('linux'):
            for line in Path('/proc/self/status').read_text().splitlines():
                if line.startswith('VmRSS:'): return int(line.split()[1])*1024
        if sys.platform == 'darwin':
            return int(subprocess.check_output(['ps','-o','rss=','-p',str(os.getpid())],text=True))*1024
    except (OSError,ValueError,subprocess.SubprocessError): pass
    return None

class MemorySampler:
    def __init__(self, interval=.02): self.interval=interval;self.peak=None;self.stop=threading.Event();self.samples=[]
    def sample(self):
        value=rss_bytes()
        if value is not None:
            self.peak=max(self.peak or 0,value);self.samples.append({'phase':'memory','rss_bytes':value,'monotonic_sec':time.perf_counter()})
    def loop(self):
        while not self.stop.wait(self.interval):self.sample()
    def __enter__(self):
        self.sample();self.thread=threading.Thread(target=self.loop,daemon=True);self.thread.start();return self
    def __exit__(self,*args):self.stop.set();self.thread.join();self.sample()

def hardware(device):
    versions={}
    for name in ['fastapi','pydantic','torch','transformers','psutil']:
        try:versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:versions[name]=None
    return {'os':platform.platform(),'machine':platform.machine(),'cpu':platform.processor() or platform.machine(),
        'physical_ram_bytes':os.sysconf('SC_PAGE_SIZE')*os.sysconf('SC_PHYS_PAGES') if hasattr(os,'sysconf') else None,'logical_cores':os.cpu_count(),'python':platform.python_version(),'libraries':versions,'device':device,
        'rss_method':'current process RSS sampled every 20ms; includes loader/warmup; not process tree or lifetime high-water',
        'disk_cache':'uncontrolled; cold means new target instance, not cold OS disk cache'}

def summarize(raw):
    warm=[r for r in raw if r['phase']=='warm']
    good=[r for r in warm if r['status']=='ok']
    durations=sorted(r['elapsed_sec'] for r in good)
    windows=[r['elapsed_sec'] for r in raw if r['phase']=='warm_run']
    wall=sum(windows) if windows else sum(r['elapsed_sec'] for r in warm)
    return {'attempts':len(warm),'successes':len(good),'errors':len(warm)-len(good),
        'schema_valid_percent':100*sum(r['schema_valid'] for r in warm)/len(warm) if warm else None,
        'mean_latency_sec':statistics.mean(durations) if durations else None,
        'median_latency_sec':statistics.median(durations) if durations else None,
        'p95_latency_sec':durations[math.ceil(.95*len(durations))-1] if durations else None,
        'p95_method':'nearest-rank; successful warm calls only',
        'warm_wall_sec':wall,'calls_per_sec':len(good)/wall if wall>0 else None,
        'calls_per_hour':3600*len(good)/wall if wall>0 else None,
        'cold_start_sec':[r['elapsed_sec'] for r in raw if r['phase']=='cold_start'],
        'load_sec':[r['elapsed_sec'] for r in raw if r['phase']=='load']}

def checkpoint_size(paths):
    if not paths:return None
    files=set()
    for name in paths:
        p=Path(name)
        if not p.exists():return None
        files.update(x.resolve() for x in (p.rglob('*') if p.is_dir() else [p]) if x.is_file())
    return sum(p.stat().st_size for p in files)

def benchmark(build, calls, *, out, config, device='cpu', warmup_calls=None, repeats=3, cold_trials=3,
              clock=time.perf_counter, exclusive_gpu=False, experiment_id='benchmark'):
    if repeats<1 or cold_trials<1:raise ValueError('positive repeats/cold_trials required')
    calls=tuple(calls);warmup_calls=tuple(warmup_calls or ())
    if not calls:raise ValueError('empty benchmark inputs')
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    hw=hardware(device);raw=[];target=None;cuda=None
    manifest={**identity(),'experiment_id':experiment_id,'config':config,'device':device,'batch':1,'concurrency':1,
        'repeats':repeats,'cold_trials':cold_trials,'warmup_calls':len(warmup_calls),
        'call_ids':[c.call_id for c in calls],'warmup_ids':[c.call_id for c in warmup_calls],
        'status':'running','gpu_allocation':'exclusive one GPU' if exclusive_gpu else 'not asserted'}
    write_json(out/'hardware.json',hw)
    try:
        if device!='cpu':
            if not device.startswith('cuda:'):raise ValueError('supported devices: cpu or explicit cuda:N')
            try:
                import torch
                if not torch.cuda.is_available():raise RuntimeError()
                cuda=torch.cuda;cuda.set_device(device)
                hw['cuda']={'name':cuda.get_device_name(device),'runtime':torch.version.cuda,'vram_total_bytes':cuda.get_device_properties(device).total_memory}
            except (ImportError,RuntimeError,ValueError):
                manifest.update(status='not_measured',reason='requested CUDA device unavailable')
                write_json(out/'benchmark_summary.json',{'status':'not_measured','reason':manifest['reason']})
                return None
        def sync():
            if cuda:cuda.synchronize(device)
        def measured(phase,fn,repeat=0,cid=None):
            sync();start=clock();status='ok';valid=False;error_type=None;result=None
            try:
                result=fn()
                if phase!='load':CallPrediction.model_validate(result[0]);valid=True
            except Exception as exc:status='error';error_type=type(exc).__name__
            finally:sync()
            elapsed=clock()-start
            if elapsed<0:raise ValueError('clock moved backwards')
            raw.append({'phase':phase,'repeat':repeat,'call_id':cid,'elapsed_sec':elapsed,'status':status,'schema_valid':valid,'error_type':error_type})
            if phase=='load' and status!='ok':raise RuntimeError('loader failed')
            return result
        if cuda:cuda.reset_peak_memory_stats(device)
        with MemorySampler() as memory:
            for trial in range(cold_trials):
                begin=clock()
                target=measured('load',lambda:build({**config,'device':device}),trial)
                if target.device!=device:raise ValueError('factory device mismatch')
                manifest.update(model_id=target.pipeline.extractor.descriptor.model_id,mode=target.pipeline.extractor.descriptor.mode,
                    engine=target.pipeline.extractor.descriptor.engine,synthetic=target.synthetic or target.pipeline.extractor.descriptor.mode=='mock')
                measured('cold',lambda:target.pipeline.run(calls[0]),trial,calls[0].call_id)
                raw.append({'phase':'cold_start','elapsed_sec':clock()-begin,'repeat':trial})
                if trial<cold_trials-1:target.close();target=None
            for call in warmup_calls:measured('warmup',lambda:target.pipeline.run(call),cid=call.call_id)
            for repeat in range(repeats):
                sync();begin=clock()
                for call in calls:measured('warm',lambda:target.pipeline.run(call),repeat,call.call_id)
                sync();raw.append({'phase':'warm_run','repeat':repeat,'elapsed_sec':clock()-begin})
        raw.extend(memory.samples)
        summary=summarize(raw)
        summary.update(status='complete',synthetic=manifest['synthetic'],device=device,peak_ram_bytes=memory.peak,
            ram_status='measured' if memory.peak is not None else 'not_measured',parameters=target.parameters,
            checkpoint_bytes=checkpoint_size(target.checkpoint_paths),
            peak_vram_bytes=cuda.max_memory_reserved(device) if cuda else None,
            vram_status='pytorch allocator reserved peak' if cuda else 'not_measured: CPU run',
            gpu_seconds_per_1000=summary['warm_wall_sec']*1000/summary['attempts'] if cuda and exclusive_gpu else None,
            gpu_seconds_method='one exclusively allocated GPU wall seconds including CPU phases/errors' if cuda and exclusive_gpu else 'not_measured: no exclusive CUDA allocation asserted')
        raw.append({'phase':'resources',**{k:summary[k] for k in ['peak_ram_bytes','peak_vram_bytes','parameters','checkpoint_bytes']}})
        write_json(out/'benchmark_summary.json',summary)
        manifest['status']='complete'
        return summary
    except Exception as exc:
        manifest.update(status='failed',error_type=type(exc).__name__)
        write_json(out/'benchmark_summary.json',{'status':'failed','error_type':type(exc).__name__})
        raise
    finally:
        try:
            if target:target.close()
        finally:
            with (out/'benchmark_raw.jsonl').open('w') as f:
                for row in raw:f.write(json.dumps(row,allow_nan=False)+'\n')
            write_json(out/'hardware.json',hw)
            manifest['raw_sha256']=sha(out/'benchmark_raw.jsonl')
            write_json(out/'benchmark_manifest.json',manifest)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for arg in ['factory','inputs','out','config','experiment-id']:p.add_argument('--'+arg,required=True)
    p.add_argument('--warmup-inputs');p.add_argument('--device',default='cpu');p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--cold-trials',type=int,default=3);p.add_argument('--exclusive-gpu',action='store_true')
    a=p.parse_args()
    try:
        benchmark(factory(a.factory),[features(r) for r in rows(a.inputs).values()],out=a.out,config=read_json(a.config),device=a.device,
            warmup_calls=[features(r) for r in rows(a.warmup_inputs).values()] if a.warmup_inputs else (),
            repeats=a.repeats,cold_trials=a.cold_trials,exclusive_gpu=a.exclusive_gpu,experiment_id=a.experiment_id)
        return 0
    except Exception as exc:print('Benchmark failed: '+type(exc).__name__,file=sys.stderr);return 2

if __name__=='__main__':sys.exit(main())
