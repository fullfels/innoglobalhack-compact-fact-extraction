"""Artifact helpers. Experiment outputs are private and never overwritten."""
import hashlib
import importlib
import json
import math
import subprocess
from pathlib import Path
from datetime import datetime, timezone

class ExperimentError(ValueError):
    pass

def finite_float(value):
    number=float(value)
    if not math.isfinite(number):raise ExperimentError('non-finite JSON number')
    return number

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''): h.update(chunk)
    return h.hexdigest()

def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')

def read_json(path):
    def pairs(items):
        out = {}
        for k,v in items:
            if k in out: raise ExperimentError('duplicate JSON key')
            out[k]=v
        return out
    return json.loads(Path(path).read_text(encoding='utf-8'), object_pairs_hook=pairs, parse_float=finite_float,
        parse_constant=lambda _: (_ for _ in ()).throw(ExperimentError('non-finite JSON value')))

def rows(path):
    out = {}
    with Path(path).open(encoding='utf-8') as f:
        for line_no,line in enumerate(f,1):
            if not line.strip(): raise ExperimentError(f'blank JSONL line {line_no}')
            try:
                # Reuse the same strict decoder, without including sensitive content in errors.
                row = json.loads(line, object_pairs_hook=unique_keys, parse_float=finite_float,
                    parse_constant=lambda _: (_ for _ in ()).throw(ValueError()))
                cid = row['call_id']
                if not isinstance(cid,str) or not cid.strip(): raise ValueError()
                if cid in out: raise ExperimentError(f'duplicate call_id at line {line_no}')
                out[cid] = row
            except ExperimentError: raise
            except (ValueError,TypeError,KeyError): raise ExperimentError(f'invalid JSONL record at line {line_no}') from None
    if not out: raise ExperimentError('empty JSONL')
    return out

def unique_keys(items):
    out = {}
    for k,v in items:
        if k in out: raise ExperimentError('duplicate JSON key')
        out[k]=v
    return out

def factory(spec):
    module, sep, name = spec.partition(':')
    if not sep: raise ExperimentError('factory must be module:function')
    return getattr(importlib.import_module(module),name)

def identity():
    try: commit = subprocess.check_output(['git','rev-parse','HEAD'],stderr=subprocess.DEVNULL,text=True).strip()
    except (OSError,subprocess.CalledProcessError): commit = None
    return {'git_commit':commit, 'timestamp_utc':datetime.now(timezone.utc).isoformat()}
