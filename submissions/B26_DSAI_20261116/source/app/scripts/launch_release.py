"""Start the existing API from a generated config; preflight always runs first."""
import argparse
import json
import os
from pathlib import Path
import sys


def command(config_path,*,port=8000,check_only=False,environ=None):
    if type(port) is not int or not 1<=port<=65535:raise ValueError('Invalid port')
    c=json.loads(Path(config_path).read_text(encoding='utf-8'))
    if c.get('version')!=1:raise ValueError('Unknown deployment config version')
    model=c['model']
    if set(model)!={'model_dir','base_path','device','precision','manifest_sha256','model_id'}:
        raise ValueError('Unknown or missing deployment settings')
    if model['device'] not in {'cpu','cuda'} or model['precision']!='fp16':
        raise ValueError('HTTP serving uses the verified fp16 / CPU fp32 profile')
    env=dict(os.environ if environ is None else environ)
    root=Path(__file__).resolve().parents[1]
    env.update({'MWS_MODE':'real','MWS_PIPELINE_FACTORY':'ml.qwen.serving:factory',
        'MTS_MODEL_DIR':model['model_dir'],'MTS_MODEL_BASE_PATH':model['base_path'],
        'MTS_MODEL_DEVICE':model['device'],'MWS_MODEL_ID':model['model_id'],
        'MWS_ARTIFACT_SHA256':model['manifest_sha256'],'MWS_TORCH_VERSION':c['torch_release'],
        'HF_HUB_OFFLINE':'1'})
    env['PYTHONPATH']=str(root)+(os.pathsep+env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
    args=[sys.executable,'-m','scripts.release_preflight']
    if check_only:args+=['--check-only']
    else:args += [sys.executable,'-m','uvicorn','src.api.app:app','--host','127.0.0.1','--port',str(port),'--workers','1']
    return args,env,root


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=Path('runtime_config.json'))
    p.add_argument('--port',type=int,default=8000)
    p.add_argument('--check-only',action='store_true')
    a=p.parse_args();args,env,root=command(a.config,port=a.port,check_only=a.check_only)
    os.chdir(root)
    os.execvpe(args[0],args,env)

if __name__=='__main__':main()
