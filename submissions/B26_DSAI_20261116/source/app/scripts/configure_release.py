"""Prepare a deployment configuration; never changes model weights or proof fields."""
import argparse
import importlib.metadata
import json
import re
from pathlib import Path

from ml.qwen.serving import verify_export, verify_base, sha


def quoted(value):
    value=str(value)
    if '\n' in value or '\r' in value or '\0' in value:
        raise ValueError('Configuration values cannot contain line breaks or NUL')
    # Compose single-quoted values are literal, including dollar signs.
    return "'"+value.replace("'", "\\'")+"'"


def prepare(export,base,out,*,device='cpu',torch_version=None,torch_index=None):
    if device not in {'cpu','cuda'}:raise ValueError('Use cpu or cuda')
    export,m=verify_export(export)
    base=verify_base(base)
    if torch_version is None:
        try:torch_version=importlib.metadata.version('torch')
        except importlib.metadata.PackageNotFoundError:
            raise ValueError('Specify --torch-version using the verified runtime release') from None
    if not re.fullmatch(r'\d+(?:\.\d+){1,3}(?:\+[A-Za-z0-9.]+)?',torch_version):
        raise ValueError('An exact stable Torch version is required')
    if device=='cpu':
        torch_version=torch_version.split('+')[0]
        torch_index=torch_index or 'https://download.pytorch.org/whl/cpu'
        if torch_index.rstrip('/')!='https://download.pytorch.org/whl/cpu':
            raise ValueError('The prepared CPU profile requires the official CPU wheel index')
    elif not torch_index or not re.fullmatch(r'https://download\.pytorch\.org/whl/cu\d+/?',torch_index):
        raise ValueError('For CUDA provide the verified --torch-index URL matching that Torch build')
    model={'model_dir':str(export),'base_path':str(base),'device':device,'precision':'fp16',
           'manifest_sha256':sha(export/'model_manifest.json'),'model_id':m['model_id']}
    config={'version':1,'status':'configuration_only_not_native_acceptance','model':model,
            'torch_release':torch_version,'torch_index_url':torch_index,
            'weights_modified':False,'model_loaded':False,'quality_threshold_pass':None}
    values={'MWS_EXPORT_DIR':export.as_posix(),'MWS_BASE_DIR':base.as_posix(),
            'MTS_MODEL_DEVICE':device,'MWS_MODEL_ID':m['model_id'],
            'MWS_ARTIFACT_SHA256':model['manifest_sha256'],
            'MWS_TORCH_VERSION':torch_version,'MWS_TORCH_INDEX_URL':torch_index}
    env='\n'.join(k+'='+quoted(v) for k,v in values.items())+'\n'
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    (out/'runtime_config.json').write_text(json.dumps(config,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (out/'.env.release').write_text(env,encoding='utf-8')
    return config


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--export',type=Path,required=True)
    p.add_argument('--base',type=Path,required=True)
    p.add_argument('--out',type=Path,default=Path('.'))
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    p.add_argument('--torch-version')
    p.add_argument('--torch-index')
    a=p.parse_args()
    c=prepare(a.export,a.base,a.out,device=a.device,torch_version=a.torch_version,torch_index=a.torch_index)
    print(json.dumps({'status':c['status'],'model_id':c['model']['model_id'],
                      'files':[str(a.out/'runtime_config.json'),str(a.out/'.env.release')],
                      'model_loaded':False,'quality_threshold_pass':None},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
