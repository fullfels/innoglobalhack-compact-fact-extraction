"""Tokenizer audit without optimization. Local compatibility override is explicitly recorded."""
import argparse,hashlib,importlib.metadata,json
from pathlib import Path
from collections import Counter
from temporal_loss import encode_record

def audit(root,compat=False):
    from transformers import AutoTokenizer
    kwargs={'local_files_only':True}
    if compat:kwargs['extra_special_tokens']={}
    tok=AutoTokenizer.from_pretrained(root/'parent500/tokenizer',**kwargs)
    if not tok.is_fast:raise ValueError('Fast tokenizer with offsets required')
    rows=[json.loads(s) for s in (root/'sft/train.jsonl').read_text(encoding='utf-8').splitlines()]
    lengths=[];masked=weighted=supervised=0;levels=Counter()
    for r in rows:
        x=encode_record(r,tok);lengths.append(len(x['input_ids']));masked+=x['masked_timestamp_tokens'];weighted+=x['weighted_timestamp_tokens']
        supervised+=sum(y!=-100 for y in x['labels']);levels.update(p['level'] for p in r['date_supervision'])
    return {'passed':True,'rows':len(rows),'max_length':max(lengths),'min_length':min(lengths),'mean_length':sum(lengths)/len(lengths),
        'masked_timestamp_tokens':masked,'weighted_timestamp_tokens':weighted,'supervised_tokens':supervised,'date_policy_levels':dict(levels),
        'transformers':importlib.metadata.version('transformers'),'tokenizers':importlib.metadata.version('tokenizers'),
        'local_compatibility_override':compat,'native_colab_audit_required':True,'tokenizer_sha256':hashlib.sha256((root/'parent500/tokenizer/tokenizer.json').read_bytes()).hexdigest()}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--local-compat',action='store_true');p.add_argument('--out',type=Path);a=p.parse_args()
    result=audit(a.root,a.local_compat);text=json.dumps(result,ensure_ascii=False,indent=2)
    if a.out:a.out.write_text(text+'\n',encoding='utf-8')
    print(text)
