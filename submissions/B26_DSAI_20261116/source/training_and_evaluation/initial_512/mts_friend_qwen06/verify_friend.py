"""Verify the sealed friend payload without loading model weights."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))

def read(path):
    return [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines() if s.strip()]

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tokenizer',action='store_true');args=ap.parse_args()
    manifest=json.loads((ROOT/'manifest.json').read_text(encoding='utf-8'))
    for rel,expected in manifest['sha256'].items():
        path=(ROOT/rel).resolve()
        assert path.is_relative_to(ROOT.resolve()) and path.is_file(),rel
        assert hashlib.sha256(path.read_bytes()).hexdigest()==expected,rel
    from ml.qwen.bridge import parse_candidates,model_input,messages
    from ml.qwen.train import input_call,load_module
    from ml.qwen.prompting import effective_example,effective_instruction
    from src.pipeline.service import validate_output
    catalog=json.loads((ROOT/'official_catalog.json').read_text(encoding='utf-8'))
    instruction=effective_instruction((ROOT/'prompt.txt').read_text(encoding='utf-8'))
    report={'status':'passed','sha256_checked':True,'gpu_training_run':False,'quality_measured':False,'splits':{}}
    accounts=[];texts=[];all_ids=[]
    if args.tokenizer:
        from transformers import AutoTokenizer
        tokenizer=AutoTokenizer.from_pretrained(ROOT/'tokenizer',local_files_only=True)
        encode=load_module('friend_tokenization_check',ROOT/'tokenize_sft.py').encode_record
    for name,n in [('train_512',512),('dev_20',20)]:
        raw=read(ROOT/'data'/(name+'.jsonl'));sft=read(ROOT/'sft'/(name+'.jsonl'))
        assert len(raw)==len(sft)==n
        assert [r['call_id'] for r in raw]==[r['call_id'] for r in sft]
        assert len({r['call_id'] for r in raw})==n
        ac={(r.get('series') or {}).get('account_id') or r['call_id'].split(':')[0] for r in raw}
        tx={json.dumps(r['turns'],ensure_ascii=False,sort_keys=True) for r in raw}
        accounts.append(ac);texts.append(tx);all_ids.append({r['call_id'] for r in raw})
        lengths=[];answers=[];attrs=Counter()
        for r,s in zip(raw,sft):
            view=effective_example(s);call=input_call(r)
            assert set(model_input(call))=={'call_id','started_at','turns'}
            assert messages(call,instruction,catalog)==view['prompt'],r['call_id']
            # A changed label must not change inference input.
            fake={**r,'facts':[{'leak_probe':'DO_NOT_LEAK_GOLD'}]}
            assert model_input(input_call(fake))==model_input(call)
            batch=parse_candidates(view['completion'][0]['content'],call)
            validate_output(call,[x.fact for x in batch.candidates])
            assert len(batch.candidates)==len(r['facts'])
            for target,original in zip(json.loads(view['completion'][0]['content'])['facts'],r['facts']):
                assert target['object']=={k:original['object'][k] for k in ('name','type')}
                for key in ('attribute','value','polarity','speaker','valid_from','valid_to'):
                    assert target[key]==original[key],(r['call_id'],key)
                assert target['source']['turn_idx']==original['source']['turn_idx']
                attrs[target['attribute']]+=1
            if args.tokenizer:
                enc=encode(view,tokenizer,max_length=16384)
                assert enc['labels'].count(-100)>0
                answers.append(sum(v!=-100 for v in enc['labels']));lengths.append(len(enc['input_ids']))
        item={'calls':n,'facts':sum(len(r['facts']) for r in raw),'attributes':dict(attrs),
            'empty_calls':sum(not r['facts'] for r in raw),'negative_calls':sum(any(f['polarity']==-1 for f in r['facts']) for r in raw),
            'gold_input_isolation':True,'source_roles_and_schema_checked':True,'targets_preserved':True,'training_inference_messages_exact':True}
        if lengths:
            assert max(answers)<1536
            item.update(tokenizer_checked=True,max_total_tokens=max(lengths),max_answer_tokens_with_eos=max(answers),
                total_supervised_tokens=sum(answers),mean_total_tokens=round(sum(lengths)/len(lengths),2))
        report['splits'][name]=item
    assert not(accounts[0]&accounts[1]) and not(texts[0]&texts[1]) and not(all_ids[0]&all_ids[1])
    report['split_isolation']=True
    output=ROOT/'LOCAL_VALIDATION_TOKENIZER.json' if args.tokenizer else ROOT/'LOCAL_VALIDATION.json'
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
