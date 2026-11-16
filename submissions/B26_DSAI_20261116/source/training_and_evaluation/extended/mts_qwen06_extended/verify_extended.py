"""Sealed data, input isolation, dates, exact token offsets/masks; no model weights."""
import argparse,hashlib,json,sys
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
def read(p):return [json.loads(s) for s in p.read_text(encoding='utf-8').splitlines() if s.strip()]
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tokenizer',action='store_true');args=ap.parse_args()
    from ml.qwen.extended_qwen06 import verify_payload
    from ml.qwen.night_support import supervised_target,encoded_with_mask,DATE_VALUE
    from ml.qwen.train import input_call,load_module
    from ml.qwen.bridge import model_input,messages,parse_candidates
    from ml.qwen.prompting import effective_example,effective_instruction
    from src.pipeline.service import validate_output
    manifest=verify_payload(ROOT)
    catalog=json.loads((ROOT/'official_catalog.json').read_text(encoding='utf-8'))
    instruction=effective_instruction((ROOT/'prompt.txt').read_text(encoding='utf-8'))
    if args.tokenizer:
        from transformers import AutoTokenizer
        tokenizer=AutoTokenizer.from_pretrained(ROOT/'tokenizer',local_files_only=True)
        encode=load_module('qwen06_plain_tokenizer',ROOT/'tokenize_sft.py').encode_record
    selection=json.loads((ROOT/'selection.json').read_text(encoding='utf-8'))
    report={'status':'passed','weights_loaded':False,'gpu_used':False,'training_executed':False,'splits':{}}
    accounts=[];texts=[];ids=[]
    for split,n in [('train',2863),('dev',366)]:
        raw=read(ROOT/'data'/(split+'_ready.jsonl'));sft=read(ROOT/'sft'/(split+'.jsonl'))
        assert len(raw)==len(sft)==n
        accounts.append({(c.get('series') or {}).get('account_id') or c['call_id'].split(':')[0] for c in raw})
        texts.append({json.dumps(c['turns'],ensure_ascii=False,sort_keys=True) for c in raw});ids.append({c['call_id'] for c in raw})
        lengths=[];prompts=[];answers=[];masked_tokens=0;proofs=0;masked_fields=0;attrs=Counter()
        for c,row in zip(raw,sft):
            assert c['call_id']==row['call_id'];call=input_call(c);view=effective_example(row)
            assert set(model_input(call))=={'call_id','started_at','turns'}
            assert messages(call,instruction,catalog)==view['prompt']
            assert input_call({**c,'facts':[{'leak':'DO_NOT_USE'}]}).model_dump()==call.model_dump()
            target,proof,mask=supervised_target(c)
            assert json.loads(row['completion'][0]['content'])==target
            assert row['masked_date_fields']==[]
            assert row['date_review_candidates']==[{**m,'reason':'Not independently verified by conservative rules; retained as v2 gold supervision in this run'} for m in mask]
            parsed=parse_candidates(row['completion'][0]['content'],call)
            validate_output(call,[f.fact for f in parsed.candidates])
            assert len(target['facts'])==len(c['facts'])
            for f,g in zip(target['facts'],c['facts']):
                assert f['object']=={k:g['object'][k] for k in ('name','type')}
                assert all(f[k]==g[k] for k in ('attribute','value','polarity','speaker','valid_from','valid_to'))
                assert f['source']['turn_idx']==g['source']['turn_idx']
                attrs[f['attribute']]+=1
            proofs+=len(proof);masked_fields+=len(mask)
            if args.tokenizer:
                e=encode(view,tokenizer,max_length=24576)
                e['answer_start']=next(i for i,v in enumerate(e['labels']) if v!=-100)
                a=e['answer_start'];assert all(x==-100 for x in e['labels'][:a])
                assert e['labels'][-1]==tokenizer.eos_token_id
                assert all(y==-100 or y==x for x,y in zip(e['input_ids'],e['labels']))
                # Full assistant target, including all dates, is supervised exactly.
                assert e['labels'][a:]==e['input_ids'][a:]
                assert tokenizer.decode(e['input_ids'][a:-1],skip_special_tokens=False,clean_up_tokenization_spaces=False)==view['completion'][0]['content']
                lengths.append(len(e['input_ids']));prompts.append(a);answers.append(len(e['input_ids'])-a)
        item={'calls':n,'facts':sum(len(c['facts']) for c in raw),'attributes':dict(attrs),
            'verified_date_fields':proofs,'date_review_candidates':masked_fields,'masked_date_fields':0,'targets_preserved':True,
            'source_roles_checked':True,'gold_input_isolation':True,'training_inference_prompt_identical':True}
        if args.tokenizer:
            assert max(answers)<2048 and max(prompts)+2048<=24576
            item.update(max_total_tokens=max(lengths),max_prompt_tokens=max(prompts),max_answer_tokens=max(answers),
                mean_total_tokens=round(sum(lengths)/n,2),date_tokens_excluded_from_loss=masked_tokens,
                completion_mask_checked_for_all_rows=True,truncation=False)
        report['splits'][split]=item
    assert not(accounts[0]&accounts[1]) and not(texts[0]&texts[1]) and not(ids[0]&ids[1])
    assert not(ids[0]&set(selection['old_train512_excluded']))
    assert set(selection['full_eval32_ids'])<=ids[1] and len(selection['full_eval32_ids'])==32
    assert selection['full_eval32_ids'][:8]==selection['selection_dev8_ids']
    assert len(selection['train_order_indices'])==2863 and sorted(selection['train_order_indices'])==list(range(2863))
    report['no_account_call_text_overlap']=True;report['old512_not_reused']=True
    (ROOT/'LOCAL_VALIDATION.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
