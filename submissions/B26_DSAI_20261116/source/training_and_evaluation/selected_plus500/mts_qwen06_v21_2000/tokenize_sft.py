"""Exact chat boundary/EOS, assistant-only loss and selective timestamp uncertainty masks."""
import json,re
DATE=re.compile(r'(?<!\\)"(valid_from|valid_to)":("[^"\\]*"|null)')

def encode_record(row,tokenizer,max_length=24576):
    prompt=row['prompt'];answer=row['completion'][0]['content']
    assert [m['role'] for m in prompt]==['system','user'] and row['completion'][0]['role']=='assistant'
    prefix=tokenizer.apply_chat_template(prompt,tokenize=False,add_generation_prompt=True,enable_thinking=False)
    full=tokenizer.apply_chat_template(prompt+row['completion'],tokenize=False,add_generation_prompt=False,enable_thinking=False)
    assert full.startswith(prefix),'Training/inference prefix mismatch'
    p=tokenizer.encode(prefix,add_special_tokens=False)
    result=tokenizer(full,add_special_tokens=False,return_offsets_mapping=True)
    ids=result['input_ids'];offsets=result['offset_mapping'];n=len(p)
    assert ids[:n]==p,'Prefix token boundary merged'
    eos=tokenizer.eos_token_id;assert ids[n:].count(eos)==1
    end=ids.index(eos,n)+1;ids=ids[:end];offsets=offsets[:end]
    assert tokenizer.decode(ids[n:-1],skip_special_tokens=False,clean_up_tokenization_spaces=False)==answer
    labels=[-100]*n+ids[n:];fields=list(DATE.finditer(answer))
    facts=json.loads(answer)['facts'];assert len(fields)==2*len(facts)
    masked=[]
    for item in row.get('masked_date_fields',[]):
        m=fields[2*item['fact_index']+int(item['field']=='valid_to')]
        assert m.group(1)==item['field'] and m.group(2)!='null'
        # Mask timestamp content; a BPE token overlapping it must be wholly excluded.
        a,b=len(prefix)+m.start(2)+1,len(prefix)+m.end(2)-1
        for i,(lo,hi) in enumerate(offsets):
            if lo<b and hi>a:
                assert i>=n and i<len(ids)-1
                if labels[i]!=-100:masked.append(i)
                labels[i]=-100
    assert ids[-1]==labels[-1]==eos and all(v==-100 for v in labels[:n])
    assert any(v!=-100 for v in labels[n:])
    if len(ids)>max_length:raise ValueError('No truncation allowed: '+row['call_id'])
    return {'input_ids':ids,'attention_mask':[1]*len(ids),'labels':labels,'answer_start':n,
        'masked_date_tokens':len(masked),'masked_date_token_indices':masked}

def suffix_loss(model,row):
    import torch
    import torch.nn.functional as F
    n=row['answer_start'];labels=row['labels'];a=len(labels)-n
    assert n>0 and all(v==-100 for v in labels[:n])
    x=torch.tensor([row['input_ids']],device='cuda');y=torch.tensor([labels[n:]],device='cuda')
    logits=model(input_ids=x,attention_mask=torch.ones_like(x),use_cache=False,logits_to_keep=a+1).logits[:,:-1]
    assert logits.shape[:2]==y.shape
    total=logits.new_zeros((),dtype=torch.float32)
    for i in range(0,a,64):total=total+F.cross_entropy(logits[:,i:i+64].float().reshape(-1,logits.shape[-1]),y[:,i:i+64].reshape(-1),ignore_index=-100,reduction='sum')
    return total/(y!=-100).sum()

def loss_equivalence(model,tokenizer):
    import torch
    from ml.qwen.train import autocast_for,check_loss_equivalence
    report=check_loss_equivalence(model,tokenizer)
    row={'call_id':'loss-mask-selftest','prompt':[{'role':'system','content':'Верни JSON.'},{'role':'user','content':'Проверка.'}],
         'completion':[{'role':'assistant','content':'{"facts":[{"valid_from":"2026-11-15T00:00:00","valid_to":null}]}'}],
         'masked_date_fields':[{'fact_index':0,'field':'valid_from'}]}
    encoded=encode_record(row,tokenizer);assert encoded['masked_date_tokens']>0
    x=torch.tensor([encoded['input_ids']],device='cuda');labels=torch.tensor([encoded['labels']],device='cuda')
    model.eval()
    with torch.no_grad(),autocast_for(model):
        reference=model(input_ids=x,attention_mask=torch.ones_like(x),labels=labels,use_cache=False).loss
        actual=suffix_loss(model,encoded)
    torch.testing.assert_close(reference.float(),actual.float(),rtol=.003,atol=.003)
    report['selective_timestamp_mask']={'passed':True,'masked_tokens':encoded['masked_date_tokens'],'reference':float(reference),'suffix':float(actual)}
    return report
