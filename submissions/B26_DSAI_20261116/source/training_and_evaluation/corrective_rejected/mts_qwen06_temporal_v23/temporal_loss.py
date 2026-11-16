"""Completion-only labels with independently auditable timestamp character masks."""
import json,re
DATE=re.compile(r'(?<!\\)"(valid_from|valid_to)":("[^"\\]*"|null)')

def encode_record(row,tokenizer,max_length=24576,date_weight=2.):
    prompt=row['prompt'];answer=row['completion'][0]['content']
    if [m['role'] for m in prompt]!=['system','user'] or row['completion'][0]['role']!='assistant':raise ValueError('Wrong chat roles')
    prefix=tokenizer.apply_chat_template(prompt,tokenize=False,add_generation_prompt=True,enable_thinking=False)
    full=tokenizer.apply_chat_template(prompt+row['completion'],tokenize=False,add_generation_prompt=False,enable_thinking=False)
    if not full.startswith(prefix):raise ValueError('Training/inference prefix mismatch')
    prefix_ids=tokenizer.encode(prefix,add_special_tokens=False);n=len(prefix_ids)
    tokenized=tokenizer(full,add_special_tokens=False,return_offsets_mapping=True);ids=tokenized['input_ids'];offsets=tokenized['offset_mapping']
    if ids[:n]!=prefix_ids:raise ValueError('Prefix merged at token boundary')
    eos=tokenizer.eos_token_id
    if ids[n:].count(eos)!=1:raise ValueError('Expected exactly one answer EOS')
    end=ids.index(eos,n)+1;ids=ids[:end];offsets=offsets[:end]
    if tokenizer.decode(ids[n:-1],skip_special_tokens=False,clean_up_tokenization_spaces=False)!=answer:raise ValueError('Answer roundtrip mismatch')
    if len(ids)>max_length:raise ValueError('No silent truncation: '+row['call_id'])
    labels=[-100]*n+ids[n:];weights=[0.]*n+[1.]*(len(ids)-n)
    fields=list(DATE.finditer(answer));facts=json.loads(answer)['facts']
    if len(fields)!=2*len(facts):raise ValueError('Unexpected timestamp serialization')
    policy={}
    for p in row['date_supervision']:
        key=p['fact_index'],p['field']
        if key in policy or p['level'] not in {'day','full'}:raise ValueError('Duplicate/invalid date policy')
        if not 0<=p['fact_index']<len(facts) or p['field'] not in {'valid_from','valid_to'}:raise ValueError('Invalid date-policy index')
        policy[key]=p['level']
    masked=weighted=0;field_audit=[]
    for index,m in enumerate(fields):
        fact_index=index//2;field=m[1]
        if field!=('valid_from' if index%2==0 else 'valid_to'):raise ValueError('Timestamp field order changed')
        if m[2]=='null':continue
        lo=len(prefix)+m.start(2)+1;hi=len(prefix)+m.end(2)-1
        level=policy.get((fact_index,field),'unknown');allowed_hi=lo+10 if level=='day' else hi if level=='full' else lo
        field_masked=[];field_weighted=[]
        for i,(a,b) in enumerate(offsets):
            if a>=hi or b<=lo:continue
            if i<n or i==len(ids)-1:raise ValueError('Date overlaps prompt/EOS')
            # Any BPE token touching unknown characters is wholly excluded.
            if b>allowed_hi or level=='unknown':
                if labels[i]!=-100:masked+=1
                labels[i]=-100;weights[i]=0.;field_masked.append(i)
            else:
                weights[i]=date_weight;weighted+=1;field_weighted.append(i)
        field_audit.append({'fact_index':fact_index,'field':field,'level':level,'masked_indices':field_masked,'weighted_indices':field_weighted})
    if ids[-1]!=labels[-1] or labels[-1]!=eos or weights[-1]!=1.:raise ValueError('EOS label was masked')
    if any(l!=-100 or w!=0 for l,w in zip(labels[:n],weights[:n])):raise ValueError('Prompt labels leaked')
    if any((l==-100)!=(w==0) for l,w in zip(labels,weights)):raise ValueError('Weight/mask disagreement')
    return {'input_ids':ids,'labels':labels,'weights':weights,'answer_start':n,'masked_timestamp_tokens':masked,
            'weighted_timestamp_tokens':weighted,'supervised_weight':sum(weights),'field_audit':field_audit}

def suffix_loss(model,row):
    import torch
    import torch.nn.functional as F
    n=row['answer_start'];a=len(row['labels'])-n
    x=torch.tensor([row['input_ids']],device='cuda');labels=torch.tensor([row['labels'][n:]],device='cuda')
    weights=torch.tensor([row['weights'][n:]],device='cuda',dtype=torch.float32)
    logits=model(input_ids=x,attention_mask=torch.ones_like(x),use_cache=False,logits_to_keep=a+1).logits[:,:-1]
    if logits.shape[:2]!=labels.shape:raise ValueError('Logit/label shift mismatch')
    total=logits.new_zeros((),dtype=torch.float32)
    for i in range(0,a,64):
        ce=F.cross_entropy(logits[:,i:i+64].float().reshape(-1,logits.shape[-1]),labels[:,i:i+64].reshape(-1),ignore_index=-100,reduction='none')
        total=total+(ce*weights[:,i:i+64].reshape(-1)).sum()
    return total/weights.sum()

def loss_selftest(model,tokenizer):
    import torch
    import torch.nn.functional as F
    row={'call_id':'loss-selftest','prompt':[{'role':'system','content':'Верни JSON.'},{'role':'user','content':'Проверка.'}],
        'completion':[{'role':'assistant','content':'{"facts":[{"valid_from":"2026-11-15T13:45:00","valid_to":"2026-11-20T00:00:00"}]}'}],
        'date_supervision':[{'fact_index':0,'field':'valid_from','level':'day'}]}
    encoded=encode_record(row,tokenizer);x=torch.tensor([encoded['input_ids']],device='cuda')
    y=torch.tensor([encoded['labels']],device='cuda');w=torch.tensor([encoded['weights']],device='cuda')
    model.eval()
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):
        out=model(input_ids=x,attention_mask=torch.ones_like(x),labels=y,use_cache=False)
        plain={**encoded,'weights':[float(l!=-100) for l in encoded['labels']]}
        truncated=suffix_loss(model,plain)
        torch.testing.assert_close(out.loss.float(),truncated.float(),rtol=.003,atol=.003)
        ce=F.cross_entropy(out.logits[:,:-1].float().reshape(-1,out.logits.shape[-1]),y[:,1:].reshape(-1),ignore_index=-100,reduction='none')
        reference=(ce*w[:,1:].reshape(-1)).sum()/w[:,1:].sum()
        actual=suffix_loss(model,encoded)
        torch.testing.assert_close(reference.float(),actual.float(),rtol=.003,atol=.003)
    return {'passed':True,'hf_loss_equivalence':True,'full_logits_weighted_reference_equivalence':True,
        'masked_tokens':encoded['masked_timestamp_tokens'],'weighted_tokens':encoded['weighted_timestamp_tokens']}
