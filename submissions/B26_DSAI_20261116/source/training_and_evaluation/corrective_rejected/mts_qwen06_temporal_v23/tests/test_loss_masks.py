import json
import pytest
from temporal_loss import encode_record

class CharTokenizer:
    eos_token_id=2
    def __init__(self,merge=False):self.tokens={'\ue000':2};self.inverse={2:'\ue000'};self.merge=merge
    def apply_chat_template(self,messages,*,add_generation_prompt=False,**kwargs):
        text=''.join(m['role']+':'+m['content']+('\ue000\n' if m['role']=='assistant' else '\n') for m in messages)
        return text+('assistant:' if add_generation_prompt else '')
    def __call__(self,text,**kwargs):
        ids=[];offsets=[];i=0
        while i<len(text):
            token=text[i:i+4] if self.merge and text.startswith('5T13',i) else text[i]
            if token not in self.tokens:
                idx=len(self.tokens)+10;self.tokens[token]=idx;self.inverse[idx]=token
            ids.append(self.tokens[token]);offsets.append((i,i+len(token)));i+=len(token)
        return {'input_ids':ids,'offset_mapping':offsets}
    def encode(self,text,**kwargs):return self(text)['input_ids']
    def decode(self,ids,**kwargs):return ''.join(self.inverse[i] for i in ids)

def sample(level='day'):
    return {'call_id':'unit','prompt':[{'role':'system','content':'Extract JSON.'},{'role':'user','content':'Untrusted input.'}],
        'completion':[{'role':'assistant','content':'{"facts":[{"valid_from":"2026-11-15T13:45:00","valid_to":"2026-11-20T00:00:00"}]}'}],
        'date_supervision':[{'fact_index':0,'field':'valid_from','level':level}]}

def test_only_verified_day_is_weighted_and_unknown_clock_is_masked():
    r=encode_record(sample(),CharTokenizer());a,b=r['field_audit']
    assert len(a['weighted_indices'])==10 and len(a['masked_indices'])==9
    assert len(b['weighted_indices'])==0 and len(b['masked_indices'])==19
    assert all(r['labels'][i]==-100 and r['weights'][i]==0 for i in a['masked_indices']+b['masked_indices'])
    assert all(r['weights'][i]==2 for i in a['weighted_indices'])

def test_full_source_proof_supervises_timestamp_but_not_another_unknown_field():
    r=encode_record(sample('full'),CharTokenizer());assert len(r['field_audit'][0]['weighted_indices'])==19
    assert len(r['field_audit'][1]['masked_indices'])==19

def test_prompt_has_zero_loss_and_answer_eos_is_kept():
    r=encode_record(sample(),CharTokenizer())
    assert all(y==-100 for y in r['labels'][:r['answer_start']])
    assert r['labels'][-1]==2 and r['weights'][-1]==1

def test_bpe_token_crossing_day_clock_boundary_is_fully_masked():
    r=encode_record(sample(),CharTokenizer(merge=True));a=r['field_audit'][0]
    assert len(a['weighted_indices'])==9 and r['masked_timestamp_tokens']>0

def test_no_truncation_or_unknown_policy_silently_accepted():
    with pytest.raises(ValueError,match='No silent truncation'):encode_record(sample(),CharTokenizer(),max_length=10)
    row=sample();row['date_supervision'][0]['fact_index']=99
    with pytest.raises(ValueError):encode_record(row,CharTokenizer())

def test_empty_facts_still_have_supervised_structure_and_eos():
    row=sample();row['completion'][0]['content']='{"facts":[]}';row['date_supervision']=[]
    r=encode_record(row,CharTokenizer());assert r['supervised_weight']>0 and not r['field_audit'] and r['labels'][-1]==2
