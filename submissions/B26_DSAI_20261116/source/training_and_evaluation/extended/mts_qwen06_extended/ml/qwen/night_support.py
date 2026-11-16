"""Conservative evidence/date supervision for the private overnight experiment.

Unverified gold dates are masked, not corrected or declared true.
Source roles and indices are never repaired. No inference normalizer is enabled.
"""
from collections import Counter, defaultdict
from datetime import datetime, timedelta
import json
import re

from ml.qwen.bridge import strict_json

STATE = {'service_state', 'tariff', 'ticket_status'}
STOP = {'этот','того','было','была','были','котор','тариф','клиент','опера','сегод','вчера','сейча','работ','услуг'}
NUMBERS = {'один':1,'одну':1,'два':2,'две':2,'три':3,'четыре':4,'пять':5,'шесть':6,
           'семь':7,'восемь':8,'девять':9,'десять':10,'одиннадцать':11,'двенадцать':12,
           'тринадцать':13,'четырнадцать':14}
DAY = re.compile(r'\b(позавчера|вчера|послезавтра|завтра|сегодня|сейчас)\b', re.I)
REL = re.compile(r'\b(?:(через)\s+)?(\d{1,2}|один|одну|два|две|три|четыре|пять|шесть|семь|восемь|девять|десять|одиннадцать|двенадцать|тринадцать|четырнадцать)\s+(день|дня|дней|сутки|суток|неделю|недели|недель)(?:\s+(назад))?\b',re.I)
DATE_VALUE = re.compile(r'(?<!\\)"(valid_from|valid_to)":("[^"]*"|null)')


def words(text):
    return {w[:5] for w in re.findall(r'[а-яёa-z0-9]+',text.casefold().replace('ё','е'))
            if len(w)>=4 and w[:5] not in STOP}


def overlaps_value(value,text):
    v=words(value)
    return bool(v and len(v & words(text)) >= min(2,len(v)))


def date_anchors(text, started_at):
    """Only day-relative explicit anchors; no weekday/year or clock guesses."""
    base=datetime.fromisoformat(started_at);midnight=base.replace(hour=0,minute=0,second=0,microsecond=0)
    found=[]
    delta={'позавчера':-2,'вчера':-1,'послезавтра':2,'завтра':1,'сегодня':0}
    for m in DAY.finditer(text):
        word=m.group().casefold()
        dt=base if word=='сейчас' else midnight+timedelta(days=delta[word])
        found.append({'expression':m.group(),'start':m.start(),'end':m.end(),'datetime':dt.isoformat(timespec='seconds'),'day':word})
    for m in REL.finditer(text):
        if bool(m.group(1)) == bool(m.group(4)):continue
        n=int(m.group(2)) if m.group(2).isdigit() else NUMBERS[m.group(2).casefold()]
        if not 0<n<=31:continue
        n*=7 if m.group(3).casefold().startswith('недел') else 1
        dt=midnight+timedelta(days=n if m.group(1) else -n)
        found.append({'expression':m.group(),'start':m.start(),'end':m.end(),'datetime':dt.isoformat(timespec='seconds'),'day':'relative'})
    return sorted(found,key=lambda x:x['start'])


def bound_date(fact, quote, started_at):
    """Require one literal time anchor and evidence overlapping the fact value.

    State changes need an explicit onset/end construction. Completed/open state
    distinction is not guessed from arbitrary mentions of yesterday elsewhere.
    """
    if not quote or len(quote)>500 or not overlaps_value(fact['value'],quote):return None
    # Do not turn questions, conditional plans or denied dates into supervision.
    if '?' in quote or re.search(r'\b(?:если|возможно|наверное|кажется)\b',quote,re.I):return None
    if re.search(r'\bне\s+(?:поза)?вчера|\bне\s+(?:после)?завтра|\bне\s+сегодня',quote,re.I):return None
    anchors=date_anchors(quote,started_at)
    if len(anchors)!=1:return None
    a=anchors[0];prefix=quote[max(0,a['start']-24):a['start']].casefold()
    attr=fact['attribute']
    if attr in STATE:
        if re.search(r'\bдо\s*$',prefix):field='valid_to'
        elif re.search(r'\b(?:с|со)\s*$',prefix):field='valid_from'
        elif a['day'] in ('сегодня','сейчас') and fact.get('valid_to') is None:field='valid_from'
        else:return None
    elif attr=='action':
        # A future instruction is not a completed action.
        if datetime.fromisoformat(a['datetime']).date()>datetime.fromisoformat(started_at).date():return None
        field='valid_from'
    elif attr=='agreement':
        if datetime.fromisoformat(a['datetime']).date()<datetime.fromisoformat(started_at).date():return None
        field='valid_from'
    else:return None
    # An explicit clock in the source is unsupported here; do not erase its hour.
    if re.search(r'\b(?:после|до|к|в)\s+\d{1,2}(?::\d{2}|\s+час)|\b(?:к обеду|утром|вечером|ночью|часов|часа)\b',quote,re.I):return None
    return {**a,'field':field}


def supervised_target(call):
    """Retain all original facts; alter only optional literal evidence fields."""
    source_counts=Counter(f['source']['turn_idx'] for f in call['facts'])
    turns={t['idx']:t for t in call['turns']}
    result=[];proof=[];masked=[]
    for i,f in enumerate(call['facts']):
        out={k:f[k] for k in ('attribute','value','polarity','speaker','valid_from','valid_to')}
        out={'object':{k:f['object'][k] for k in ('name','type')},**out,
             'source':{'turn_idx':f['source']['turn_idx'],'span':None,'quote':None}}
        text=turns[f['source']['turn_idx']]['text']
        anchor=bound_date(f,text,call['started_at']) if source_counts[f['source']['turn_idx']]==1 else None
        accepted=anchor and f.get(anchor['field'])==anchor['datetime']
        if accepted:
            out['source']['quote']=text
            proof.append({'fact_index':i,'turn_idx':f['source']['turn_idx'],**anchor,
                          'evidence':text,'checks':['one_fact_in_source_turn','one_time_anchor','lexical_value_overlap','exact_gold_datetime_match']})
        for field in ('valid_from','valid_to'):
            if f[field] is not None and not (accepted and anchor['field']==field):
                masked.append({'fact_index':i,'field':field,'reason':'No uniquely bound explicit source anchor verified; original gold retained but excluded from loss'})
        result.append(out)
    # Official single-valued state attributes: an explicitly dated next state can
    # support the previous interval end. Never change the original end date.
    # This is training evidence only; no gold facts are consulted at inference.
    directly_dated=[p for p in proof if p['field']=='valid_from']
    for i,f in enumerate(call['facts']):
        if f['attribute'] not in STATE or f['polarity']!=1 or f['valid_to'] is None:continue
        possible=[]
        for p in directly_dated:
            g=call['facts'][p['fact_index']]
            if (p['fact_index']!=i and g['polarity']==1 and g['attribute']==f['attribute']
                    and g['object']['id']==f['object']['id'] and g['value']!=f['value']
                    and f['valid_from']<g['valid_from'] and f['valid_to']==g['valid_from']):
                possible.append(p)
        if len(possible)!=1:continue
        p=possible[0]
        if any(x['fact_index']==i and x['field']=='valid_to' for x in proof):continue
        proof.append({'fact_index':i,'turn_idx':p['turn_idx'],'field':'valid_to','datetime':f['valid_to'],
            'expression':p['expression'],'evidence':p['evidence'],'next_fact_index':p['fact_index'],
            'checks':['same_gold_object_id_and_single_valued_attribute','different_positive_state',
                      'unique_next_state_with_direct_explicit_date','original_end_equals_next_onset']})
    approved={(p['fact_index'],p['field']) for p in proof}
    masked=[m for m in masked if (m['fact_index'],m['field']) not in approved]
    return {'facts':result},proof,masked


def encoded_with_mask(row,tokenizer,max_length=None):
    from ml.qwen.train import load_module
    from pathlib import Path
    # Reuse the exact checked chat-boundary/EOS formatter.
    mod=load_module('night_original_tokenizer',Path(__file__).resolve().parents[2]/'tokenize_sft.py')
    encoded=mod.encode_record(row,tokenizer,max_length=max_length)
    prefix=tokenizer.apply_chat_template(row['prompt'],tokenize=False,add_generation_prompt=True,enable_thinking=False)
    full=tokenizer.apply_chat_template(row['prompt']+row['completion'],tokenize=False,add_generation_prompt=False,enable_thinking=False)
    offsets=tokenizer(full,add_special_tokens=False,return_offsets_mapping=True)['offset_mapping']
    answer=row['completion'][0]['content'];fields=list(DATE_VALUE.finditer(answer))
    facts=json.loads(answer)['facts'];assert len(fields)==2*len(facts)
    spans=[]
    for item in row.get('masked_date_fields',[]):
        m=fields[2*item['fact_index']+(item['field']=='valid_to')]
        assert m.group(1)==item['field']
        spans.append((len(prefix)+m.start(2),len(prefix)+m.end(2)))
    before=sum(v!=-100 for v in encoded['labels'])
    for i,(a,b) in enumerate(offsets[:len(encoded['labels'])]):
        if any(a<hi and b>lo for lo,hi in spans):encoded['labels'][i]=-100
    assert encoded['labels'][-1]==tokenizer.eos_token_id
    assert sum(v!=-100 for v in encoded['labels'])>0
    encoded['masked_date_tokens']=before-sum(v!=-100 for v in encoded['labels'])
    encoded['answer_start']=len(tokenizer.encode(prefix,add_special_tokens=False))
    return encoded


def suffix_loss(model,encoded,*,device='cuda'):
    import torch
    import torch.nn.functional as F
    labels=encoded['labels'];n=encoded.get('answer_start',next(i for i,v in enumerate(labels) if v!=-100))
    assert n>0 and all(v==-100 for v in labels[:n])
    a=len(labels)-n
    ids=torch.tensor([encoded['input_ids']],device=device)
    targets=torch.tensor([labels[n:]],device=device)
    logits=model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False,logits_to_keep=a+1).logits[:,:-1]
    assert logits.shape[:2]==targets.shape
    total=logits.new_zeros((),dtype=torch.float32)
    for start in range(0,a,64):
        total=total+F.cross_entropy(logits[:,start:start+64].float().reshape(-1,logits.shape[-1]),targets[:,start:start+64].reshape(-1),ignore_index=-100,reduction='sum')
    return total/(targets!=-100).sum()


def check_loss_equivalence(model,tokenizer):
    import torch
    from ml.qwen.train import check_loss_equivalence as original_check,autocast_for
    result=original_check(model,tokenizer)
    prefix=tokenizer.apply_chat_template([{'role':'user','content':'Верни JSON.'}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
    p=tokenizer.encode(prefix,add_special_tokens=False)
    a=tokenizer.encode('{"facts":[],"probe":"2026-11-15"}',add_special_tokens=False)+[tokenizer.eos_token_id]
    row={'input_ids':p+a,'labels':[-100]*len(p)+a,'answer_start':len(p)}
    # Exercise a non-contiguous assistant mask, not just the old contiguous suffix.
    row['labels'][len(p)+3:len(p)+6]=[-100]*3
    x=torch.tensor([row['input_ids']],device='cuda');y=torch.tensor([row['labels']],device='cuda')
    model.eval()
    with torch.no_grad(),autocast_for(model):
        ref=model(input_ids=x,attention_mask=torch.ones_like(x),labels=y,use_cache=False).loss
        actual=suffix_loss(model,row)
    torch.testing.assert_close(ref.float(),actual.float(),rtol=.003,atol=.003)
    result['non_contiguous_mask']={'passed':True,'reference_loss':float(ref),'suffix_loss':float(actual)}
    return result

