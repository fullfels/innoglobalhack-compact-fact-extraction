"""Data checks independent of model fitting; synthetic dates re-read from literal text."""
from pathlib import Path
from datetime import datetime,timedelta
from collections import Counter,defaultdict
import argparse,hashlib,json,re
from ml.qwen.bridge import model_input,parse_candidates
from src.contracts.models import CallInput

WORDS=dict(zip('один два три четыре пять шесть семь восемь девять десять одиннадцать двенадцать тринадцать четырнадцать пятнадцать шестнадцать семнадцать восемнадцать девятнадцать двадцать'.split(),range(1,21)));WORDS['две']=2
def canonical_time(text):
    aliases={'вчира':'вчера','зафтра':'завтра','севодня':'сегодня','сигодня':'сегодня','позавчира':'позавчера','щерез':'через','назат':'назад'}
    return re.sub(r'\b(?:'+ '|'.join(aliases)+r')\b',lambda m:aliases[m[0].casefold()],text,flags=re.I)
def rows(p):return [json.loads(s) for s in Path(p).read_text(encoding='utf-8').splitlines() if s.strip()]
def account(c):return (c.get('series')or{}).get('account_id',c['call_id'].split(':')[0])
def calendar_from_expression(expression,start):
    s=expression.lower().strip();day=datetime.fromisoformat(start).replace(hour=0,minute=0,second=0,microsecond=0)
    if re.fullmatch(r'\d{2}\.\d{2}\.\d{4}',s):return datetime.strptime(s,'%d.%m.%Y').date().isoformat()
    if s in {'вчера','позавчера','сегодня','завтра'}:return (day+timedelta(days={'вчера':-1,'позавчера':-2,'сегодня':0,'завтра':1}[s])).date().isoformat()
    if 'недел' in s:
        number=2 if 'две' in s else 1;direction=1 if s.startswith('через ') else -1
        return (day+timedelta(days=number*7*direction)).date().isoformat()
    m=re.fullmatch(r'(?:через )?(\d+|'+ '|'.join(WORDS)+r') (?:день|дня|дней)(?: назад)?',s)
    if not m:raise ValueError('Unverified synthetic date expression: '+s)
    n=int(m[1]) if m[1].isdigit() else WORDS[m[1]]
    if s.startswith('через '):return (day+timedelta(days=n)).date().isoformat()
    if s.endswith(' назад'):return (day-timedelta(days=n)).date().isoformat()
    raise ValueError('No date direction')

def verify(root,official_source=None):
    root=Path(root);train=rows(root/'data/train1024.jsonl');sft=rows(root/'sft/train.jsonl');dev=rows(root/'data/selection32.jsonl');audit=rows(root/'data/audit24.jsonl')
    assert len(train)==len(sft)==1024 and len(dev)==32 and len(audit)==24
    assert len({r['call_id'] for r in train})==1024
    groups=[{account(c) for c in data} for data in [train,dev,audit,rows(root/'evaluation/val.jsonl')]]
    assert all(not(groups[i]&groups[j]) for i in range(4) for j in range(i))
    original={c['call_id']:c for c in rows(official_source)} if official_source else None
    origin_counts=Counter();families=defaultdict(list);fields=Counter();real_unchanged=0
    for call,row in zip(train,sft):
        assert call['call_id']==row['call_id'];origin=row['origin'];origin_counts[origin['origin']]+=1
        inp=CallInput.model_validate({k:call[k] for k in ['call_id','started_at','turns']});user=json.loads(row['prompt'][1]['content'])
        assert user==model_input(inp) and set(user)=={'call_id','started_at','turns'}
        answer=json.loads(row['completion'][0]['content']);parse_candidates(row['completion'][0]['content'],inp)
        assert len(answer['facts'])==len(call['facts'])
        for f,g in zip(answer['facts'],call['facts']):
            assert all(f[k]==g[k] for k in ['attribute','value','polarity','speaker','valid_from','valid_to'])
            assert f['source']['turn_idx']==g['source']['turn_idx'] and f['source']['quote'] is None and f['source']['span'] is None
            assert f['object']=={k:v for k,v in g['object'].items() if k!='id'}
        policy={(p['fact_index'],p['field']):p for p in row['date_supervision']}
        assert len(policy)==len(row['date_supervision'])
        fields.update(p['level'] for p in row['date_supervision'])
        if origin['origin']=='official_train_v3':
            if original is not None:
                assert call==original[call['call_id']];real_unchanged+=1
            assert hashlib.sha256(json.dumps(call['turns'],ensure_ascii=False,sort_keys=True).encode()).hexdigest()==origin['original_input_sha256']
        else:
            families[origin['family_id']].append((call,row))
            turnmap={t['idx']:t['text'] for t in call['turns']}
            for (i,field),p in policy.items():
                fact=call['facts'][i];proof=p['proof'];value=fact[field]
                if field=='valid_from':
                    expr=proof['expression']
                    if expr=='default_call_time':assert fact['polarity']==-1 and value==call['started_at'];continue
                    text=turnmap[proof['source_turn_idx']]
                    assert canonical_time(expr).casefold() in canonical_time(text).casefold(),(call['call_id'],expr,text)
                    assert value[:10]==calendar_from_expression(expr,call['started_at'])
                    if proof.get('explicit_clock'):
                        assert 'в '+proof['explicit_clock'] in text and value[11:16]==proof['explicit_clock']
                else:
                    siblings=[j for j,g in enumerate(call['facts']) if j!=i and g['object']['id']==fact['object']['id'] and g['attribute']==fact['attribute'] and g['valid_from']==value]
                    assert len(siblings)==1 and (siblings[0],'valid_from') in policy
    assert origin_counts=={'official_train_v3':768,'synthetic':256} and len(families)==128
    for family,pair in families.items():
        assert len(pair)==2
        pair=sorted(pair,key=lambda p:p[1]['origin']['variant']);a,b=pair[0][0],pair[1][0]
        assert a['facts']==b['facts'] and a['started_at']==b['started_at']
        for x,y in zip(a['turns'],b['turns']):
            assert x['idx']==y['idx'] and x['speaker']==y['speaker']
            assert re.findall(r'\d+',x['text'])==re.findall(r'\d+',y['text'])
            protected=r'\b(?:не|нет|без|если|да|через|назад|вчера|позавчера|сегодня|завтра|послезавтра|'+ '|'.join(WORDS)+r')\b'
            assert Counter(re.findall(protected,canonical_time(x['text']).casefold()))==Counter(re.findall(protected,canonical_time(y['text']).casefold()))
    result={'passed':True,'training_rows':1024,'real_rows':768,'synthetic_rows':256,'selection_rows':32,'audit_rows':24,
        'synthetic_families_checked':128,'date_supervision_counts':dict(fields),'source_inputs_and_dates_unchanged_verified':real_unchanged,
        'native_tokenizer_and_cuda_loss_checks_still_required':True,'human_verified_all_semantics':False}
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--official-source',type=Path);a=p.parse_args()
    print(json.dumps(verify(a.root,a.official_source),ensure_ascii=False,indent=2))
