"""Source-bound calendar arithmetic; abstain on ambiguous dates and onset claims.

No gold access. Names, values, speakers, source indices, and input text are immutable.
"""
from copy import deepcopy
from datetime import datetime,timedelta
import re
from ml.qwen.date_anchors_v21 import anchors, lexical

STATE={'service_state','tariff','ticket_status'}
UNCERTAIN=re.compile(r'\b(?:если|возможно|наверное|кажется|не\s+(?:вчера|завтра|через|позавчера)|то\s+ли)\b',re.I)
EXPLICIT_CLOCK=re.compile(r'\b\d{1,2}:\d{2}\b|\b(?:утром|вечером|ночью|обед\w*|час\w*)\b|\b(?:после|до|к|в)\s+(?:двух|тр[её]х|четыр[её]х|пяти|шести|семи|восьми|девяти|десяти|\d{1,2})\b',re.I)
ACCEPT=re.compile(r'\b(?:да|подходит|подтверждаю|согласна|согласен|договорились)\b',re.I)
REJECT=re.compile(r'\b(?:нет|если|но|только|вместо|позже|раньше|не|отказываюсь)\b',re.I)
ONSET=re.compile(r'перестал|пропал|сломал|сменил|изменил|начал|подключил|появил|стало?\s|выполн|закры|принят|зарегистр|назнач|перев[её]л|переш[её]л',re.I)
DONE_ACTION=re.compile(r'делал|сделал|перезагруз|перезапус|проверил|отправил|оформил|создал|настроил|сбросил|подключил|отключил|заменил|синхронизац',re.I)
TIME_ASR={'вчира':'вчера','зафтра':'завтра','севодня':'сегодня','сигодня':'сегодня','позавчира':'позавчера','щерез':'через','назат':'назад'}
assert all(len(a)==len(b) for a,b in TIME_ASR.items())
def normalized_time_words(text):
    return re.sub(r'\b(?:'+ '|'.join(TIME_ASR)+r')\b',lambda m:TIME_ASR[m.group().lower()],text,flags=re.I)

def calendar_anchors(text,started_at):
    original_text=text;text=normalized_time_words(text)
    found=anchors(text,started_at)
    day=datetime.fromisoformat(started_at).replace(hour=0,minute=0,second=0,microsecond=0)
    for m in re.finditer(r'\b(?:(через)\s+)?(день|сутки|неделю)(?:\s+(назад))?\b',text,re.I):
        if bool(m[1])==bool(m[3]):continue
        if any(m.start()<a['span'][1] and a['span'][0]<m.end() for a in found):continue
        dt=day+timedelta(days=(7 if m[2].lower()=='неделю' else 1)*(1 if m[1] else -1))
        found.append({'expression':m.group(),'span':[m.start(),m.end()],'datetime':dt.isoformat(timespec='seconds'),'kind':'relative_days'})
    patterns=[(r'\b(20\d{2})-(\d{2})-(\d{2})\b','ymd'),(r'\b(\d{1,2})[./](\d{1,2})[./](20\d{2})\b','dmy')]
    for pattern,order in patterns:
        for m in re.finditer(pattern,text):
            if re.search(r'(?:заявка|номер|версия)\s*[№#:]?\s*$',text[max(0,m.start()-20):m.start()],re.I):continue
            values=list(map(int,m.groups()));year,month,d=values if order=='ymd' else values[::-1]
            try:dt=day.replace(year=year,month=month,day=d)
            except ValueError:continue
            if not any(m.start()<a['span'][1] and a['span'][0]<m.end() for a in found):
                found.append({'expression':m.group(),'span':[m.start(),m.end()],'datetime':dt.isoformat(timespec='seconds'),'kind':'explicit_date'})
    return [{**a,'expression':original_text[a['span'][0]:a['span'][1]]} for a in sorted(found,key=lambda a:a['span'])]

def event_kinds(text):
    patterns={'visit':r'выезд|приезд|приед|прид[её]т|приход\w*\s+(?:мастер|специалист)',
              'callback':r'перезвон|перезв[ао]н|созвон|обратн\w*\s+звон|набер[её]т|позвон',
              'sms':r'\bсмс\b|\bsms\b',
              'refund':r'возврат|вернут|верн[её]м|перерасч',
              'reboot':r'перезагруз|перезапус|отключ\w*[- ]включ',
              'settings':r'проверк\w*\s+настро|настройк\w*\s+(?:дат|времен)'}
    return {k for k,p in patterns.items() if re.search(p,text,re.I)}

def status_kinds(text):
    patterns={'received':r'принят|зарегистр|создан', 'working':r'в\s+р[ао]бот',
              'done':r'выполн|закры|заверш', 'scheduled':r'назнач\w*\s+выезд'}
    return {k for k,p in patterns.items() if re.search(p,text,re.I)}

def linked(fact,text):
    value=fact.get('value','')
    if lexical(value,text):return True
    if fact.get('attribute') in {'action','agreement'}:
        kinds=event_kinds(value)
        return len(kinds)==1 and bool(kinds & event_kinds(text))
    if fact.get('attribute')=='ticket_status':
        kinds=status_kinds(value)
        return len(kinds)==1 and bool(kinds & status_kinds(text))
    return False

def explicit_acceptance(fact,call):
    """Confirm one event proposal and a nearby explicit client acceptance, by turn positions."""
    if fact.get('polarity')!=1:return None
    turns=call['turns'];positions={t['idx']:i for i,t in enumerate(turns)}
    pos=positions.get(fact.get('source',{}).get('turn_idx'))
    if pos is None:return None
    src=turns[pos];pairs=[]
    for a in range(max(0,pos-5),pos+1):
        proposal=turns[a]
        if proposal['speaker']!='operator' or not linked(fact,proposal['text']) or UNCERTAIN.search(normalized_time_words(proposal['text'])):continue
        for b in range(a+1,min(len(turns),a+4)):
            client=turns[b]
            if client['speaker']!='client' or not ACCEPT.search(client['text']) or REJECT.search(client['text']):continue
            is_original=a==pos or b==pos
            is_summary=(src['speaker']=='operator' and b<pos<=b+3 and linked(fact,src['text'])
                and not UNCERTAIN.search(src['text']) and not REJECT.search(src['text']))
            if not(is_original or is_summary):continue
            if any(t['speaker']=='operator' and event_kinds(t['text']) for t in turns[a+1:b]):continue
            pairs.append({'proposal_turn_idx':proposal['idx'],'acceptance_turn_idx':client['idx'],'source_is_later_summary':is_summary})
    return pairs[0] if len(pairs)==1 else None

def source_proofs(fact,call):
    turns={t['idx']:t for t in call['turns']};src=fact.get('source',{}).get('turn_idx')
    if src not in turns or fact.get('speaker')!=turns[src]['speaker']:return []
    attr=fact.get('attribute');sources=[turns[src]];acceptance=None
    if attr=='agreement':
        acceptance=explicit_acceptance(fact,call)
        if not acceptance:return []
        sources=[turns[acceptance['proposal_turn_idx']]]
    result=[]
    for turn in sources:
        for clause in re.split(r'(?<=[.!?;])\s+',turn['text']):
            if UNCERTAIN.search(normalized_time_words(clause)) or not linked(fact,clause):continue
            if fact.get('polarity')!=1 or re.search(r'\bне\s+(?:делал|сделал|перезагруз|проверил|отправил|оформил|сменил|перестал|назнач|подключил)',clause,re.I):continue
            found=[a for a in calendar_anchors(clause,call['started_at']) if a['expression'].lower()!='сейчас']
            if re.search(r'месяц|прошл\w*\s+недел|следующ\w*\s+недел',clause,re.I):continue
            # A period such as last week does not prove an exact day; never select a day in it.
            if len(found)!=1:continue
            a=found[0]
            clocks=list(re.finditer(r'\b(?:в|с)\s+([01]?\d|2[0-3]):([0-5]\d)\b',clause,re.I))
            if len(clocks)>1:continue
            precision='day';clock_note=None
            exact_clock=bool(clocks) and not re.match(r'\s*(?:утра|дня|вечера|ночи)\b',clause[clocks[0].end():],re.I)
            if clocks and re.search(r'примерно|около',clause[max(0,clocks[0].start()-18):clocks[0].start()],re.I):exact_clock=False
            if exact_clock:
                dt=datetime.fromisoformat(a['datetime']).replace(hour=int(clocks[0][1]),minute=int(clocks[0][2]))
                a={**a,'datetime':dt.isoformat(timespec='seconds')};precision='minute'
            elif EXPLICIT_CLOCK.search(clause):clock_note='Only day resolved; a vague time bound is not an exact hour. 00:00 denotes day precision.'
            if a['kind'].startswith('weekday'):
                if attr!='agreement':continue  # Bare weekday may be historical or future.
                if datetime.fromisoformat(a['datetime']).weekday()==datetime.fromisoformat(call['started_at']).weekday():continue
            if attr in STATE:
                if fact.get('valid_to') is not None and attr!='ticket_status':continue
                if re.search(r'ещ[её]\s+(?:вчера|позавчера)|был[аои]?\s+в\s+(?:статусе|работе)',clause,re.I):continue
                if not ONSET.search(clause):continue  # "worked yesterday" does not identify onset.
                if a['datetime'][:10]>call['started_at'][:10]:continue
            elif attr=='action':
                if not DONE_ACTION.search(clause):continue
                if a['datetime'][:10]>call['started_at'][:10]:continue
                if re.search(r'начал\w*|заметил\w*|обнаружил\w*|появил\w*|перестал\w*',clause,re.I):continue
            elif attr=='agreement':
                if a['datetime'][:10]<call['started_at'][:10]:continue
                if len(event_kinds(clause))!=1:continue  # Visit date cannot become a callback deadline.
            else:continue
            result.append({'field':'valid_from','datetime':a['datetime'],'expression':a['expression'],
                'source_turn_idx':turn['idx'],'source_clause':clause,'acceptance':acceptance,
                'precision':precision,'clock_note':clock_note,'rule':'v23_unique_source_bound_calendar_anchor'})
    dates={p['datetime'] for p in result}
    return result[:1] if len(dates)==1 else []

def normalize_fact(fact,call):
    value=deepcopy(fact);changes=[]
    try:proofs=source_proofs(value,call)
    except (KeyError,TypeError,ValueError,OverflowError):return value,changes
    if not proofs:return value,changes
    proof=proofs[0];new=proof['datetime'];old=value.get('valid_from')
    if old==new:return value,changes
    end=value.get('valid_to')
    if end is not None:
        try:
            if datetime.fromisoformat(end)<datetime.fromisoformat(new):return value,changes
        except (ValueError,TypeError):return value,changes
    value['valid_from']=new;changes.append({'field':'valid_from','before':old,'after':new,'proof':proof})
    return value,changes

def linked_interval_updates(candidates,originals,changes):
    """Preserve an already generated shared boundary when a proved new onset moves.

    Restricted to named client tariffs and numbered tickets. Unknown/generic devices
    and unrelated objects are never merged. Only fully validated candidates participate.
    """
    groups={};updates=[]
    for index,candidate in candidates:
        f=candidate.fact
        if f.polarity!=1 or not f.object.name.strip():continue
        if not (f.attribute=='tariff' and f.object.type=='client' or f.attribute=='ticket_status' and f.object.type=='ticket' and re.search(r'\d{3,}',f.object.name)):continue
        key=(f.object.type,f.object.name.strip().lower().replace('ё','е'),f.attribute)
        groups.setdefault(key,[]).append((index,candidate))
    for group in groups.values():
        if len(group)!=2:continue
        closed=[p for p in group if p[1].fact.valid_to is not None];opened=[p for p in group if p[1].fact.valid_to is None]
        if len(closed)!=1 or len(opened)!=1:continue
        oi,old=closed[0];ni,new=opened[0]
        if old.fact.value==new.fact.value:continue
        if originals[oi].get('valid_to')!=originals[ni].get('valid_from'):continue
        proof=[c for c in changes if c['fact_index']==ni and c['field']=='valid_from']
        if len(proof)!=1:continue
        try:
            if old.fact.valid_from>new.fact.valid_from or old.fact.valid_to==new.fact.valid_from:continue
        except TypeError:continue
        updates.append({'fact_index':oi,'field':'valid_to','before':old.fact.valid_to.isoformat(),
            'after':new.fact.valid_from.isoformat(),'proof':{'rule':'v22_preserve_existing_shared_state_boundary',
            'new_state_fact_index':ni,'onset_evidence':proof[0]['proof']}})
    return updates
