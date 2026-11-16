"""Evidence-gated calendar fixes and explicit uncertainty; no gold input at inference."""
from copy import deepcopy
from datetime import datetime
import json,re
from ml.qwen.date_anchors_v21 import anchors,lexical

STATE={'service_state','tariff','ticket_status'}
TIME_CUE=re.compile(r'\b(?:вчера|вкчера|позавчера|завтра|зафтра|послезавтра|сегодня|сег[ао]дня|через|щерез|назад|назат|раньше|недел\w*|месяц\w*|прошл\w*|позапрошл\w*|следующ\w*|год\w*|полгода|дн[еяийу]\w*|сут\w*|утр\w*|вечер\w*|ноч\w*|час\w*|понедель\w*|вторник\w*|сред[ауы]|четверг\w*|пятниц\w*|суббот\w*|воскресен\w*)\b|\d{1,4}[./:]\d{1,2}|\b(?:после|до|к)\s+(?:двух|тр[её]х|четыр[её]х|пяти|шести|семи|восьми|девяти|десяти|\d{1,2})\b',re.I)
EXPLICIT_CLOCK=re.compile(r'\d{1,2}:\d{2}|\b(?:утр\w*|вечер\w*|ноч\w*|обед\w*|час\w*)\b|\b(?:после|до|к|в)\s+(?:двух|тр[её]х|четыр[её]х|пяти|шести|семи|восьми|девяти|десяти|\d{1,2})\b',re.I)
ACCEPT=re.compile(r'\b(?:да|подходит|подтверждаю|согласна|согласен|договорились)\b',re.I)
DECLINE=re.compile(r'\b(?:не\s+подходит|не\s+соглас\w*|не\s+сейчас|отказываюсь|не\s+нужно|не\s+надо)\b',re.I)

def has_time_cue(call):return any(TIME_CUE.search(t['text']) or re.search(r'\b(?:январ\w*|феврал\w*|март\w*|апрел\w*|ма[йяе]|июн\w*|июл\w*|август\w*|сентябр\w*|октябр\w*|ноябр\w*|декабр\w*)\b',t['text'],re.I) for t in call['turns'])

def event_link(value,text,started_at):
    def strip(s):
        for a in reversed(anchors(s,started_at)):
            lo,hi=a['span'];s=s[:lo]+' '+s[hi:]
        return re.sub(r'\b(?:подтверд\w*|подходит|соглас\w*|клиент\w*|через|срок\w*|сегодня|завтра|вчера|дней|недел\w*)\b',' ',s,flags=re.I)
    return lexical(strip(value),strip(text))

def event_kinds(text):
    patterns={'visit':r'выезд|приезд|приед|прид[её]т','callback':r'перезвон|созвон|обратн\w*\s+звон|набер[её]т|позвон',
              'sms':r'\bсмс\b|\bsms\b|уведомлен|подтвержден\w*\s+заявк','refund':r'возврат|вернут|верн[её]м|перерасч'}
    return {k for k,p in patterns.items() if re.search(p,text,re.I)}

def acceptance(fact,call):
    src=fact.get('source',{}).get('turn_idx');turns={t['idx']:t for t in call['turns']}
    if src not in turns:return None
    source=turns[src]
    if source['speaker']=='operator':
        proposals=[source]
        clients=[t for t in call['turns'] if src<t['idx']<=src+2 and t['speaker']=='client']
    else:
        clients=[source]
        proposals=[t for t in call['turns'] if src-2<=t['idx']<src and t['speaker']=='operator']
    proposals=[t for t in proposals if event_link(fact['value'],t['text'],call['started_at'])]
    clients=[t for t in clients if ACCEPT.search(t['text']) and not DECLINE.search(t['text'])
        and not re.search(r'\b(?:нет|но|если|только|вместо|давайте|позже|раньше|не)\b',t['text'],re.I)]
    if len(proposals)!=1 or len(clients)!=1:return None
    return {'proposal_turn_idx':proposals[0]['idx'],'acceptance_turn_idx':clients[0]['idx']}

def bound_dates(fact,call):
    """Use only a uniquely linked clause, not arbitrary dates elsewhere in the call."""
    turns={t['idx']:t for t in call['turns']};src=fact['source']['turn_idx']
    if src not in turns:return []
    sources=[turns[src]];confirmed=None
    if fact['attribute']=='agreement':
        confirmed=acceptance(fact,call)
        if not confirmed:return []
        if confirmed['proposal_turn_idx']!=src:sources.append(turns[confirmed['proposal_turn_idx']])
    out=[]
    for t in sources:
        for clause in re.split(r'(?<=[.!?;])\s+',t['text']):
            if not event_link(fact['value'],clause,call['started_at']):continue
            if re.search(r'\b(?:если|возможно|наверное|кажется|не\s+вчера|не\s+завтра|не\s+через)\b',clause,re.I):continue
            if EXPLICIT_CLOCK.search(clause):continue # No guessed morning/hour.
            found=anchors(clause,call['started_at'])
            if len(found)!=1:continue
            a=found[0];prefix=clause[max(0,a['span'][0]-32):a['span'][0]]
            if a['kind'].startswith('weekday') and datetime.fromisoformat(a['datetime']).weekday()==datetime.fromisoformat(call['started_at']).weekday() and not re.search(r'прошл|следующ|ближайш',a['expression'],re.I):continue
            field='valid_from'
            if fact['attribute'] in STATE:
                # No automatic onset inference: a clause may mention both old and new state.
                continue
            elif fact['attribute']=='action':
                if a['datetime'][:10]>call['started_at'][:10]:continue
                # Do not date an action from a nearby symptom-onset statement.
                around=clause[max(0,a['span'][0]-40):a['span'][1]+45]
                if re.search(r'начал\w*|заметил\w*|обнаружил\w*|появил\w*|перестал\w*',around,re.I):continue
            elif fact['attribute']=='agreement':
                if a['datetime'][:10]<call['started_at'][:10]:continue
                if len(event_kinds(clause))>1:continue # Visit date is not automatically an SMS/callback deadline.
            else:continue
            out.append({'field':field,'datetime':a['datetime'],'expression':a['expression'],
                'source_turn_idx':t['idx'],'literal_clause':clause,'confirmation':confirmed,
                'rule':'unique_event_linked_relative_day_without_invented_clock'})
    unique={(p['field'],p['datetime']) for p in out}
    return out[:1] if len(unique)==1 else []

def prepare_target(call,covered_days):
    """Preserve every fact; repair only proved day anchors, mask unknown historical numerals.

    An official whole-call date bank is an uncertainty screen, not source evidence.
    Unsupported dates stay in the archived target but receive no timestamp-token loss.
    """
    facts=[];patches=[];proofs=[];masked=[]
    no_cue=not has_time_cue(call)
    for i,original in enumerate(call['facts']):
        f={k:deepcopy(original[k]) for k in ['object','attribute','value','polarity','speaker','valid_from','valid_to','source']}
        f['object'].pop('id',None);f['source'].update(quote=None,span=None)
        direct=bound_dates(f,call)
        conflicts=set()
        for p in direct:
            field=p['field'];new=p['datetime'];old=f[field]
            # A lexical calendar heuristic is NOT strong enough to relabel official gold.
            # Disagreement is a reason to withhold date loss, not to manufacture certainty.
            if old!=new:conflicts.add(field)
            else:proofs.append({'fact_index':i,**p})
        if no_cue and f['valid_to'] is None and f['valid_from']!=call['started_at']:
            old=f['valid_from'];f['valid_from']=call['started_at']
            patches.append({'fact_index':i,'field':'valid_from','before':old,'after':call['started_at'],
                'evidence':{'rule':'official_v21_missing_date_convention','whole_call_time_cue_screen':'none',
                'caveat':'Automated conservative absence screen, not a claim to know actual onset.'}})
        supported={(p['fact_index'],p['field']) for p in proofs}
        for field in ['valid_from','valid_to']:
            value=f[field]
            if value is None:continue
            if field in conflicts:
                masked.append({'fact_index':i,'field':field,'reason':'Potential conflict between original date and linked relative-day heuristic; original retained, no automatic relabel.',
                    'candidate_anchors':[p for p in direct if p['field']==field]})
                continue
            if (i,field) in supported or value[:10]==call['started_at'][:10] or value[:10] in covered_days:continue
            masked.append({'fact_index':i,'field':field,'reason':'Historical/deadline timestamp not verified by source anchor or official date bank; no blind fallback for closed history.'})
        facts.append(f)
    # A verified next onset may SUPPORT an unchanged prior interval end. No implicit relabel.
    for i,f in enumerate(call['facts']):
        if f['attribute'] not in STATE or f['polarity']!=1 or f['valid_to'] is None:continue
        candidates=[p for p in proofs if p['field']=='valid_from' and p['fact_index']!=i
            and call['facts'][p['fact_index']]['object']['id']==f['object']['id']
            and call['facts'][p['fact_index']]['attribute']==f['attribute']
            and call['facts'][p['fact_index']]['polarity']==1
            and call['facts'][p['fact_index']]['value']!=f['value']
            and p['datetime']==facts[i]['valid_to'] and facts[i]['valid_from']<p['datetime']]
        if len(candidates)==1:
            masked=[m for m in masked if (m['fact_index'],m['field'])!=(i,'valid_to')]
    return {'facts':facts},patches,proofs,masked

def normalize_generated(value,call):
    """No gold and no overwriting valid-but-wrong dates. No role/index repair."""
    result=deepcopy(value);changes=[]
    if not isinstance(result,dict) or set(result)!={'facts'} or not isinstance(result['facts'],list):return result,changes
    for i,f in enumerate(result['facts']):
        if not isinstance(f,dict) or not isinstance(f.get('source'),dict):continue
        old=f.get('valid_from')
        try:datetime.fromisoformat(old);continue
        except (ValueError,TypeError):pass
        try:proof=bound_dates(f,call)
        except (KeyError,TypeError,ValueError):continue
        candidates=[p for p in proof if p['field']=='valid_from']
        replacement=candidates[0]['datetime'] if len(candidates)==1 else call['started_at'] if old is None and not has_time_cue(call) else None
        if replacement is None:continue
        end=f.get('valid_to')
        if end is not None:
            try:
                if datetime.fromisoformat(end)<datetime.fromisoformat(replacement):continue
            except (ValueError,TypeError):continue
        f['valid_from']=replacement
        changes.append({'fact_index':i,'field':'valid_from','before':old,'after':replacement,
            'rule':'event_linked_calendar_or_guarded_missing_date_convention','evidence':candidates})
    return result,changes
