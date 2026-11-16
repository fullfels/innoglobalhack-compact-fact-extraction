"""Small, conservative calendar parser. Never rewrites a source utterance."""
import re, calendar
from datetime import datetime, timedelta
NUM = dict(zip('один одну два две три четыре пять шесть семь восемь девять десять одиннадцать двенадцать тринадцать четырнадцать пятнадцать шестнадцать семнадцать восемнадцать девятнадцать двадцать'.split(),[1,1,2,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]))
def anchors(text, started_at):
    base=datetime.fromisoformat(started_at); day=base.replace(hour=0,minute=0,second=0,microsecond=0)
    result=[]
    def add(m,dt,kind):result.append({'expression':m.group(),'span':[m.start(),m.end()],'datetime':dt.isoformat(timespec='seconds'),'kind':kind})
    for m in re.finditer(r'\b(позавчера|вчера|сегодня|завтра|послезавтра|сейчас)\b',text,re.I):
        word=m.group().lower(); dt=base if word=='сейчас' else day+timedelta(days={'позавчера':-2,'вчера':-1,'сегодня':0,'завтра':1,'послезавтра':2}[word]);add(m,dt,'day')
    num='|'.join(sorted(NUM,key=len,reverse=True))
    pat=rf'\b(?:(через|щерез)\s+)?(\d{{1,2}}|{num})\s+(день|дня|дней|днуй|сутки|суток|неделю|недели|недель)(?:\s+(назад))?\b'
    for m in re.finditer(pat,text,re.I):
        if bool(m[1])==bool(m[4]):continue
        n=int(m[2]) if m[2].isdigit() else NUM[m[2].lower()]
        if not 1<=n<=31:continue
        n*=7 if m[3].lower().startswith('недел') else 1
        add(m,day+timedelta(days=n if m[1] else -n),'relative_days')
    weekdays={'понедельник':0,'вторник':1,'среду':2,'среды':2,'четверг':3,'пятницу':4,'пятницы':4,'субботу':5,'субботы':5,'воскресенье':6,'воскресенья':6}
    for m in re.finditer(r'\b(с|со|в|на)\s+(прошлого\s+|прошлой\s+|ближайший\s+|ближайшую\s+)?('+ '|'.join(weekdays)+r')\b',text,re.I):
        target=weekdays[m[3].lower()];past=m[1].lower() in ('с','со') or bool(m[2] and m[2].lower().startswith('прошл'))
        delta=-((base.weekday()-target)%7 or 7) if past else ((target-base.weekday())%7 or 7)
        add(m,day+timedelta(days=delta),'weekday_past' if past else 'weekday_future')
    return sorted(result,key=lambda a:a['span'])

def lexical(value,text):
    stop={'клиент','оператор','услуг','котор','тариф','было','была','работ','нормаль','сейчас','вчера','сегодня'}
    def tokens(s):return {w[:5] for w in re.findall(r'[а-яёa-z0-9]+',s.lower().replace('ё','е')) if len(w)>=4 and not any(w.startswith(t) for t in stop)}
    v=tokens(value); t=tokens(text)
    return bool(v and len(v&t)>=min(2,len(v)))
