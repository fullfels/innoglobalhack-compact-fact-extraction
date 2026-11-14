"""Idempotent importer: source docs/tasks.csv, live status stays in GitHub Issues.
Requires authenticated gh. Does not invite users, close issues or change branch protection.
"""
import argparse
import csv
import json
import re
import subprocess
from pathlib import Path

REPO = 'fullfels/innoglobalhack-compact-fact-extraction'
MILESTONES = ['Contracts & Data','Mock E2E','Real Model Integration','Model Experiments','Final Evaluation & Release']
LABELS = {
 **{f'area:{a}':'1d76db' for a in ['data','ml','backend','frontend','devops','integration','docs','evaluation']},
 'priority:P0':'b60205','priority:P1':'fbca04','priority:P2':'c5def5',
 **{f'type:{t}':'5319e7' for t in ['bug','feature','research','testing']},
 'blocked':'d93f0b','status:ready':'0e8a16','status:in-progress':'0052cc','status:in-review':'bfd4f2'}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--gh',default='gh');ap.add_argument('--apply',action='store_true');a=ap.parse_args()
 root=Path(__file__).resolve().parents[1]
 tasks=list(csv.DictReader((root/'docs/tasks.csv').open(encoding='utf-8-sig')))
 assert len({t['id'] for t in tasks})==len(tasks)
 if not a.apply:
  print(json.dumps({'repository':REPO,'tasks':len(tasks),'milestones':MILESTONES,'labels':list(LABELS)},ensure_ascii=False,indent=2));return
 def api(path,method='GET',data=None):
  cmd=[a.gh,'api',path,'--method',method]
  if data is not None:cmd+=['--input','-']
  r=subprocess.run(cmd,input=json.dumps(data) if data is not None else None,text=True,capture_output=True)
  if r.returncode:raise RuntimeError(r.stderr.strip())
  return json.loads(r.stdout) if r.stdout.strip() else None
 def pages(path):
  rows=[];page=1
  while True:
   part=api(path+('&' if '?' in path else '?')+f'per_page=100&page={page}');rows+=part
   if len(part)<100:return rows
   page+=1
 prefix='repos/'+REPO
 repo=api(prefix);assert repo['private'] and repo['owner']['login']=='fullfels'
 existing_labels={l['name'] for l in pages(prefix+'/labels')}
 for name,color in LABELS.items():
  if name not in existing_labels:api(prefix+'/labels','POST',{'name':name,'color':color})
 milestones={m['title']:m['number'] for m in pages(prefix+'/milestones?state=all')}
 for title in MILESTONES:
  if title not in milestones:milestones[title]=api(prefix+'/milestones','POST',{'title':title})['number']
 issues={}
 for i in pages(prefix+'/issues?state=all'):
  if 'pull_request' in i:continue
  m=re.match(r'^\[(T\d{2})\] ',i['title'])
  if m:
   if m[1] in issues:raise RuntimeError('Duplicate task ID already exists: '+m[1])
   issues[m[1]]=i
 collaborators={c['login'].lower() for c in pages(prefix+'/collaborators')}
 mapping={};created=0
 for t in tasks:
  tid=t['id'];num=int(tid[1:]);owner=t['owner']
  area='data' if owner=='DE' else 'ml' if owner=='ML' else 'backend'
  if num in {17,42}:area='frontend'
  if num in {18,34,41}:area='devops'
  if num in {2,19,20}:area='integration'
  if num in {11,25,26,27,28,30,31,35,43}:area='evaluation'
  if num in {1,36,37,38,39,40}:area='docs'
  milestone=0 if num in {1,2,3,4,5,6,7,8,9,12,13,14,15,29,44} else 1 if num in {16,17,18} else 2 if num in {10,11,19,20,21,22} else 3 if num in {23,24,25,26,27,28,30,31,32,42,43,45} else 4
  body=(f"<!-- task-id:{tid} -->\nTask ID: **{tid}**\n\n{t['technical_description']}\n\n"
        f"- Владелец роли: **{owner}**; reviewer: {t['reviewer']}\n- Приоритет: {t['priority']}\n"
        f"- Оценка активной работы: {t['estimate_hours']} ч\n- Плановый deadline: {t['deadline_msk']} (не фактический статус)\n"
        f"- Зависимости: {t['dependencies'] or 'нет'}\n- Deliverable: `{t['deliverable']}`\n\n"
        f"## Definition of Done\n{t['definition_of_done']}\n\n## Проверка\n{t['verification']}\n\n"
        f"## Риск\n{t['risk']} — docs/12_RISKS.md\n\nИсточник: docs/tasks.csv и docs/06_BACKLOG.md. "
        "Оперативный статус ведётся в этом Issue; существующие задачи importer не перезаписывает.\n")
  if tid not in issues:
   assignees=['fullfels'] if owner=='FS' else ['QFa1'] if owner=='ML' and 'qfa1' in collaborators else []
   issues[tid]=api(prefix+'/issues','POST',{'title':f'[{tid}] {t["title"]}','body':body,'labels':[f'area:{area}',f'priority:{t["priority"]}','type:research' if owner=='ML' else 'type:feature'],'milestone':milestones[MILESTONES[milestone]],'assignees':assignees});created+=1
  i=issues[tid];mapping[tid]={'number':i['number'],'url':i['html_url'],'role':owner}
 (root/'docs/github_task_map.json').write_text(json.dumps({'repository':REPO,'tasks':mapping},ensure_ascii=False,indent=2)+'\n')
 print(json.dumps({'created':created,'mapped':len(mapping),'milestones':len(milestones),'collaborators':sorted(collaborators)},ensure_ascii=False))

if __name__=='__main__':main()
