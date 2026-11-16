"""One persisted wall-clock budget, shared by setup, training, evaluation and export."""
from dataclasses import dataclass
import hashlib,json,math,time
from pathlib import Path

class BudgetStop(RuntimeError):pass

@dataclass
class Budget:
    origin_epoch:float
    total_minutes:float=300.
    def __post_init__(self):
        if not 60<=self.total_minutes<=300:raise ValueError('Budget must be60..300 minutes, including setup/evaluation')
        self._wall_anchor=max(time.time(),self.origin_epoch)
        self._mono_anchor=time.monotonic()
    @property
    def hard_deadline(self):return self.origin_epoch+self.total_minutes*60
    @property
    def finish_deadline(self):return self.hard_deadline-20*60
    @property
    def train_deadline(self):return self.hard_deadline-max(35*60,min(80*60,self.total_minutes*60*.28))
    @property
    def supervisor_deadline(self):return self.hard_deadline-5*60
    def now(self):return max(time.time(),self._wall_anchor+time.monotonic()-self._mono_anchor)
    def left(self,phase='finish'):
        deadline={'finish':self.finish_deadline,'train':self.train_deadline,'hard':self.hard_deadline,'supervisor':self.supervisor_deadline}[phase]
        return max(0.,deadline-self.now())
    def require(self,seconds,phase='finish'):
        if self.left(phase)<seconds:raise BudgetStop('Not enough shared experiment time for '+phase)
    def as_dict(self):return {'origin_epoch':self.origin_epoch,'total_minutes':self.total_minutes,
        'hard_deadline_epoch':self.hard_deadline,'train_deadline_epoch':self.train_deadline,
        'finish_deadline_epoch':self.finish_deadline,'supervisor_deadline_epoch':self.supervisor_deadline,
        'elapsed_seconds':max(0,self.now()-self.origin_epoch),'scope':'Setup/upload/install/load/probe/train/eval/reload/export. Not a claim about platform quota availability.'}

def load_budget(path):
    v=json.loads(Path(path).read_text(encoding='utf-8'));return Budget(v['origin_epoch'],v['total_minutes'])

def verify_snapshot(path):
    root=Path(path).resolve();m=json.loads((root/'snapshot_hashes.json').read_text(encoding='utf-8'))
    required={'adapter/adapter_model.safetensors','adapter/adapter_config.json','tokenizer/tokenizer.json','prompt.txt','catalog.json','training_state.pt','reload_reference.pt','checkpoint.json'}
    if not required.issubset(m):raise ValueError('Incomplete snapshot manifest')
    for rel,want in m.items():
        p=(root/rel).resolve()
        if not p.is_relative_to(root) or p.is_symlink() or not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=want:raise ValueError('Snapshot checksum mismatch: '+rel)
    return root

def validate_resume(state,manifest_sha,goal,learning_rate,order,budget):
    if state['manifest_sha256']!=manifest_sha or state['goal']!=goal or state['learning_rate']!=learning_rate or state['order']!=order:
        raise ValueError('Resume payload/data/prompt/order/schedule mismatch; use a deliberate new warm-start instead')
    if state['budget_origin_epoch']!=budget.origin_epoch or state['budget_total_minutes']!=budget.total_minutes:
        raise ValueError('Resume cannot reset or expand the shared experiment budget')
    if not 0<=state['new_updates']<goal:raise ValueError('Checkpoint already completed or invalid')
    if not isinstance(state.get('grad_scaler'),dict):raise ValueError('Missing FP16 GradScaler state')

def score_key(report,baseline,*,initial=False):
    if not report.get('complete') or not baseline.get('complete'):return (False,-1.,-1.,-1.,-999)
    m,b=report['metrics'],baseline['metrics']
    def num(d,k):return d.get(k) if d.get(k) is not None else -1.
    valid=initial or (all(num(m,k)>=num(b,k)-.02 for k in ['fact_recall','temporal_recall','state_accuracy'])
                       and report['failed_generations']<=baseline['failed_generations'])
    p,r=num(m,'fact_precision'),num(m,'fact_recall');f1=2*p*r/(p+r) if p+r>0 else 0.
    return (valid,min(num(m,'temporal_recall'),num(m,'state_accuracy')),f1,r,-report['failed_generations'])

def midpoint_affordable(budget,remaining_updates,step_seconds,eval_seconds):
    return budget.left('train')>remaining_updates*step_seconds*1.15+eval_seconds*1.2+120

def checkpoint_paths_to_prune(run,keep_recent=2):
    run=Path(run).resolve();base=(run/'checkpoints').resolve()
    if not base.is_relative_to(run) or base==run:raise ValueError('Unsafe checkpoint root')
    paths=sorted([p for p in base.glob('update*') if p.is_dir() and not p.is_symlink()])
    keep=set(paths[-keep_recent:])
    for name in ['best_pointer.json','last_pointer.json']:
        pointer=run/name
        if pointer.is_file():
            p=(run/json.loads(pointer.read_text())['relative_path']).resolve()
            if not p.is_relative_to(base):raise ValueError('Unsafe checkpoint pointer')
            keep.add(p)
    keep|={p for p in paths if p.name=='update000000'}
    return [p for p in paths if p not in keep and p.resolve().is_relative_to(base)]
