"""Pinned Qwen3 engine with a wall-clock stopping criterion; no alternate HTTP service."""
import json,time,gc,os,hashlib
from pathlib import Path
from ml.qwen.bridge import messages
from ml.qwen.v21_bridge import V21Bridge
from src.pipeline.adapters import PipelineError,ModelDescriptor,ModelResources
from src.pipeline.lifecycle import LoadedModel,ManagedAdapter
from src.pipeline.service import Pipeline

MODEL='Qwen/Qwen3-0.6B';REVISION='c1899de289a04d12100db370d81485cdf75e47ca'

class Engine:
    def __init__(self,model,tokenizer,instruction,catalog,*,deadline_epoch=None,max_call_seconds=120,device='cuda'):
        self.model,self.tokenizer=model,tokenizer;self.instruction,self.catalog=instruction,catalog
        self.deadline_epoch=deadline_epoch;self.max_call_seconds=max_call_seconds;self.device=device
        self.last_raw=None;self.last_diagnostics=None
    def generate(self,call):
        import torch
        from transformers import StoppingCriteria,StoppingCriteriaList
        self.last_raw=None;self.last_diagnostics=None;self.model.eval()
        end=min(self.deadline_epoch or float('inf'),time.time()+self.max_call_seconds)
        if time.time()>=end:raise PipelineError('QWEN_GENERATION_TIMEOUT',503,'qwen.budget')
        x=self.tokenizer.apply_chat_template(messages(call,self.instruction,self.catalog),tokenize=True,add_generation_prompt=True,
            enable_thinking=False,return_tensors='pt',return_dict=True)
        n=int(x['input_ids'].shape[-1]);limit=2048
        if n+limit>24576:raise PipelineError('QWEN_CONTEXT_OVERFLOW',422,'qwen.context')
        eos=self.model.generation_config.eos_token_id;ids=set(eos if isinstance(eos,list) else [eos])|{self.tokenizer.eos_token_id};ids.discard(None)
        class ClockStop(StoppingCriteria):
            def __call__(self,input_ids,scores,**kwargs):
                return torch.full((input_ids.shape[0],),time.time()>=end,dtype=torch.bool,device=input_ids.device)
        try:
            with torch.inference_mode():
                y=self.model.generate(**x.to(self.device),do_sample=False,use_cache=True,max_new_tokens=limit,
                    eos_token_id=sorted(ids),pad_token_id=self.tokenizer.pad_token_id,
                    stopping_criteria=StoppingCriteriaList([ClockStop()]))
        except torch.cuda.OutOfMemoryError:raise PipelineError('QWEN_OUT_OF_MEMORY',503,'qwen.generate') from None
        suffix=y[0,n:];finished=bool(len(suffix) and int(suffix[-1]) in ids)
        self.last_raw=self.tokenizer.decode(suffix,skip_special_tokens=True)
        self.last_diagnostics={'input_tokens':n,'output_tokens':len(suffix),'ended_with_eos':finished,'max_call_seconds':self.max_call_seconds}
        if not finished and time.time()>=end:raise PipelineError('QWEN_GENERATION_TIMEOUT',503,'qwen.generate')
        if not finished and len(suffix)>=limit:raise PipelineError('QWEN_OUTPUT_LIMIT',503,'qwen.generate')
        return self.last_raw

def factory():
    """MWS_PIPELINE_FACTORY=ml.qwen.v21_runtime:factory; base must already be cached locally."""
    root=Path(os.environ['MTS_MODEL_DIR']).resolve();m=json.loads((root/'model_manifest.json').read_text(encoding='utf-8'))
    if not m.get('reload_verified') or m['base_model']!=MODEL or m['base_revision']!=REVISION:raise PipelineError('QWEN_ARTIFACT_NOT_VERIFIED',503,'qwen.loader')
    for rel,want in m['sha256'].items():
        p=(root/rel).resolve()
        if not p.is_relative_to(root) or hashlib.sha256(p.read_bytes()).hexdigest()!=want:raise PipelineError('QWEN_ARTIFACT_HASH_MISMATCH',503,'qwen.loader')
    device=os.getenv('MTS_MODEL_DEVICE','cpu')
    def loader(resources):
        import torch
        from transformers import AutoTokenizer,AutoModelForCausalLM
        from peft import PeftModel
        dtype=torch.float16 if device=='cuda' else torch.float32
        base=AutoModelForCausalLM.from_pretrained(os.getenv('MTS_MODEL_BASE_PATH',MODEL),revision=REVISION,local_files_only=True,dtype=dtype,attn_implementation='sdpa').to(device)
        model=PeftModel.from_pretrained(base,root/'adapter').eval();tok=AutoTokenizer.from_pretrained(root/'tokenizer',local_files_only=True)
        engine=Engine(model,tok,(root/'prompt.txt').read_text(encoding='utf-8'),json.loads((root/'catalog.json').read_text(encoding='utf-8')),device=device)
        def close():
            engine.model=None;gc.collect()
            if device=='cuda':torch.cuda.empty_cache()
        return LoadedModel(inference=V21Bridge(engine),close=close)
    adapter=ManagedAdapter(ModelResources(str(root),None,{'device':device}),loader,
        descriptor=ModelDescriptor(m['model_id'],'real','transformers-peft-v21'),revision=REVISION,
        artifact_sha256=hashlib.sha256((root/'model_manifest.json').read_bytes()).hexdigest(),
        dtype='float16' if device=='cuda' else 'float32',context=24576,parameters=m['total_parameters'],manifest_id=m['model_id'])
    return Pipeline(adapter)
