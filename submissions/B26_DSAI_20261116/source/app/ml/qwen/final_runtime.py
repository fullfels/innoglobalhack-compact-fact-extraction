"""Pinned generation and a shared-contract factory for the verified v22 export."""
import gc,hashlib,json,os,time
from pathlib import Path
from ml.qwen.bridge import messages
from ml.qwen.final_bridge import FinalBridge
from src.pipeline.adapters import PipelineError,ModelDescriptor,ModelResources
from src.pipeline.lifecycle import LoadedModel,ManagedAdapter
from src.pipeline.service import Pipeline

MODEL='Qwen/Qwen3-0.6B'
REVISION='c1899de289a04d12100db370d81485cdf75e47ca'

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def load_model(adapter,tokenizer_path,*,precision='fp16',local_only=False,device='cuda',trainable=False):
    import torch
    from transformers import AutoTokenizer,AutoModelForCausalLM,BitsAndBytesConfig
    from peft import PeftModel
    if precision not in {'fp16','int8','int4'}:raise ValueError('Unknown precision')
    if precision!='fp16' and device!='cuda':raise ValueError('Quantized experiments require CUDA')
    kw=dict(revision=REVISION,local_files_only=local_only,trust_remote_code=False,
            dtype=torch.float16 if device=='cuda' else torch.float32,attn_implementation='sdpa')
    if precision=='int8':kw.update(device_map={'':0},quantization_config=BitsAndBytesConfig(load_in_8bit=True))
    if precision=='int4':kw.update(device_map={'':0},quantization_config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.float16))
    base=AutoModelForCausalLM.from_pretrained(os.getenv('MTS_MODEL_BASE_PATH',MODEL),**kw)
    if precision=='fp16':base=base.to(device)
    if base.config.model_type!='qwen3':raise ValueError('Unexpected base architecture')
    model=PeftModel.from_pretrained(base,adapter,is_trainable=trainable).eval()
    if trainable:
        for p in model.parameters():
            if p.requires_grad:p.data=p.data.float()
    tok=AutoTokenizer.from_pretrained(tokenizer_path,local_files_only=True)
    if base.config.eos_token_id!=tok.eos_token_id:raise ValueError('Tokenizer/base EOS mismatch')
    return model,tok

class Engine:
    def __init__(self,model,tokenizer,instruction,catalog,*,deadline_epoch=None,max_call_seconds=120,device='cuda'):
        self.model,self.tokenizer=model,tokenizer;self.instruction,self.catalog=instruction,catalog
        self.deadline_epoch=deadline_epoch;self.max_call_seconds=max_call_seconds;self.device=device
        self.last_raw=None;self.last_diagnostics=None
    def generate(self,call):
        import torch
        from transformers import StoppingCriteria,StoppingCriteriaList
        self.last_raw=None;self.last_diagnostics=None
        end=min(self.deadline_epoch or float('inf'),time.time()+self.max_call_seconds)
        if time.time()>=end:raise PipelineError('QWEN_GENERATION_TIMEOUT',503,'qwen.budget')
        x=self.tokenizer.apply_chat_template(messages(call,self.instruction,self.catalog),tokenize=True,add_generation_prompt=True,enable_thinking=False,return_tensors='pt',return_dict=True)
        n=int(x['input_ids'].shape[-1]);limit=2048
        if n+limit>24576:raise PipelineError('QWEN_CONTEXT_OVERFLOW',422,'qwen.context')
        eos=self.model.generation_config.eos_token_id;ids=set(eos if isinstance(eos,list) else [eos])|{self.tokenizer.eos_token_id};ids.discard(None)
        class ClockStop(StoppingCriteria):
            def __call__(self,input_ids,scores,**kwargs):return torch.full((input_ids.shape[0],),time.time()>=end,dtype=torch.bool,device=input_ids.device)
        try:
            with torch.inference_mode():y=self.model.generate(**x.to(self.device),do_sample=False,use_cache=True,max_new_tokens=limit,eos_token_id=sorted(ids),pad_token_id=self.tokenizer.pad_token_id,stopping_criteria=StoppingCriteriaList([ClockStop()]))
        except torch.cuda.OutOfMemoryError:raise PipelineError('QWEN_OUT_OF_MEMORY',503,'qwen.generate') from None
        suffix=y[0,n:];finished=bool(len(suffix) and int(suffix[-1]) in ids)
        self.last_raw=self.tokenizer.decode(suffix,skip_special_tokens=True)
        self.last_diagnostics={'input_tokens':n,'output_tokens':len(suffix),'ended_with_eos':finished,'max_call_seconds':self.max_call_seconds}
        if not finished:
            raise PipelineError('QWEN_GENERATION_TIMEOUT' if time.time()>=end else 'QWEN_OUTPUT_LIMIT',503,'qwen.generate')
        return self.last_raw

def factory():
    root=Path(os.environ['MTS_MODEL_DIR']).resolve();m=json.loads((root/'model_manifest.json').read_text(encoding='utf-8'))
    if not m.get('reload_verified') or m['base_revision']!=REVISION or m['base_model']!=MODEL:raise PipelineError('QWEN_ARTIFACT_NOT_VERIFIED',503,'qwen.loader')
    for relative,expected in m['sha256'].items():
        p=(root/relative).resolve()
        if not p.is_relative_to(root) or sha(p)!=expected:raise PipelineError('QWEN_ARTIFACT_HASH_MISMATCH',503,'qwen.loader')
    device=os.getenv('MTS_MODEL_DEVICE','cpu')
    if device not in {'cpu','cuda'}:raise ValueError('MTS_MODEL_DEVICE must be cpu or cuda')
    def loader(resources):
        model,tok=load_model(root/'adapter',root/'tokenizer',local_only=True,device=device)
        engine=Engine(model,tok,(root/'prompt.txt').read_text(encoding='utf-8'),json.loads((root/'catalog.json').read_text(encoding='utf-8')),device=device)
        def close():
            engine.model=None;gc.collect()
            if device=='cuda':
                import torch
                torch.cuda.empty_cache()
        return LoadedModel(FinalBridge(engine,calendar=m['calendar_enabled']),close)
    descriptor=ModelDescriptor(m['model_id'],'real','transformers-peft-temporal-v23')
    adapter=ManagedAdapter(ModelResources(str(root),None,{'device':device}),loader,descriptor=descriptor,revision=REVISION,artifact_sha256=sha(root/'model_manifest.json'),dtype='float16' if device=='cuda' else 'float32',context=24576,parameters=m['total_parameters'],manifest_id=m['model_id'])
    return Pipeline(adapter)
