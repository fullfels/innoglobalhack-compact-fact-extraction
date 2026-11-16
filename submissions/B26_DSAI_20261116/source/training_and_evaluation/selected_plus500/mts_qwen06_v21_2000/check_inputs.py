"""Full input/target/token audit before any model download or optimizer update."""
from pathlib import Path
import argparse,hashlib,json,statistics,sys

def read(path):return [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines() if s.strip()]
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def account(call):return (call.get('series') or {}).get('account_id',call['call_id'].split(':')[0])

def verify(root,*,compat_local=False):
    sys.path.insert(0,str(root))
    from transformers import AutoTokenizer
    from ml.qwen.bridge import messages,parse_candidates
    from src.contracts.models import CallInput
    from tokenize_sft import encode_record
    kwargs={'extra_special_tokens':{}} if compat_local else {}
    tokenizer=AutoTokenizer.from_pretrained(root/'tokenizer',local_files_only=True,**kwargs)
    assert tokenizer.eos_token_id==151645 and tokenizer.pad_token_id==151643
    manifest=json.loads((root/'manifest.json').read_text(encoding='utf-8')) if (root/'manifest.json').exists() else None
    if manifest:
        for rel,want in manifest['sha256'].items():
            p=(root/rel).resolve();assert p.is_relative_to(root.resolve()) and sha(p)==want,rel
    catalog=json.loads((root/'official_catalog.json').read_text(encoding='utf-8'))
    prompt=(root/'prompt.txt').read_text(encoding='utf-8')
    from ml.qwen.train import validate_lora_targets
    expected=json.loads((root/'expected_lora_targets.json').read_text(encoding='utf-8'))
    cfg=json.loads((root/'initial_adapter/adapter_config.json').read_text(encoding='utf-8'))
    assert len(expected)==196 and cfg['r']==16 and cfg['lora_alpha']==32 and cfg['lora_dropout']==0
    assert cfg['base_model_name_or_path']=='Qwen/Qwen3-0.6B'
    validate_lora_targets(expected+['lm_head'],expected,cfg['target_modules'])
    report={'all_checks_passed':False,'local_compatibility_tokenizer_override':compat_local,
            'native_colab_audit_required_before_training':True,'splits':{}}
    sets={};accounts={}
    for split,name in [('train','train2000.jsonl'),('dev','dev32_gold_official.jsonl')]:
        raw=read(root/'data'/name);sft=read(root/'sft'/(split+'.jsonl'))
        assert len(raw)==len(sft)==(2000 if split=='train' else 32)
        sets[split]={c['call_id'] for c in raw};assert len(sets[split])==len(raw)
        accounts[split]={account(c) for c in raw}
        lengths=[];answers=[];prefixes=[];loss_tokens=[];masked_tokens=[];fact_count=0;repairs=[]
        for call,row in zip(raw,sft):
            assert call['call_id']==row['call_id']
            inp=CallInput.model_validate({k:call[k] for k in ['call_id','started_at','turns']})
            assert row['prompt']==messages(inp,prompt,catalog),'SFT / inference serialization mismatch'
            value=json.loads(row['completion'][0]['content']);facts=value['facts']
            assert len(facts)==len(call['facts']),'Fact list shortened'
            for index,(old,new) in enumerate(zip(call['facts'],facts)):
                assert set(new)=={'object','attribute','value','polarity','speaker','valid_from','valid_to','source'}
                assert new['object']=={k:v for k,v in old['object'].items() if k!='id'}
                for key in ['attribute','value','polarity','speaker','valid_to']:assert new[key]==old[key],(call['call_id'],index,key)
                assert new['source']=={'turn_idx':old['source']['turn_idx'],'quote':None,'span':None}
                if new['valid_from']!=old['valid_from']:
                    assert new['valid_from']==call['started_at'] and new['valid_to'] is None
                    from ml.qwen.temporal_v21 import has_time_cue
                    assert not has_time_cue(call)
                    repairs.append({'call_id':call['call_id'],'fact_index':index})
            parsed=parse_candidates(row['completion'][0]['content'],inp)
            assert len(parsed.candidates)==len(facts)
            encoded=encode_record(row,tokenizer)
            assert len(encoded['input_ids'])==len(encoded['labels'])
            assert encoded['input_ids'][-1]==encoded['labels'][-1]==151645
            assert all(encoded['labels'][i]==-100 for i in encoded['masked_date_token_indices'])
            a=len(encoded['input_ids'])-encoded['answer_start'];p=encoded['answer_start']
            assert a<2048 and p+2048<=24576
            lengths.append(p+a);answers.append(a);prefixes.append(p)
            loss_tokens.append(sum(t!=-100 for t in encoded['labels']));masked_tokens.append(encoded['masked_date_tokens'])
            fact_count+=len(facts)
        report['splits'][split]={'calls':len(raw),'facts':fact_count,'min_tokens':min(lengths),'max_tokens':max(lengths),
            'median_tokens':statistics.median(lengths),'max_prefix_tokens':max(prefixes),'max_answer_tokens':max(answers),
            'loss_tokens':sum(loss_tokens),'masked_timestamp_tokens':sum(masked_tokens),
            'masked_fraction_of_completion_tokens':sum(masked_tokens)/sum(answers),'missing_date_convention_edits':repairs,
            'no_truncation':True,'all_targets_pass_candidate_schema':True,'exact_train_inference_prompt':True}
    assert not sets['train']&sets['dev'] and not accounts['train']&accounts['dev']
    val=read(root/'evaluation/val.jsonl')
    assert not (sets['train']|sets['dev'])&{c['call_id'] for c in val}
    assert not (accounts['train']|accounts['dev'])&{account(c) for c in val}
    report.update(all_checks_passed=True,native_colab_tokenizer_audit_performed=not compat_local,
                  payload_manifest_sha256=sha(root/'manifest.json') if manifest else None,
                  official_validation_unmodified_sha256=sha(root/'evaluation/val.jsonl'),
                  evaluator_unmodified_sha256=sha(root/'evaluation/case_metrics.py'))
    return report

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--compat-local',action='store_true');args=ap.parse_args()
    report=verify(args.root.resolve(),compat_local=args.compat_local)
    args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
