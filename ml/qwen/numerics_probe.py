"""Actual FP32 forward/backward/update and longest-call memory probe; no quality claim."""
import argparse
import gc
import hashlib
import importlib.metadata
import json
from pathlib import Path
import random
import time

from ml.qwen.train import read, save, sha, load_module, suffix_loss, check_loss_equivalence
from ml.qwen.prompting import effective_example


def main():
    import torch
    from peft import get_peft_model, LoraConfig
    from ml.qwen.runtime import load_base, MODEL, REVISION
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists():
        raise RuntimeError('Use a new numerics output directory')
    args.out.mkdir(parents=True)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    config = {'purpose': 'Numerical feasibility only, not extraction quality', 'precision': 'float32',
              'autocast': False, 'grad_scaler': False, 'seed': 42, 'learning_rate': 5e-5,
              'gpu': torch.cuda.get_device_name(0), 'base': MODEL, 'revision': REVISION,
              'dependencies': {p: importlib.metadata.version(p) for p in ('torch', 'transformers', 'peft')},
              'sources': {name: sha(Path(__file__).with_name(name)) for name in ('numerics_probe.py', 'train.py', 'runtime.py', 'prompting.py')}}
    save(args.out / 'config.json', config)
    manifest = json.loads((args.data_root / 'manifest.json').read_text(encoding='utf-8'))
    for rel, expected in manifest['data_sha256'].items():
        assert sha(args.data_root / rel) == expected, rel
    tokenize = load_module('probe_tokenizer', args.data_root / 'tokenize_sft.py')
    model, tokenizer = load_base(precision='float32')
    targets = [name for name, m in model.named_modules() if isinstance(m, torch.nn.Linear) and '.language_model.layers.' in name]
    assert targets and all('visual' not in name for name in targets)
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0., bias='none', task_type='CAUSAL_LM', target_modules=targets))
    assert model.dtype == torch.float32
    save(args.out / 'loss_equivalence.json', check_loss_equivalence(model, tokenizer))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model.enable_input_require_grads()
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=5e-5)
    rows = read(args.data_root / 'pilot/sft_train_32.jsonl')
    order = list(range(len(rows))); random.Random(42).shuffle(order)
    records = []
    for index in order[:3]:
        row = tokenize.encode_record(effective_example(rows[index]), tokenizer, max_length=8192)
        model.train(); optimizer.zero_grad(set_to_none=True); torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize(); start = time.perf_counter()
        loss = suffix_loss(model, row)
        if not torch.isfinite(loss): raise RuntimeError('Non-finite FP32 loss')
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(parameters, 1.)
        if not torch.isfinite(norm): raise RuntimeError('Non-finite FP32 gradients')
        tracked = next(p for p in parameters if p.grad is not None and bool(torch.any(p.grad != 0)))
        old = tracked.detach().clone(); optimizer.step()
        changed = bool(torch.any(tracked.detach() != old)); del old
        if not changed: raise RuntimeError('No parameter update')
        torch.cuda.synchronize()
        records.append({'call_id': rows[index]['call_id'], 'loss': float(loss.detach()),
                        'gradient_norm': float(norm), 'weight_changed': changed,
                        'elapsed_sec': time.perf_counter() - start,
                        'peak_cuda_allocated_mib': torch.cuda.max_memory_allocated() / 1024**2})
        save(args.out / 'steps.json', records)
        print('fp32_update', len(records), records[-1], flush=True)
        del loss
    optimizer.zero_grad(set_to_none=True)
    lengths = read(args.data_root / 'audit/token_lengths_by_call.jsonl')
    longest = max((r for r in lengths if r['split'] == 'train'), key=lambda r: r['total_tokens'])['call_id']
    raw = next(r for r in read(args.data_root / 'sft/train.jsonl') if r['call_id'] == longest)
    row = tokenize.encode_record(effective_example(raw), tokenizer, max_length=16384)
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    long_result = {'tokens': len(row['input_ids']), 'optimizer_update': False}
    try:
        loss = suffix_loss(model, row); loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(parameters, 1.)
        long_result.update(passed=bool(torch.isfinite(loss) and torch.isfinite(norm)),
                           peak_cuda_allocated_mib=torch.cuda.max_memory_allocated() / 1024**2)
    except torch.cuda.OutOfMemoryError:
        long_result.update(passed=False, error='CUDA_OOM')
    finally:
        optimizer.zero_grad(set_to_none=True)
    result = {'status': 'complete', 'three_finite_updates': len(records) == 3,
              'long_call': long_result, 'pilot_feasible': len(records) == 3 and long_result['passed'],
              'quality_measured': False}
    save(args.out / 'numerics_probe.json', result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
