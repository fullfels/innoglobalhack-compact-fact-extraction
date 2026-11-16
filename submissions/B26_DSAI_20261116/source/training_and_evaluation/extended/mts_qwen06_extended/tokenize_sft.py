"""Completion-only labels, real checkpoint tokenizer, and strict length auditing.

No GPU required. Never truncates or drops examples. encode_record() can also be
used by a causal-LM Trainer; preserve its labels when padding batches.
"""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path

MODEL = "Qwen/Qwen3-0.6B"
REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"


def encode_record(row, tokenizer, max_length=None):
    assert [m["role"] for m in row["prompt"]] == ["system", "user"]
    assert len(row["completion"]) == 1 and row["completion"][0]["role"] == "assistant"
    prefix = tokenizer.apply_chat_template(row["prompt"], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    full = tokenizer.apply_chat_template(row["prompt"] + row["completion"], tokenize=False, add_generation_prompt=False, enable_thinking=False)
    assert full.startswith(prefix), "Training/inference template prefix mismatch"
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    ids = tokenizer.encode(full, add_special_tokens=False)
    n = len(prefix_ids)
    assert ids[:n] == prefix_ids, "Token boundary merged; do not guess label start"
    eos = tokenizer.eos_token_id
    assert eos is not None and ids[n:].count(eos) == 1
    end = ids.index(eos, n) + 1
    ids = ids[:end]
    labels = [-100] * n + ids[n:]
    assert len(ids) == len(labels) and len(ids) > n and ids[-1] == eos
    actual = tokenizer.decode(ids[n:-1], skip_special_tokens=False, clean_up_tokenization_spaces=False)
    assert actual == row["completion"][0]["content"], "Masked answer differs from target JSON"
    assert json.loads(actual).keys() == {"facts"}
    if max_length is not None and len(ids) > max_length:
        raise ValueError(f'{row["call_id"]}: {len(ids)} tokens > {max_length}; no truncation allowed')
    return {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": labels}


def summary(values):
    vs = sorted(values)
    return {"min": vs[0], "mean": round(sum(vs) / len(vs), 2),
            **{label: vs[min(len(vs) - 1, int((len(vs) - 1) * q))] for label, q in (("p50", .5), ("p90", .9), ("p95", .95), ("p99", .99), ("max", 1))}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    ap.add_argument("--tokenizer", default=None, help="Local pinned tokenizer folder, or default HF checkpoint")
    ap.add_argument("--local-only", action="store_true")
    args = ap.parse_args()
    from transformers import AutoTokenizer
    import transformers
    kwargs = {"local_files_only": args.local_only}
    if args.tokenizer is None:
        kwargs["revision"] = REVISION
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer or MODEL, **kwargs)
    output = {"model": MODEL, "revision": REVISION, "transformers_tokenizer_audit_version": transformers.__version__,
              "tokenizer_class": type(tokenizer).__name__, "vocab_with_added_tokens": len(tokenizer),
              "enable_thinking": False, "eos_id": tokenizer.eos_token_id, "pad_id": tokenizer.pad_token_id,
              "model_weights_loaded": False, "gpu_used": False, "truncation": False, "splits": {}}
    per_call = []
    for name, path in (("train", args.root / "sft/train.jsonl"), ("dev", args.root / "sft/dev.jsonl"),
                       ("pilot_train", args.root / "pilot/sft_train_32.jsonl"), ("pilot_dev", args.root / "pilot/sft_dev_8.jsonl")):
        lengths, answer_lengths, prompt_lengths = [], [], []
        ids_by_budget = {"le8192": [], "8193to16384": [], "gt16384": []}
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            encoded = encode_record(row, tokenizer)
            total = len(encoded["input_ids"])
            prompt = encoded["labels"].count(-100)
            lengths.append(total)
            prompt_lengths.append(prompt)
            answer_lengths.append(total - prompt)
            bucket = "le8192" if total <= 8192 else "8193to16384" if total <= 16384 else "gt16384"
            ids_by_budget[bucket].append(row["call_id"])
            per_call.append({"split": name, "call_id": row["call_id"], "prompt_tokens": prompt,
                             "answer_tokens_with_eos": total - prompt, "total_tokens": total})
        output["splits"][name] = {"calls": len(lengths), "total": summary(lengths), "prompt": summary(prompt_lengths),
                                  "answer_with_eos": summary(answer_lengths), "budget_counts": {k: len(v) for k, v in ids_by_budget.items()},
                                  "all_prefixes_and_masks_checked": True, "all_json_answers_exactly_decoded": True,
                                  "call_ids_by_budget": ids_by_budget}
        print(name, len(lengths), summary(lengths), flush=True)
    report = args.root / "audit/token_lengths.json"
    report.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (args.root / "audit/token_lengths_by_call.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for row in per_call:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
