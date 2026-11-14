"""Qwen3.5 probe: no training; separate run with an improved prompt and fenced-JSON handling."""
from __future__ import annotations
import argparse
from collections import Counter
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import sys
import time

from common import atomic_json, atomic_jsonl, decode_and_validate, make_messages, read_jsonl, sha256, unwrap_json

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--revision", default="2fc06364715b967f1860aea9cf38778875588b17")
    ap.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    ap.add_argument("--max-new-tokens", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    if args.limit is not None and args.limit < 1:
        ap.error("--limit должен быть положительным")
    if args.max_new_tokens < 1:
        ap.error("--max-new-tokens должен быть положительным")
    args.out.mkdir(parents=True, exist_ok=True)
    records = args.out / "records"
    records.mkdir(exist_ok=True)
    all_calls = read_jsonl(args.data / "val.jsonl")
    calls = all_calls[:args.limit] if args.limit else all_calls
    baseline = {c["call_id"]: c for c in read_jsonl(args.data / "baseline_pred.jsonl")}
    prompt_path = Path(__file__).with_name("prompt.txt")
    instruction = prompt_path.read_text(encoding="utf-8")
    catalog = json.loads((args.data / "catalog.json").read_text(encoding="utf-8"))

    # Heavy imports stay out of audit/check-only paths.
    import torch
    from huggingface_hub import HfApi
    from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("GPU недоступна. В Colab выберите среду выполнения с GPU.")
    versions = {}
    for name in ("torch", "transformers", "accelerate", "huggingface-hub", "tokenizers", "safetensors"):
        versions[name] = importlib.metadata.version(name)
    hardware = {"cpu": platform.processor(), "cpu_count": os.cpu_count(), "torch_threads": torch.get_num_threads()}
    if args.device == "cuda":
        props = torch.cuda.get_device_properties(0)
        hardware.update(gpu=props.name, gpu_total_bytes=props.total_memory, cuda_runtime=torch.version.cuda)
    fingerprint = {
        "model": args.model, "requested_revision": args.revision,
        "prompt_sha256": sha256(prompt_path), "data_sha256": sha256(args.data / "val.jsonl"),
        "catalog_sha256": sha256(args.data / "catalog.json"),
        "script_sha256": sha256(__file__), "common_sha256": sha256(Path(__file__).with_name("common.py")),
        "seed": args.seed, "device": args.device, "versions": versions, "hardware": hardware,
        "do_sample": False, "enable_thinking": False, "max_new_tokens": args.max_new_tokens,
    }
    manifest_path = args.out / "run_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["fingerprint"] != fingerprint:
            raise RuntimeError("Настройки отличаются от сохранённого прогона. Используйте новую папку --out.")
        revision = manifest["resolved_revision"]
    else:
        revision = HfApi().model_info(args.model, revision=args.revision).sha
        manifest = {"fingerprint": fingerprint, "resolved_revision": revision}
        atomic_json(manifest_path, manifest)

    selected = []
    pending = []
    for i, call in enumerate(calls):
        # Position and hash together prevent path injection and call_id collisions.
        import hashlib
        key = hashlib.sha256(call["call_id"].encode()).hexdigest()[:16]
        path = records / f"{i:05d}_{key}.json"
        if path.exists():
            record = json.loads(path.read_text(encoding="utf-8"))
            if record["call_id"] != call["call_id"]:
                raise RuntimeError("Повреждённое сохранение: call_id не совпадает")
            selected.append(record)
        else:
            selected.append(None)
            pending.append((i, call, path))

    def write_outputs():
        ready = [(call, record) for call, record in zip(calls, selected) if record is not None]
        # A failed call counts as missing facts in metrics, but is ALWAYS counted as a failure in summary.
        atomic_jsonl(args.out / "pred_before.jsonl", [{"call_id": c["call_id"], "facts": r["facts"]} for c, r in ready])
        atomic_jsonl(args.out / "gold_evaluation.jsonl", [c for c, r in ready])
        atomic_jsonl(args.out / "baseline_selected.jsonl", [baseline[c["call_id"]] for c, r in ready])
        status = Counter(r["status"] for c, r in ready)
        times = sorted(r["latency_sec"] for c, r in ready if r["status"] == "ok" and r["latency_sec"] is not None)
        summary = {
            "n_selected": len(calls), "n_completed": len(ready), "status_counts": dict(status),
            "n_failed": sum(n for s, n in status.items() if s != "ok"),
            "n_markdown_wrappers_removed": sum(r.get("markdown_wrapper_removed", False) for c, r in ready),
            "n_valid_empty_answers": sum(r["status"] == "ok" and not r["facts"] for c, r in ready),
            "n_null_start_dates": sum(f.get("valid_from") is None for c, r in ready for f in r["facts"]),
            "n_input_too_long": status.get("input_too_long", 0),
            "valid_json_and_structure_fraction": status.get("ok", 0) / len(ready) if ready else None,
            "latency_mean_sec": sum(times)/len(times) if times else None,
            "latency_p50_sec": times[len(times)//2] if times else None,
            "latency_p95_sec": times[min(len(times)-1, int(len(times)*0.95))] if times else None,
            "latency_success_count": len(times),
            "all_attempts_total_sec": sum(r["latency_sec"] or 0 for c, r in ready),
            "max_peak_cuda_allocated_mb": max((r.get("peak_cuda_allocated_mb", 0) for c, r in ready), default=0),
            "measurement_note": "Latency mean/p50/p95 use successful calls only; failed attempts are reported separately. batch=1; warmed, CUDA-synchronised end-to-end call latency excluding model loading. Not a full CPU/GPU benchmark or proof of teacher savings.",
            "validation_note": "Local structural/source checks; not proof of semantic correctness or a final organiser-approved schema.",
            "evaluation_note": "Failed generations are explicit in n_failed and contribute empty facts; inspect records. 10 calls are a smoke test, not representative final quality.",
        }
        atomic_json(args.out / "summary.json", summary)
        preview = [{"call_id": c["call_id"], "turns": c["turns"], "expected_facts": c["facts"],
                    "predicted_facts": r["facts"], "status": r["status"], "error": r["error"], "raw": r["raw"]}
                   for c, r in ready[:3]]
        atomic_json(args.out / "preview.json", preview)
        return summary

    if not pending:
        print("Все выбранные звонки уже обработаны. Повторная генерация не нужна.")
        print(json.dumps(write_outputs(), ensure_ascii=False, indent=2))
        return

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    start_load = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.float16 if args.device == "cuda" else torch.float32
    config = AutoConfig.from_pretrained(args.model, revision=revision)
    loader = AutoModelForImageTextToText if config.model_type == "qwen3_5" else AutoModelForCausalLM
    model = loader.from_pretrained(
        args.model, revision=revision, dtype=dtype, attn_implementation="sdpa",
    ).to(args.device).eval()
    text_config = getattr(model.config, "text_config", model.config)
    native_context_limit = int(getattr(text_config, "max_position_embeddings", 32768))
    context_limit = min(native_context_limit, 32768)
    if args.max_new_tokens >= context_limit:
        raise ValueError("Лимит ответа должен быть меньше контекста модели")
    manifest.update({
        "resolved_revision": revision,
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "context_limit": context_limit, "native_context_limit": native_context_limit, "dtype": str(dtype), "python": sys.version,
        "platform": platform.platform(), "gpu": torch.cuda.get_device_name(0) if args.device == "cuda" else None,
        "model_load_seconds": time.perf_counter() - start_load,
    })
    atomic_json(manifest_path, manifest)
    print("Model:", args.model, "revision:", revision, "parameters:", manifest["total_parameters"], flush=True)

    def sync():
        if args.device == "cuda":
            torch.cuda.synchronize()

    warm = tokenizer.apply_chat_template(
        [{"role": "user", "content": 'Верни JSON {"facts":[]}'}],
        add_generation_prompt=True, enable_thinking=False, tokenize=True, return_dict=True, return_tensors="pt",
    ).to(args.device)
    with torch.inference_mode():
        model.generate(**warm, max_new_tokens=8, do_sample=False, pad_token_id=tokenizer.pad_token_id)
    sync()
    del warm

    for i, call, path in pending:
        sync()
        t0 = time.perf_counter()
        record = {"call_id": call["call_id"], "status": "ok", "facts": [], "raw": "", "error": None,
                  "markdown_wrapper_removed": False, "input_tokens": None, "output_tokens": 0, "latency_sec": None, "peak_cuda_allocated_mb": 0}
        inputs = output = generated = None
        if args.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        try:
            messages = make_messages(call, instruction, catalog)
            inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=False,
                       tokenize=True, return_dict=True, return_tensors="pt")
            record["input_tokens"] = int(inputs["input_ids"].shape[-1])
            if record["input_tokens"] + args.max_new_tokens > context_limit:
                record["status"] = "input_too_long"
                record["error"] = "Вход + резерв ответа превышают контекст. Вход не обрезан; нужен отдельный способ обработки длинных звонков."
            else:
                inputs = inputs.to(args.device)
                with torch.inference_mode():
                    output = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False,
                                            use_cache=True, pad_token_id=tokenizer.pad_token_id)
                sync()
                generated = output[0, record["input_tokens"]:]
                record["output_tokens"] = len(generated)
                record["raw"] = tokenizer.decode(generated, skip_special_tokens=True)
                record["markdown_wrapper_removed"] = unwrap_json(record["raw"]) != record["raw"].strip()
                eos = model.generation_config.eos_token_id
                eos_ids = set(eos if isinstance(eos, list) else [eos])
                if len(generated) >= args.max_new_tokens and int(generated[-1]) not in eos_ids:
                    record["status"] = "output_limit"
                    record["error"] = "Ответ достиг лимита токенов; не считаем его успешным. Сырой текст сохранён."
                else:
                    record["facts"] = decode_and_validate(record["raw"], call)
        except torch.cuda.OutOfMemoryError as exc:
            record["status"] = "out_of_memory"
            record["error"] = str(exc)
        except (ValueError, KeyError, TypeError, IndexError) as exc:
            record["status"] = "invalid_output"
            record["error"] = str(exc)
        except RuntimeError as exc:
            record["status"] = "runtime_error"
            record["error"] = str(exc)
        finally:
            if args.device == "cuda":
                record["peak_cuda_allocated_mb"] = torch.cuda.max_memory_allocated() / 1024**2
            record["latency_sec"] = round(time.perf_counter() - t0, 4)
            del inputs, output, generated
            if record["status"] == "out_of_memory":
                torch.cuda.empty_cache()
        selected[i] = record
        atomic_json(path, record)
        if (i + 1) % 10 == 0 or i + 1 == len(calls):
            write_outputs()
        print(f"[{i+1}/{len(calls)}] {call['call_id']} {record['status']} facts={len(record['facts'])} sec={record['latency_sec']}", flush=True)
    print(json.dumps(write_outputs(), ensure_ascii=False, indent=2), flush=True)

if __name__ == "__main__":
    main()
