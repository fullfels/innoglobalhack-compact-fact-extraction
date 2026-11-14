"""A real PyTorch/PEFT LoRA pilot and one bounded main SFT; no synthetic ML results.

Uses completion-only suffix logits to avoid allocating vocabulary logits for the
large instruction/catalog prefix. Equivalence to the model loss is checked first.
"""
from __future__ import annotations
import argparse
from collections import Counter
import gc
import hashlib
import importlib.metadata
import importlib.util
import json
import math
from pathlib import Path
import random
import shutil
import statistics
import time
from ml.qwen.prompting import effective_example, effective_instruction
RUN_CREATED = False


def read(path):
    return [json.loads(s) for s in Path(path).read_text(encoding="utf-8").splitlines() if s.strip()]


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def autocast_for(model):
    import torch
    return torch.autocast("cuda", dtype=torch.float16, enabled=model.dtype == torch.float16)


def supervised_tokens(row):
    n = sum(label != -100 for label in row["labels"])
    if n == 0:
        raise ValueError("No supervised tokens")
    return n


def token_loss_weights(rows):
    counts = [supervised_tokens(row) for row in rows]
    if not counts:
        raise ValueError("Empty accumulation window")
    total = sum(counts)
    return [n / total for n in counts]


def validate_lora_targets(available, expected, selectors):
    if not isinstance(selectors, (list, tuple, set)) or not selectors:
        raise ValueError("Expected explicit LoRA target names")
    selected = {name for name in available if any(name == key or name.endswith('.' + key) for key in selectors)}
    if selected != set(expected):
        raise ValueError("Adapter targets do not resolve to exactly the expected language layers")


def record_failure(out, phase, error_type, own_run):
    out = Path(out)
    if out.exists() and not own_run:
        return False  # Refusing an existing output directory must not modify that run.
    path = out / ("pilot.json" if phase == "pilot" else "sft_report.json")
    existing = json.loads(path.read_text()) if path.exists() else {}
    save(path, {**existing, "status": "failed", "phase": phase, "main_sft_ready": False,
                "error_type": error_type, "successful_training_claimed": False})
    return True


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def suffix_loss(model, encoded, *, device="cuda"):
    import torch
    import torch.nn.functional as F
    labels = encoded["labels"]
    n = next(i for i, label in enumerate(labels) if label != -100)
    assert n > 0 and all(x == -100 for x in labels[:n])
    assert labels[n:] == encoded["input_ids"][n:]
    a = len(labels) - n
    ids = torch.tensor([encoded["input_ids"]], dtype=torch.long, device=device)
    mask = torch.ones_like(ids)
    logits = model(input_ids=ids, attention_mask=mask, use_cache=False, logits_to_keep=a + 1).logits[:, :-1]
    targets = ids[:, n:]
    assert logits.shape[:2] == targets.shape
    # FP32 loss only on answer positions, in chunks; prefix labels never participate.
    loss = logits.new_zeros((), dtype=torch.float32)
    for start in range(0, a, 64):
        loss = loss + F.cross_entropy(logits[:, start:start + 64].float().reshape(-1, logits.shape[-1]),
                                     targets[:, start:start + 64].reshape(-1), reduction="sum")
    return loss / a


def check_loss_equivalence(model, tokenizer):
    import torch
    import torch.nn.functional as F
    logits = torch.tensor([[[3.,0.,0.]]*4, [[3.,0.,0.]]*4], device="cuda", requires_grad=True)
    target = torch.tensor([[-100,-100,0,0], [1,1,1,1]], device="cuda")
    reference_tokens = F.cross_entropy(logits.reshape(-1,3), target.reshape(-1), ignore_index=-100)
    toy_rows = [{"labels": row} for row in target.tolist()]
    weights = token_loss_weights(toy_rows)
    combined = sum(w*F.cross_entropy(logits[i], target[i], ignore_index=-100) for i,w in enumerate(weights))
    torch.testing.assert_close(reference_tokens, combined, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(torch.autograd.grad(reference_tokens, logits, retain_graph=True)[0],
                               torch.autograd.grad(combined, logits)[0], atol=1e-6, rtol=1e-6)
    normalization_check = {"passed": True, "reference_token_mean": float(reference_tokens.detach()),
                           "weighted_accumulation": float(combined.detach()), "gradient_equivalence": True}
    p = tokenizer.apply_chat_template([{"role": "user", "content": "Верни JSON."}], tokenize=False,
                                      add_generation_prompt=True, enable_thinking=False)
    prefix = tokenizer.encode(p, add_special_tokens=False)
    ids = prefix + tokenizer.encode('{"facts":[]}', add_special_tokens=False) + [tokenizer.eos_token_id]
    encoded = {"input_ids": ids, "labels": [-100] * len(prefix) + ids[len(prefix):]}
    tensor = torch.tensor([ids], device="cuda")
    labels = torch.tensor([encoded["labels"]], device="cuda")
    model.eval()
    with torch.no_grad(), autocast_for(model):
        reference = model(input_ids=tensor, attention_mask=torch.ones_like(tensor), labels=labels, use_cache=False).loss
        optimized = suffix_loss(model, encoded)
    torch.testing.assert_close(reference.float(), optimized.float(), rtol=.003, atol=.003)
    return {"passed": True, "full_masked_loss": float(reference), "suffix_loss": float(optimized),
            "prefix_tokens_masked": len(prefix), "answer_tokens": len(ids) - len(prefix),
            "accumulation_token_normalization": normalization_check}


def input_call(raw):
    from src.contracts.models import CallInput
    return CallInput.model_validate({k: raw[k] for k in ("call_id", "started_at", "turns")})


def quality_run(model, tokenizer, calls, data_root, out, *, max_new_tokens=1536):
    import torch
    from ml.qwen.bridge import parse_candidates
    from ml.qwen.runtime import HFTextEngine
    from src.pipeline.adapters import PipelineError, ModelDescriptor
    from src.pipeline.service import Pipeline
    instruction = (data_root / "prompt.txt").read_text(encoding="utf-8")
    catalog = json.loads((data_root / "official_catalog.json").read_text(encoding="utf-8"))
    engine = HFTextEngine(model, tokenizer, instruction, catalog, max_new_tokens=max_new_tokens)
    out.mkdir(parents=True, exist_ok=True)
    records, predictions, serving_predictions = [], [], []
    for raw_call in calls:
        call = input_call(raw_call)
        print("evaluation_start", call.call_id, flush=True)
        record = {"call_id": call.call_id, "status": "ok", "raw": None, "error_code": None, "facts": None}
        torch.cuda.synchronize(); start = time.perf_counter()
        try:
            record["raw"] = engine.generate(call)
            batch = parse_candidates(record["raw"], call)
            # Raw extraction metrics include asserted agreement facts; serving confirmation
            # is a distinct policy layer and is not falsely declared verified here.
            record["facts"] = [c.fact.model_dump(mode="json") for c in batch.candidates]
            record["unresolved_agreements"] = sum(c.fact.attribute == "agreement" and c.confirmation != "confirmed" for c in batch.candidates)
            class MeasuredBatch:
                descriptor = ModelDescriptor("measured-qwen-offline", "real")
                state = "ready"
                def extract(self, _call): return batch
            serving, serving_warnings = Pipeline(MeasuredBatch()).run(call)
            record["serving_facts"] = [f.model_dump(mode="json") for f in serving.facts]
            record["serving_warnings"] = list(serving_warnings)
        except PipelineError as exc:
            record.update(status="failed", error_code=exc.code)
            record["raw"] = record["raw"] or engine.last_raw
        except Exception as exc:
            record.update(status="failed", error_code="UNEXPECTED_" + type(exc).__name__)
        torch.cuda.synchronize()
        record["elapsed_sec"] = time.perf_counter() - start
        record["generation"] = engine.last_diagnostics
        records.append(record)
        # Official evaluator requires an entry for every call. A failure contributes
        # no correct facts, but stays failed/null in raw records, never successful empty.
        predictions.append({"call_id": call.call_id, "facts": record["facts"] if record["status"] == "ok" else []})
        serving_predictions.append({"call_id": call.call_id, "facts": record.get("serving_facts", []) if record["status"] == "ok" else []})
        name = hashlib.sha256(call.call_id.encode()).hexdigest()[:20]
        save(out / "records" / (name + ".json"), record)
        print("evaluation", call.call_id, record["status"], "facts", len(record["facts"] or []),
              "error", record["error_code"], "seconds", round(record["elapsed_sec"], 2), flush=True)
    with (out / "predictions.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for r in predictions: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with (out / "serving_predictions.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for r in serving_predictions: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    evaluator = load_module("official_case_metrics", data_root / "evaluation/case_metrics.py")
    metrics = evaluator.evaluate({c["call_id"]: c for c in calls}, {p["call_id"]: p for p in predictions})
    serving_metrics = evaluator.evaluate({c["call_id"]: c for c in calls}, {p["call_id"]: p for p in serving_predictions})
    report = {"metrics": metrics, "n_calls": len(calls), "status_counts": dict(Counter(r["status"] for r in records)),
              "serving_metrics_without_de_normalizers": serving_metrics,
              "valid_multifact_calls": sum(r["status"] == "ok" and len(r["facts"]) > 1 for r in records),
              "valid_empty_answers": sum(r["status"] == "ok" and r["facts"] == [] for r in records),
              "failed_generations": sum(r["status"] != "ok" for r in records),
              "unresolved_agreements": sum(r.get("unresolved_agreements", 0) for r in records),
              "measurement_scope": "Raw extraction; agreement confirmation/state/date DE normalizers are not connected. Not end-to-end serving acceptance."}
    save(out / "metrics.json", report)
    return report


def diagnostic_subset(calls, limit):
    """A fixed prefix chosen by count, without inspecting labels or predictions."""
    if limit < 1:
        raise ValueError("Diagnostic count must be positive")
    return calls[:limit]


def teacher_loss(model, rows):
    import torch
    model.eval(); total_loss, total_tokens = 0., 0
    with torch.no_grad(), autocast_for(model):
        for row in rows:
            n = supervised_tokens(row)
            value = float(suffix_loss(model, row))
            if not math.isfinite(value): raise RuntimeError("Non-finite dev loss")
            total_loss += value * n; total_tokens += n
    return total_loss / total_tokens


def checkpoint(model, tokenizer, path, data_root, step, probe):
    import torch
    path.mkdir(parents=True, exist_ok=True)
    model.eval()
    model.save_pretrained(path / "adapter", safe_serialization=True, save_embedding_layers=False)
    tokenizer.save_pretrained(path / "tokenizer")
    for name, dest in (("prompt.txt", "prompt.txt"), ("official_catalog.json", "catalog.json")):
        if name == "prompt.txt":
            (path / dest).write_text(effective_instruction((data_root / name).read_text(encoding="utf-8")), encoding="utf-8")
        else:
            shutil.copyfile(data_root / name, path / dest)
    x = torch.tensor([probe["input_ids"]], device="cuda")
    with torch.no_grad(), autocast_for(model):
        logits = model(input_ids=x, attention_mask=torch.ones_like(x), use_cache=False, logits_to_keep=1).logits.float().cpu()
    torch.save({"input_ids": x.cpu(), "logits": logits}, path / "reload_reference.pt")
    save(path / "checkpoint.json", {"step": step, "kind": "LoRA", "probe_sha256": sha(path / "reload_reference.pt")})


def main():
    global RUN_CREATED
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--phase", choices=("pilot", "main"), default="pilot")
    ap.add_argument("--pilot-report", type=Path)
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--accumulation", type=int)
    ap.add_argument("--learning-rate", type=float, default=2e-4)
    ap.add_argument("--budget-hours", type=float, default=4)
    ap.add_argument("--full-eval", action="store_true")
    ap.add_argument("--precision", choices=("float16", "float32"), default="float32")
    ap.add_argument("--initial-adapter", type=Path, help="Warm-start adapter only; starts a new optimizer, not a training resume")
    ap.add_argument("--min-timing-steps", type=int, default=50)
    ap.add_argument("--diagnostic-every", type=int, default=0)
    ap.add_argument("--baseline-count", type=int, default=2,
                    help="Fixed train/dev prefix for paired before/after diagnostics; not full quality evaluation")
    args = ap.parse_args()
    if args.steps < 1 or args.diagnostic_every < 0 or args.min_timing_steps < 1 or args.baseline_count < 1 or (args.accumulation is not None and args.accumulation < 1) or args.learning_rate <= 0 or args.budget_hours <= 0:
        ap.error("steps/accumulation/learning-rate/budget-hours must be positive")
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from ml.qwen.runtime import MODEL, REVISION, load_base
    if not torch.cuda.is_available(): raise RuntimeError("CUDA GPU required; no silent CPU fallback")
    if args.out.exists() and any(args.out.iterdir()): raise RuntimeError("Use a fresh output folder; previous runs are immutable")
    args.out.mkdir(parents=True, exist_ok=True)
    RUN_CREATED = True
    data_root = args.data_root.resolve()
    manifest = json.loads((data_root / "manifest.json").read_text(encoding="utf-8"))
    for rel, expected in manifest["data_sha256"].items():
        assert sha(data_root / rel) == expected, rel
    if args.phase == "main":
        if not args.pilot_report: raise RuntimeError("Main SFT requires measured successful pilot report")
        prior = json.loads(args.pilot_report.read_text(encoding="utf-8"))
        if not prior.get("main_sft_ready"): raise RuntimeError("Pilot gates failed; main SFT not authorized by experiment protocol")
        if (prior.get("base_model") != MODEL or prior.get("base_revision") != REVISION
                or prior.get("dtype") != args.precision
                or prior.get("dataset_manifest_sha256") != sha(data_root / "manifest.json")):
            raise RuntimeError("Pilot model, precision or dataset does not match this main run")
        if prior["eta_main_one_epoch_sec"] > args.budget_hours * 3600:
            raise RuntimeError("Measured main ETA exceeds the configured training-time budget; revise config explicitly")
    seed = 42; random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    config = {"phase": args.phase, "seed": seed, "base_model": MODEL, "base_revision": REVISION,
              "dtype": args.precision, "quantization": None, "rank": 16, "alpha": 32, "dropout": 0.,
              "grad_scaler_enabled": args.precision == "float16",
              "loss_normalization": "supervised_tokens_across_accumulation_window",
              "dev_loss_normalization": "all_supervised_dev_tokens",
              "min_timing_steps": args.min_timing_steps,
              "diagnostic_every": args.diagnostic_every,
              "initialization": {"kind": "plain_base"},
              "learning_rate": args.learning_rate, "accumulation": args.accumulation or (2 if args.phase == "pilot" else 4),
              "baseline_diagnostic_count_per_split": args.baseline_count,
              "diagnostic_policy": "Fixed first N calls, identical before/after; full after-train and inner-dev measured separately. Count reduced after measured 122s runaway base generation; no metric representativeness claim.",
              "gradient_checkpointing": True, "training_context": 8192 if args.phase == "pilot" else 16384,
              "budget_hours": args.budget_hours, "dataset_manifest_sha256": sha(data_root / "manifest.json"),
              "source_sha256": {name: sha(Path(__file__).with_name(name)) for name in ("train.py", "bridge.py", "runtime.py", "prompting.py")},
              "effective_prompt_sha256": hashlib.sha256(effective_instruction((data_root / "prompt.txt").read_text(encoding="utf-8")).encode()).hexdigest(),
              "prompt_view_note": "Only ML system instruction changed: current observation date differs from unknowable historical time. Same effective prompt before/after. Original data/labels/splits unchanged.",
              "dependencies": {p: importlib.metadata.version(p) for p in ("torch", "transformers", "peft", "accelerate", "tokenizers", "safetensors")},
              "hardware": {"gpu": torch.cuda.get_device_name(0), "vram_bytes": torch.cuda.get_device_properties(0).total_memory,
                           "cuda": torch.version.cuda, "torch_threads": torch.get_num_threads()}}
    save(args.out / "training_config.json", config)
    tokenize = load_module("case_sft_tokenization", data_root / "tokenize_sft.py")
    prefix = "pilot/sft_" if args.phase == "pilot" else "sft/"
    train_path = data_root / (prefix + ("train_32.jsonl" if args.phase == "pilot" else "train.jsonl"))
    dev_path = data_root / (prefix + ("dev_8.jsonl" if args.phase == "pilot" else "dev.jsonl"))
    train_raw = read(data_root / ("pilot/train_32.jsonl" if args.phase == "pilot" else "data/train.jsonl"))
    dev_raw = read(data_root / ("pilot/dev_8.jsonl" if args.phase == "pilot" else "data/dev.jsonl"))
    model, tokenizer = load_base(precision=args.precision)
    train_rows = [tokenize.encode_record(effective_example(r), tokenizer, max_length=config["training_context"]) for r in read(train_path)]
    dev_rows = [tokenize.encode_record(effective_example(r), tokenizer, max_length=config["training_context"]) for r in read(dev_path)]
    target_names = [name for name, module in model.named_modules() if isinstance(module, torch.nn.Linear) and ".language_model.layers." in name]
    assert target_names and all("visual" not in name for name in target_names)
    if args.initial_adapter:
        adapter_config = json.loads((args.initial_adapter / "adapter_config.json").read_text(encoding="utf-8"))
        if (adapter_config.get("base_model_name_or_path") != MODEL or adapter_config.get("r") != 16
                or adapter_config.get("lora_alpha") != 32):
            raise RuntimeError("Warm-start adapter does not match base/LoRA configuration")
        validate_lora_targets([name for name, module in model.named_modules() if isinstance(module, torch.nn.Linear)],
                              target_names, adapter_config.get("target_modules"))
        model = PeftModel.from_pretrained(model, args.initial_adapter, is_trainable=True)
        config["initialization"] = {"kind": "warm_start_adapter_new_optimizer", "comparison_baseline": "initial_adapter_not_plain_base",
            "sha256": {p.name: sha(p) for p in sorted(args.initial_adapter.iterdir()) if p.is_file()}}
    else:
        model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0., bias="none",
                                task_type="CAUSAL_LM", target_modules=target_names))
    for parameter in model.parameters():
        if parameter.requires_grad: parameter.data = parameter.data.float()
    config["target_modules"] = target_names
    config["total_parameters"] = sum(p.numel() for p in model.parameters())
    config["trainable_parameters"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    save(args.out / "training_config.json", config)
    equivalence = check_loss_equivalence(model, tokenizer)
    save(args.out / "loss_equivalence.json", equivalence)
    diagnostic_train = diagnostic_subset(train_raw, args.baseline_count)
    diagnostic_dev = diagnostic_subset(dev_raw, args.baseline_count)
    print("stage before_training_diagnostics", len(diagnostic_train), len(diagnostic_dev), flush=True)
    before_train = quality_run(model, tokenizer, diagnostic_train, data_root, args.out / "before_train")
    before_dev = quality_run(model, tokenizer, diagnostic_dev, data_root, args.out / "before_dev")
    if args.phase == "main" and args.full_eval:
        quality_run(model, tokenizer, read(data_root / "evaluation/val.jsonl"), data_root, args.out / "official_val_before")
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=0.)
    scaler = torch.amp.GradScaler("cuda", enabled=args.precision == "float16", init_scale=128.)
    accumulation = config["accumulation"]
    steps = args.steps if args.phase == "pilot" else math.ceil(len(train_rows) / accumulation)
    initial_dev = teacher_loss(model, dev_rows)
    print("stage optimizer_training", "steps", steps, "initial_dev_loss", initial_dev, flush=True)
    best_loss, best_step = initial_dev, None
    order = list(range(len(train_rows))); random.Random(seed).shuffle(order)
    position = 0; logs = []
    torch.cuda.reset_peak_memory_stats(); total_start = time.perf_counter()
    for step in range(steps):
        torch.cuda.synchronize(); start = time.perf_counter(); model.train(); optimizer.zero_grad(set_to_none=True)
        warmup = max(1, math.ceil(steps * .05))
        multiplier = min((step + 1) / warmup, max(0., (steps - step) / max(1, steps - warmup)))
        for group in optimizer.param_groups: group["lr"] = args.learning_rate * multiplier
        if position >= len(order):
            random.Random(seed + step).shuffle(order); position = 0
        count = min(accumulation, len(order) - position) if args.phase == "main" else accumulation
        batch_rows = []
        for _ in range(count):
            if position >= len(order): random.Random(seed + step).shuffle(order); position = 0
            row = train_rows[order[position]]; position += 1
            batch_rows.append(row)
        weights = token_loss_weights(batch_rows)
        losses = []
        for row, weight in zip(batch_rows, weights):
            with autocast_for(model): loss = suffix_loss(model, row)
            if not torch.isfinite(loss): raise RuntimeError("Non-finite training loss")
            losses.append(float(loss.detach())); scaler.scale(loss * weight).backward()
        scaler.unscale_(optimizer); norm = torch.nn.utils.clip_grad_norm_(parameters, 1.)
        if args.precision == "float32" and not torch.isfinite(norm):
            raise RuntimeError("Non-finite FP32 gradient; refusing optimizer update")
        old_scale = scaler.get_scale(); scaler.step(optimizer); scaler.update()
        torch.cuda.synchronize()
        log = {"step": step + 1, "loss": sum(v*w for v,w in zip(losses, weights)),
               "mean_example_loss_diagnostic_only": statistics.mean(losses),
               "supervised_tokens": sum(supervised_tokens(row) for row in batch_rows),
               "microbatches": count, "learning_rate": optimizer.param_groups[0]["lr"],
               "elapsed_sec": time.perf_counter() - start, "gradient_norm": float(norm),
               "optimizer_step_skipped": scaler.get_scale() < old_scale,
               "cuda_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
               "cuda_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2}
        if not math.isfinite(log["gradient_norm"]): log["gradient_norm"] = None
        logs.append(log); save(args.out / "training_log.json", logs)
        print("step", step + 1, "loss", round(log["loss"], 5), "seconds", round(log["elapsed_sec"], 2), flush=True)
        if args.diagnostic_every and (step + 1) % args.diagnostic_every == 0:
            print("stage intermediate_generation", step + 1, flush=True)
            quality_run(model, tokenizer, diagnostic_train, data_root, args.out / f"step_{step+1}_diagnostic")
        if (step + 1) % (16 if args.phase == "pilot" else max(1, steps // 2)) == 0 or step + 1 == steps:
            current = teacher_loss(model, dev_rows)
            log["dev_loss"] = current
            if current < best_loss:
                best_loss, best_step = current, step + 1
                checkpoint(model, tokenizer, args.out / "best", data_root, step + 1, train_rows[0])
            save(args.out / "training_log.json", logs)
    elapsed_training = time.perf_counter() - total_start
    checkpoint(model, tokenizer, args.out / "last", data_root, steps, train_rows[0])
    selected = args.out / ("best" if best_step is not None else "last")
    timing = logs[-50:]
    per_micro = statistics.median(r["elapsed_sec"] / r["microbatches"] for r in timing)
    eta = per_micro * len(read(data_root / "sft/train.jsonl")) * 1.2
    # Test long-call memory without learning from extra calls in the 32-example pilot.
    long_probe = {"passed": None}
    if args.phase == "pilot":
        full = read(data_root / "sft/train.jsonl")
        lengths = read(data_root / "audit/token_lengths_by_call.jsonl")
        longest = max((r for r in lengths if r["split"] == "train"), key=lambda r: r["total_tokens"])["call_id"]
        long_row = tokenize.encode_record(effective_example(next(r for r in full if r["call_id"] == longest)), tokenizer, max_length=16384)
        try:
            optimizer.zero_grad(set_to_none=True); model.train(); torch.cuda.reset_peak_memory_stats()
            with autocast_for(model): long_loss = suffix_loss(model, long_row)
            scaler.scale(long_loss).backward()
            long_probe = {"passed": True, "tokens": len(long_row["input_ids"]), "optimizer_update": False,
                          "peak_cuda_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2}
        except torch.cuda.OutOfMemoryError:
            long_probe = {"passed": False, "error": "CUDA_OOM", "optimizer_update": False}
        finally:
            optimizer.zero_grad(set_to_none=True)
            if "long_loss" in locals(): del long_loss
    del loss
    del optimizer, parameters, model; gc.collect(); torch.cuda.empty_cache()
    fresh, _ = load_base(precision=args.precision)
    model = PeftModel.from_pretrained(fresh, selected / "adapter").eval()
    reference = torch.load(selected / "reload_reference.pt", weights_only=True, map_location="cpu")
    x = reference["input_ids"].to("cuda")
    with torch.no_grad(), autocast_for(model):
        actual = model(input_ids=x, attention_mask=torch.ones_like(x), use_cache=False, logits_to_keep=1).logits.float().cpu()
    torch.testing.assert_close(reference["logits"], actual, atol=.003, rtol=.003)
    reload_result = {"passed": True, "fresh_base_and_saved_adapter": True,
                     "max_logit_abs_difference": float((reference["logits"] - actual).abs().max())}
    save(args.out / "reload_test.json", reload_result)
    after_train = quality_run(model, tokenizer, train_raw if args.phase == "pilot" else train_raw[:1], data_root, args.out / "after_train")
    after_dev = quality_run(model, tokenizer, dev_raw, data_root, args.out / "after_dev")
    after_diagnostic_train = quality_run(model, tokenizer, diagnostic_train, data_root, args.out / "after_train_same_diagnostic")
    after_diagnostic_dev = quality_run(model, tokenizer, diagnostic_dev, data_root, args.out / "after_dev_same_diagnostic")
    gates = {"loss_equivalence": equivalence["passed"], "reload": reload_result["passed"],
             "requested_timing_steps_measured": len(timing) >= args.min_timing_steps,
             "dev_loss_improved": best_step is not None,
             "paired_diagnostic_train_recall_improved": after_diagnostic_train["metrics"]["fact_recall"] > before_train["metrics"]["fact_recall"],
             "train_fact_recall_at_least_half": after_train["metrics"]["fact_recall"] >= .5,
             "multiple_facts_generated": after_train["valid_multifact_calls"] > 0,
             "after_train_failures_zero": after_train["failed_generations"] == 0,
             "no_skipped_optimizer_steps": not any(r["optimizer_step_skipped"] for r in logs),
             "long_call_memory_checked": long_probe["passed"] if args.phase == "pilot" else True}
    report = {"phase": args.phase, "gates": gates, "main_sft_ready": all(gates.values()),
              "base_model": MODEL, "base_revision": REVISION, "dtype": args.precision,
              "dataset_manifest_sha256": sha(data_root / "manifest.json"),
              "training_steps": steps, "selected_checkpoint_step": best_step or steps,
              "training_elapsed_sec_including_dev_checks": elapsed_training, "timing_steps": len(timing),
              "requested_min_timing_steps": args.min_timing_steps,
              "initialization": config["initialization"], "loss_normalization": config["loss_normalization"],
              "seconds_per_microbatch_median": per_micro, "eta_main_one_epoch_sec": eta,
              "eta_note": "Measured up to last50 actual steps (count reported); one full epoch plus20% margin, estimate not guarantee.",
              "long_call_probe": long_probe, "before_train": before_train, "after_train": after_train,
              "before_dev": before_dev, "after_dev": after_dev, "reload": reload_result,
              "paired_diagnostics": {"before_train": before_train, "after_train": after_diagnostic_train,
                                     "before_dev": before_dev, "after_dev": after_diagnostic_dev},
              "full_official_val_run": False, "serving_confirmation_resolved": False}
    save(args.out / ("pilot.json" if args.phase == "pilot" else "sft_report.json"), report)
    export = args.out / "export"; shutil.copytree(selected, export, dirs_exist_ok=True)
    hashes = {str(p.relative_to(export)).replace("\\", "/"): sha(p) for p in sorted(export.rglob("*")) if p.is_file()}
    save(export / "model_manifest.json", {"artifact_state": "trained_adapter_reload_verified",
        "model_id": "qwen35-08b-case-" + args.phase, "manifest_id": sha(args.out / "training_config.json"),
        "base_model": MODEL, "base_revision": REVISION, "context": 16384, "max_new_tokens": 1536,
        "total_parameters": config["total_parameters"], "trainable_parameters": config["trainable_parameters"],
        "dtype": args.precision, "sha256": hashes, "training_quality_gates_passed": all(gates.values()),
        "serving_acceptance": "pending DE confirmation/time normalizers and real HTTP acceptance"})
    if args.phase == "main" and args.full_eval:
        quality_run(model, tokenizer, read(data_root / "evaluation/val.jsonl"), data_root, args.out / "official_val_after")
        report["full_official_val_run"] = True
        save(args.out / "sft_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        import sys
        if "--out" in sys.argv:
            failed_out = Path(sys.argv[sys.argv.index("--out") + 1])
            phase = sys.argv[sys.argv.index("--phase") + 1] if "--phase" in sys.argv else "pilot"
            record_failure(failed_out, phase, type(exc).__name__, RUN_CREATED)
        raise
