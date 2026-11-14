# Pipeline benchmark collector (T26)

```bash
python -m src.experiments.benchmark --factory team.module:build_benchmark_target \
 --inputs /private/data/benchmark-inputs.jsonl --warmup-inputs /private/data/train-derived-warmup.jsonl \
 --config /private/config/inference.json --out experiments/A-cpu-bench --experiment-id A-cpu-bench \
 --device cpu --repeats 3 --cold-trials 3
```

The factory receives config + explicit device and returns BenchmarkTarget(pipeline, device, synthetic=False, parameters=None, checkpoint_paths=(), close=...). Real factories are supplied by ML, not invented here. Test fakes must set synthetic=True. A mock descriptor always marks the benchmark synthetic. No inference/training code is bundled.

Cold means a new target instance + load + first pipeline call, not a new OS process or flushed disk cache. Three trials by default, load and first-call durations saved separately. Use a fresh process for each published configuration/run; do not present in-process cold trials as cold-process measurements. Warmup is explicit and separate (use train-derived calls), measured calls batch=1/concurrency=1. A/B/C must use identical inputs, repeat count, warmup and device; throughput excludes load/warmup and includes wall time/errors of the measured sequential loops.

Artifacts: benchmark_raw.jsonl (load/cold/warmup/warm/loop durations, errors, sampled process RSS and resource peaks), benchmark_summary.json, hardware.json, benchmark_manifest.json. Raw durations recalculate mean/median/nearest-rank p95 and successful calls per measured wall second/hour. Success latency excludes failures; error rate and schema percentage retain them. No silent fallback predictions. RSS is sampled current-process memory, not a process tree, allocator increment or OS-wide total. Short peaks between samples may be missed. CUDA reserved peak is PyTorch allocator memory, not NVML/device-wide/process VRAM. Parameters unknown => null, not guessed; checkpoint paths must include all required base+adapter/tokenizer files.

--device cuda:N imports installed torch only and synchronizes before/after measured work. Missing CUDA records not_measured without constructing the model. CPU runs record VRAM/GPU-seconds not_measured. Only --exclusive-gpu permits one-allocated-GPU wall seconds/1000 calls; this is allocation time including CPU phases, not CUDA kernel time. Do not assert exclusive allocation on shared GPU. MPS is not labelled CPU/CUDA: unsupported devices fail clearly.

Config must exclude credentials; outputs are private/ignored. No Qwen resource figures have been produced. Tests use synthetic adapters and controlled clocks. Target.close must release engine resources between trials; reuse of an external singleton would invalidate cold measurements. A real engine benchmark needs ML to honor target.device and supply shutdown/synchronization-compatible resources.

## Audit hardening

Success now requires a prediction for the exact original CallInput plus valid grounding. Returning a coroutine is an error, not completed inference; async engines must be awaited by the synchronous ML bridge. Every phase contributes to phase_errors; failed cold/warmup/warm samples make benchmark_valid=false and CLI exit 2, while raw diagnostic samples remain available. Comparison refuses these runs. Close callbacks execute once, even on failure; cleanup errors invalidate manifest/summary. Target identity, artifact metadata, parameters and checkpoint size cannot silently change between cold trials. Declared missing checkpoint paths are errors; absent metadata remains unknown. RSS subprocess sampling has a bounded timeout. Configs are copied before each builder call.
