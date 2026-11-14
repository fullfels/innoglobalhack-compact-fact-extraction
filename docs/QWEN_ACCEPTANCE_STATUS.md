# Qwen acceptance and final PR review — 2026-11-14

## PR #52: approved archive, not serving

Latest reviewed SHA: `4958f94deb8dcf6f6c6ba7c4649bffe1e17c6700`; fixes `81bc6858` + exact-LF prompt guard. [Official APPROVE](https://github.com/fullfels/innoglobalhack-compact-fact-extraction/pull/52#pullrequestreview-5904735356) submitted; previous provenance thread resolved. PR remains open, no merge.

Independently executed all 7 provenance unittest cases and verify_metrics with original official files. Added adversarial checks on a temporary archive copy: substantive mutation of predict.py, common.py, prompt.txt and notebook each makes verify_metrics nonzero; CRLF source/notebook accepted only under declared policy, CRLF prompt rejected. Historic source/predictions/metrics/run fingerprints are byte-identical to 9701bbbb. Both 10-call metric tables reproduce; exact first 10 original IDs and model-input allowlist remain intact. README correctly distinguishes Git hashes, executed bytes, manifest-relative equivalence and inaccessible private-original verification. Private ZIP/GPU authenticity not independently verified; not claimed.

This clears the original P2. Historical parser incompatibilities do not block an explicitly archival PR. T07/T08/T20 remain open; no new model inference has occurred.

## Current integration graph

| PR | Head at inspection | Base | State |
|---|---|---|---|
| #46 | 4bc3959 | main | open, CI success, no peer approval |
| #47 | 7323902 | main | open, CI success, no peer approval |
| #48 | 898477a | main | open, CI success, no peer approval |
| #49 | b74a3e1 | fs/evaluation-runner | open, CI success, no peer approval |
| #50 | **56303b0** after FS fix | fix/review-followups | open, [CI success](https://github.com/fullfels/innoglobalhack-compact-fact-extraction/actions/runs/43603183173), peer approval still required |
| #51 | 26f5c23 | fs/benchmark-collector | open, CI success, no peer approval |
| #52 | 4958f94 | main | open, CI success, approved archive |
| #53 | b95cfce | fs/deep-integration-audit | **new ML bridge, draft**, CI success |
| #54 | ed9e110 | ml/t20-qwen35-candidate-bridge-20261114 | **new LoRA pilot code, draft**, CI success; training not executed here |

No DE PR/normalizer implementation or local checkpoint/export/tokenizer/HF cache found. main remains c8911a99936de3d7a06c5327e55abcc5fa28694e. Separate FS branch `codex/qwen-acceptance` starts from #53 and includes latest #50 fix; all current #46–#51 heads are ancestors. No main merge, force push, ML/DE source edits or training.

Safe main merge order after peer approvals and separate owner permission: #46 → #47 → #48 → #49 → #50 → #51; #52 independent archive. Retarget dependent bases and repeat CI. #53/#54 remain draft dependencies, not accepted real deployment. They must pick up #50 metadata fix in a normal branch update; no ML branch was pushed by FS.

## Confirmed FS defect fixed

Real acceptance with actual QwenBridge and a controlled test engine failed: /model/info.manifest_id was nonnull, /extract.meta.manifest_id was always null. Reproduced first in offline ASGI acceptance and a standalone regression on #50. Minimal fix propagates the existing public artifact manifest ID into Meta, without changing schemas. Commit **56303b0** on fs/real-integration-readiness, pushed to #50; merged locally into the separate FS acceptance branch. Before fix: regression failed None != export-fixture-v1. After fix: #50 suite 84 passed/1 Docker opt-in skip; integrated full suite below includes Docker.

## Executable acceptance, using existing real_smoke

Existing HTTP real_smoke now also supports `--factory`: it invokes the trusted no-argument ML factory, checks expected descriptor/hash **before engine load**, then uses actual FastAPI lifespan/routes through TestClient. Lifecycle loads once and closes on exit. Existing schema/source/metadata/raw-model gates and prediction/manifest export are reused, not a second pipeline or a replacement parser. Offline transport is recorded as in-process-asgi. Requires development HTTP test dependencies; it does not import torch until ML loader runs.

New 8 FS tests use actual ML QwenBridge with controlled engines: multiple facts/Unicode source, load once and release, malformed JSON, unknown dates, speaker mismatch, invalid fact, generation failure/privacy, and identity/hash rejection before load. These are contract tests, not real model runs. Existing tests retain cancellation, 429 concurrency, timeout physical slot ownership, readiness and load failure coverage. CandidateBatch is revalidated by existing Pipeline after extractor and each normalizer; failed inference never becomes successful empty prediction. No automatic started_at fallback was added.

After ML hands off verified artifacts, from the repository root:

```bash
export MTS_MODEL_DIR='/private/approved-qwen-export'
export MTS_MODEL_DEVICE=cpu  # only an ML-verified profile; current runtime also documents exact cuda
export MWS_MODEL_ID='<model_manifest.json:model_id>'
export MWS_ARTIFACT_SHA256='<SHA256 of model_manifest.json, as ML runtime reports>'
export MWS_INPUT='/private/features-only-one-call.json'
export MWS_RUN='/private/new-run-directory'

# Phase A: local offline reload + actual ASGI API + release, no external HTTP server.
.venv/bin/python -m scripts.real_smoke --factory ml.qwen.runtime:factory \
 --input "$MWS_INPUT" --out "$MWS_RUN/offline" --model-id "$MWS_MODEL_ID" \
 --artifact-sha256 "$MWS_ARTIFACT_SHA256" --raw-model

# Phase B: a fresh process; one model session per API process.
MWS_MODE=real MWS_PIPELINE_FACTORY=ml.qwen.runtime:factory \
 .venv/bin/uvicorn src.api.app:app --host 127.0.0.1 --port 8000 --workers 1
# In a second terminal with the same identity/input exports:
.venv/bin/python -m scripts.real_smoke --url http://127.0.0.1:8000 \
 --input "$MWS_INPUT" --out "$MWS_RUN/http" --model-id "$MWS_MODEL_ID" \
 --artifact-sha256 "$MWS_ARTIFACT_SHA256" --raw-model
```

Do not invent the expected hash/identity or mark an untrained artifact trained. Current ML runtime requires model_manifest.json with trained_adapter_reload_verified and actual passed training gates, pinned base revision and export hashes; base weights must exist offline in HF cache, plus adapter/, tokenizer/, prompt.txt and catalog.json. The effective artifact hash is now the ML runtime's model_manifest.json hash, not the older optional files.sha256 handoff convention. Install only reviewed engine dependencies/profile. Until these files exist, do not execute the example as a supposed real run. No placeholder values were executed against a real model.

Without DE normalizers this is explicitly raw-model acceptance, not a finished semantic pipeline; unconfirmed agreements are filtered with warnings. With approved DE stages remove --raw-model and record exact postprocessing/catalog revision. Stop/drain the API before offline evaluation to avoid a second GPU session. HTTP timeout cannot interrupt hung native inference; release may still wait for physical completion.

## Remaining concrete ML issues / dependencies

Recorded in [PR #53 comment](https://github.com/fullfels/innoglobalhack-compact-fact-extraction/pull/53#issuecomment-6878481702); ML files not changed:

- artifact_manifest accepts an empty sha256 inventory. Reproduced without torch using correct base/state/revision and sha256={}; it returns successfully without adapter/tokenizer files. Require nonempty inventory covering loaded adapter/tokenizer/prompt/catalog files, strict bool gates, and mutation/missing-file tests. An unlisted existing file is otherwise outside verification.
- Loader has no explicit cleanup if it fails after allocating model/session but before returning LoadedModel (including startup smoke). ManagedAdapter can only release returned handles. ML must guard partial allocations.
- Device handling distinguishes exact 'cuda' from 'cpu'; explicit cuda:N currently selects FP32 paths. Benchmark collector requires explicit cuda:N. Agree device/dtype mapping before real GPU measurements; CPU/CUDA profile must be validated by ML.
- Only API factory() is supplied. Evaluation runner requires factory(config); collector requires factory(config/device) → BenchmarkTarget with measured load + close. Supply thin approved entrypoints or an FS-reviewed mapping that actually applies/pins the recorded config; passing factory() directly to evaluate would be a signature error. Do not silently ignore configuration.

## Actual checks

- Integrated pytest with OFFICIAL_EVALUATOR and MWS_DOCKER_TESTS=1: **183 passed, 0 skipped**, 5.86 s, two upstream deprecation warnings.
- Node frontend tests: **5 passed**.
- PR #52 existing provenance tests: **7 passed**, plus 4 independent mutations and 4 EOL-policy checks; original metrics verifier exit 0. No private-original ZIP tests claimed.
- Original evaluator/comparison integration and ML synthetic bridge tests ran in full suite; no gold passed to inference fixtures. Existing archived metric recomputation is not newly generated Qwen output.
- Production mock Docker build/startup/HTTP smoke and Compose checked separately; real image still needs ml/qwen + reviewed torch/transformers/peft dependencies, offline base/export and DE wiring. Existing Dockerfile intentionally remains mock-ready; no heavy dependency downloads/model downloads/training performed.
- Actual real checkpoint reload/HTTP, 10/383 model calls, GPU memory/latency and quality gates: **not measured**, artifacts absent. T20 not done.

## Next five actions

1. Accept #52 archive with explicit owner merge permission; obtain independent peer reviews for FS PRs, then merge in dependency order and repeat CI.
2. ML completes artifact inventory/partial cleanup/device profile and supplies config-aware evaluation/benchmark entrypoints in #53; run its mutation/loader tests.
3. ML executes the controlled #54 pilot/reload and hands off genuine base/adapter/tokenizer/export manifest plus dependency lock and offline instructions. FS does not train it.
4. DE hands off ordered catalog/time/state/polarity/grounding normalizers, catalog hash and fixtures (ambiguity, negation/confirmation, year boundaries, historical state/multiobject, Unicode offsets).
5. Run offline acceptance → real HTTP → diagnostic 10 → full 383 with original evaluator, keeping predictions.jsonl/metrics.json/manifest.json/validation_report.json; then comparable A/B/C only with verified configurations and resource budget.
