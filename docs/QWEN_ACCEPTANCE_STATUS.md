# Qwen acceptance and final PR review — 2026-11-14

## Latest recheck: #53 b95cfce / #54 2a3be19 / #55 7e42f93

No trained model package was found. GitHub Actions artifacts: 0; no repository releases, local export/checkpoint/tokenizer or HF cache. Eduard-Sadykov is now confirmed by the user and GitHub API as an existing write collaborator; CODEOWNERS/team handoff updated without permission expansion/invitations. DE normalizers still not supplied.

The five previously reported ML handoff gaps are **not fixed** on latest #53; inherited runtime in #54 also retains them. New [REQUEST_CHANGES on #53](https://github.com/fullfels/innoglobalhack-compact-fact-extraction/pull/53#pullrequestreview-5905082280) and [interface review on #54](https://github.com/fullfels/innoglobalhack-compact-fact-extraction/pull/54#pullrequestreview-5905082718) record independent probes. No ML loader/training edits were made.

| Readiness gate | #53 b95cfce | #54 2a3be19 |
|---|---|---|
| Empty/partial checksum inventory and non-bool quality gate rejected (3 cases) | FAIL | FAIL |
| Partial adapter-load failure releases allocated sentinel before error propagation | FAIL | FAIL |
| cuda:0 preserves indexed CUDA dtype semantics | FAIL | FAIL |
| Config-aware evaluation entrypoint supplied | absent | absent |
| BenchmarkTarget/config/device/load/close entrypoint supplied | absent | absent |

These are **14 failed readiness checks**, separate from the **183 passing default regression tests**. They are not GPU measurements. The controlled allocation test retains the loader error and verifies a weakref release callback: resource remains held by the failed-load traceback, confirming the missing cleanup path. CUDA uses stub torch/transformers only; no GPU availability/performance claims. #54 improves dtype propagation for exact cuda but leaves cuda:N handling unresolved. Latest 2a3be19 changes notebook numerics gates, not these runtime defects. Four latest #54 lightweight training guard tests passed; 5 Python notebook cells compiled without execution. No masking/hyperparameter/training-loop changes were made.

Reproducible checks are FS-owned diagnostics in `checks/qwen_runtime_readiness.py`, explicitly outside the default testpaths=tests suite:

```bash
# Current #55 contains runtime #53; expected failure until ML fixes its handoff.
PYTHONPATH="$PWD" python -m pytest -q checks/qwen_runtime_readiness.py
# To check another immutable runtime without switching/rewriting ML branches:
git show 2a3be19:ml/qwen/runtime.py > /tmp/qwen-runtime-reviewed.py
MWS_REVIEW_RUNTIME_FILE=/tmp/qwen-runtime-reviewed.py PYTHONPATH="$PWD" \
 python -m pytest -q checks/qwen_runtime_readiness.py
```

Entry names evaluation_factory/benchmark_factory reflect the requested handoff; if ML chooses other names, supply the approved module:function paths and point the checks at those implementations. A metadata-only wrapper that ignores config is not an acceptable fix. No failing readiness gate is marked passed by the green generic CI. After fixes, integrate these contracts into ML's required tests.

Repeated current #55 default suite with original evaluator/Docker sentinel: **183 passed, 0 skipped**, 6.46 s; Node **5 passed**; Docker mock build/startup/HTTP/Compose **PASS**. Existing #50 manifest_id fix is present and its regression passed; no new FS runtime defect was confirmed. No duplicate real_smoke/API/factory/benchmark/evaluator was added. The previous accepted #52 archive and snapshots below remain historical evidence.

Remaining operational order is unchanged: peer reviews + explicit owner merge permission for FS; ML fixes its owning runtime and provides verified export/dependencies/factories; Eduard-Sadykov hands off normalizers/resolvers/semantic fixtures; then offline reload → in-process API → real Uvicorn HTTP → diagnostic 10 → full 383. The runtime currently requires a genuinely reload-verified trained adapter and passed training gates. Do not forge these flags to run a base-only model. No new real inference/quality/latency/RAM/VRAM or T20 completion is claimed.

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
| #54 | **2a3be19** (latest recheck) | ml/t20-qwen35-candidate-bridge-20261114 | **LoRA pilot/numerics code, draft**, CI success; training not executed here |

No DE PR/normalizer implementation or local checkpoint/export/tokenizer/HF cache found. main remains c8911a99936de3d7a06c5327e55abcc5fa28694e. Separate FS branch `codex/qwen-acceptance` starts from #53 and includes latest #50 fix; all current #46–#51 heads are ancestors. No main merge, force push, ML/DE source edits or training.

Acceptance is published as draft [PR #55](https://github.com/fullfels/innoglobalhack-compact-fact-extraction/pull/55), base #53. Code head 090a859 CI [43603644521](https://github.com/fullfels/innoglobalhack-compact-fact-extraction/actions/runs/43603644521) and separate PR run 43603760706 succeeded: 181 Python passed/2 external official-file tests skipped, 5 JS passed, Docker/HTTP PASS. Local original-file tests all ran. Subsequent report-only update does not change acceptance code.

Latest #54 7fffa46 changes the proposed pilot default to FP32 and adds numerics_probe after ML-reported FP16 gradient overflows/dev-loss regression. These are reported observations, not independently repeated GPU measurements here. Its runtime now accepts manifest dtype; acceptance snapshot intentionally stays on #53 b95cfce and does not silently import draft training changes. ML must hand off the approved runtime version matching the actual export dtype and a genuinely successful numerical/pilot/reload result. No successful training/export is inferred from green code-only CI.

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
