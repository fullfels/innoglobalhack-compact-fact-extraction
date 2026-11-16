# Real Qwen runtime: packaging READY, model acceptance BLOCKED

This is a separate FS-owned profile. Default Dockerfile/Compose and existing API/real_smoke are unchanged. No weights, engine packages, private data or new parser are bundled/downloaded. Do not deploy the inherited ML runtime until the serving blockers in #53 are closed.

## Runtime build and startup

`Dockerfile.qwen` requires a **reviewed Python 3.12 engine image pinned by digest**, containing exact FS requirements.lock versions, exact ML requirements.txt versions and the ML-approved Torch distribution. FS does not choose or install an untested CUDA/Torch build. The image copies only required ML runtime source files, runs UID/GID 10001, and executes the existing one-worker Uvicorn API. The build has no pip/model download steps.

Provide actual values from QFa1 after export/profile validation:

```bash
export MWS_ENGINE_IMAGE='<approved-registry-image>@sha256:<digest>'
export MWS_TORCH_VERSION='<exact-installed-version-in-that-image>'
export MWS_MODEL_DIR='/private/approved-export'
export MWS_HF_CACHE_DIR='/private/dedicated-offline-hf-cache'
export MWS_MODEL_ID='<actual-model-id>'
export MWS_ARTIFACT_SHA256='<sha256-of-model_manifest.json>'
docker compose -f compose.qwen.yaml config --quiet
docker compose -f compose.qwen.yaml build
docker compose -f compose.qwen.yaml run --rm --no-deps qwen --check-only
# Only after ML gates and the approved package are ready:
docker compose -f compose.qwen.yaml up -d
```

The dedicated cache must contain the pinned base revision and required blobs with valid local symlinks; do not mount a credentials-bearing personal HF_HOME. Export requires model_manifest.json, adapter/, tokenizer/, prompt.txt and catalog.json. ML owns complete inventory/base revision/training/reload validation; packaging checks do not substitute for these. Never forge training_quality_gates_passed or artifact_state. The profile mounts export/cache read-only and refuses to create missing host directories. No raw dataset is mounted into serving.

HF_HOME points at the supplied cache, HF_HUB_OFFLINE=1 is set before ML imports. [Official HF environment reference](https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables) defines these settings. Network access to the Hub is disabled; a missing local model fails rather than downloading a replacement. The profile itself is not a general network sandbox.

`qwen_runtime_preflight` checks paths/layout, expected manifest bytes/hash/model ID and installed distribution versions without importing Torch or loading a model. Missing/malformed/changed manifest, wrong identity, missing packages/versions and unsupported configuration produce structured blocked diagnostics and exit 2 before Uvicorn. `packaging_preflight_passed` explicitly reports model_loaded=false/model_readiness_verified=false. Native load/readiness/error/release remains with ManagedAdapter and ML; healthy readiness is established only by the actual API lifecycle and acceptance.

Current supported config values are cpu and exact cuda. cuda:N is intentionally refused until ML fixes its indexed-device mapping. Optional GPU syntax:

```bash
docker compose -f compose.qwen.yaml -f compose.qwen.gpu.yaml config --quiet
# On verified NVIDIA hardware only; not executed as a real GPU deployment here:
docker compose -f compose.qwen.yaml -f compose.qwen.gpu.yaml up -d
```

GPU image/driver/toolkit/device/dtype compatibility is still BLOCKED pending ML profile and actual measurements. CPU startup with real weights is also unverified. A 2-minute stop window permits graceful drain but is not a native inference timeout guarantee; forced kill is not successful cleanup. Keep a tested native stop/restart policy.

After startup use the **existing** `python -m scripts.real_smoke --url http://127.0.0.1:8001 ... --raw-model` with approved model ID/hash and features-only input. `--raw-model` remains required while DE stages are absent. Existing `--factory` acceptance provides offline reload/in-process API in a reviewed host environment; no alternate tool/API was created. Stop/drain the HTTP process before offline evaluation/benchmark.

## READY / BLOCKED and owners

| Stage | Status | Owner / blocker |
|---|---|---|
| Shared contracts, lifecycle, errors, timeout/slot, manifest_id | READY in controlled tests | FS; retained existing fixes |
| Docker packaging + CPU/GPU Compose syntax | READY for handoff; native startup BLOCKED | FS profile; QFa1 must provide approved engine image/export |
| Artifact inventory, failed-load cleanup, cuda:N | BLOCKED | [QFa1, #53 review](https://github.com/fullfels/innoglobalhack-compact-fact-extraction/pull/53#pullrequestreview-5905082280) |
| Config-aware evaluation / BenchmarkTarget factories | BLOCKED | QFa1, #53; configs must actually apply |
| DE temporal/confirmation resolvers and ordered normalizers | BLOCKED | Eduard-Sadykov; no modules/fixtures delivered |
| Offline reload / real HTTP / diagnostic 10 / full 383 | BLOCKED | Verified package + ML gates; only raw-model until DE handoff |
| Official evaluation / resource collectors / comparison tooling | READY synthetically; real results BLOCKED | Runtime/artifacts/hardware unavailable |

Seven ML readiness cases remain failing against current #53 and inherited #54 runtime; do not equate the green default CI with model readiness. No new real metrics, checkpoint acceptance, GPU timings or T20 completion are claimed.

## PR snapshot and merge gates

Checked main c8911a99936de3d7a06c5327e55abcc5fa28694e unchanged; no merges performed. Source SHA before this packaging commit:

| PR | SHA | Dependency | Review / CI / merge readiness |
|---|---|---|---|
| #46 | 4bc3959 | main | CI success; needs peer review + owner permission |
| #47 | 7323902 | main | CI success; needs peer review + owner permission |
| #48 | 898477a | main | CI success; needs peer review + owner permission |
| #49 | b74a3e1 | #48 | CI success; needs parent/peer review + permission |
| #50 | 56303b0 | #47 | CI success, manifest_id fix; needs peer review + permission |
| #51 | 26f5c23 | #49 | CI success; needs parent/peer review + permission |
| #52 | 4958f94 | main, independent archive | APPROVED, CI success; owner merge permission pending |
| #53 | b95cfce | FS audit snapshot | draft/CHANGES_REQUESTED; successful runs and one cancelled run, serving blockers open |
| #54 | 8534de3 (final check) | #53 | draft, CI success; training-selection update, runtime byte-identical to cc439ff |
| #55 | 65843e0 baseline, plus this profile | #53 + #50 fix | draft, CI checked after push; model acceptance blocked |

No new PRs were present at initial check. Safe order after approvals/permission: #46 → #47 → #48 → #49 → #50 → #51; #52 independently. Retarget children/recheck CI; then close #53 blockers and update #54/#55 bases by ordinary merges without force push. Do not merge drafts because code-only CI is green. Full handoff commands/tests: QWEN_ACCEPTANCE_STATUS.md and FIRST_REAL_INFERENCE.md.

## Executed tests

13 packaging metadata tests and a real Docker negative-profile test passed: absent model dir, corrupt manifest, changed hash and deliberately missing ML dependencies all exit 2. The negative image uses the existing mock base **on purpose**, no Torch/model downloads; it is not a working real image. CPU/GPU Compose configuration was validated with explicit config-only placeholders, without running a GPU container. Full local suite with original evaluator and Docker sentinel: **197 passed, 0 skipped**, two upstream deprecation warnings; frontend **5 passed**. Latest #54 lightweight training guards/result-analysis tests: **11 passed**, no training executed. Separate ML readiness checks remain failing (7 per inspected runtime), not included as passing acceptance in default CI. Remote CI is checked after push. Default mock startup/HTTP remains a separate smoke, not real Qwen acceptance.

The trained checkpoint/export/tokenizer and DE package were not found in the checked local locations. Failed inference still returns an error rather than a successful facts=[]; failed/null results and successful empty results must stay distinct when actual 10/383 runs become possible.

During final verification #54 advanced from cc439ff to 8534de3. Its serving runtime is byte-identical to the tested cc439ff version; all seven runtime blockers remain applicable. The delta changes training checkpoint selection, not loader/factory/export interfaces; no training code was changed by FS. Latest lightweight guards/result-analysis tests: **12 passed** separately (the earlier 11-case result above is for cc439ff). #55 packaging code head 98aceeb passed both remote runs, including [43612372193](https://github.com/fullfels/innoglobalhack-compact-fact-extraction/actions/runs/43612372193): 195 passed/2 external evaluator tests skipped, frontend 5/5, Docker/HTTP PASS. This subsequent report-only update does not change packaging/runtime code.
