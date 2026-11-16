# Evaluation runner

Run from repository root, Python 3.12 environment. The official schema/evaluator/dataset must be obtained separately; no organizer files are bundled or modified.

```bash
python -m src.experiments.evaluate --gold /private/data/val.jsonl \
  --predictions /private/runs/A/predictions.jsonl --evaluator /private/data/case_metrics.py \
  --out experiments/A-eval --model-id qwen-A --experiment-id A-eval \
  --dataset-version official-val-v1 --model-config /private/config/model.json \
  --evaluation-config /private/config/eval.json
```

Config objects are saved verbatim for reproduction: use experiment-safe configuration without tokens/credentials. Evaluation config supports timeout_sec (default 300). Output directory must not exist. All outputs are private/ignored. Exit 2 on failure; manifest and validation report identify failure. No fallback facts are fabricated.

Alternatively replace --predictions with --factory package.module:build_pipeline. This trusted local factory receives model_config only, constructs the existing Pipeline once and must supply a matching model descriptor. The runner calls adapter.load() if provided, requires state=ready, and closes engine resources in finally (including on inference failure). Only explicitly whitelisted CallInput fields reach extractor and normalizers; never gold facts. Windowing, tokenizer and model parsing remain ML bridge responsibilities. --mode labels supplied predictions real/mock/synthetic/unknown; external predictions default unknown, not real.

Preflight checks strict JSONL/duplicates/IDs, official minimum fact fields, dates/intervals, polarity, attributes and original source references/spans/quotes. Source speaker can be inferred for minimum predictions. Optional quote/span are independently accepted per official schema. Present transcript metadata must match gold features. Gold annotation confirmed is accepted only for validating gold and never included in features. Known noisy gold quote/span alignment is not corrected or enforced; structural schema/source references are still checked.

Artifacts: predictions.jsonl, metrics.json (unmodified official output), manifest.json (config/git/input/output/evaluator SHA256), evaluation.log, validation_report.json. Successful empty facts are valid. Errors stop evaluation and return nonzero. Do not use dict loading to silently deduplicate predictions.

Published and recomputed baseline JSON may be attached with --published-baseline and --recomputed-baseline; they are stored separately with provenance/hashes, never substituted for run metrics. The provided baseline has at least one reversed interval; strict preflight correctly rejects it even though the permissive official script evaluates it. For reproducing organizer baseline numbers, run the unchanged script directly and attach its JSON as recomputed baseline; do not repair/relax model validation silently.

Local tests with official evaluator:
`OFFICIAL_EVALUATOR=/path/case_metrics.py pytest -q tests/test_evaluation_runner.py`.
CI tests the invocation protocol with a clearly labelled evaluator spy; the actual official-script test skips unless the external path is supplied.

Official val includes empty ASR turns. Turn.text therefore accepts an empty string; the turns collection remains nonempty and unique idx is preserved. Empty turns are not dropped or renumbered. This is an input-compatibility relaxation within v1, requiring DE/ML review; facts/value/object constraints remain unchanged.

Official val also contains 81 facts with optional object.type outside the Markdown enum (other). Gold validation checks essential evaluator fields and source references, records these annotations in gold_annotation_warnings, and passes the original gold file unchanged to the evaluator. Predictions with a supplied unsupported object.type remain invalid; models can use the official minimum object.name without type. No correction of annotations or evaluator is performed.

## Audit hardening

Prediction records reject unknown top-level fields; gold remains a separate allowlisted feature source. JSON rejects duplicate keys and non-finite numbers (including overflowing exponent literals). Factory configs are deep-copied: caller mutation cannot rewrite the recorded experiment. Dataset, prediction and evaluator hashes are checked again after subprocess completion; supplied aggregate counts must agree with preflight. Failed evaluator/input-integrity checks invalidate metrics.json instead of leaving success-shaped numbers. evaluation_config.expected_evaluator_sha256 optionally pins an approved official version before execution; the evaluator path is trusted local code, never an uploaded model artifact.

Artifact identity is recorded from model_config.artifact and effective public adapter metadata. Reported metadata still needs verification against exported files by the ML loader; generic infrastructure does not guess checkpoint identity.
