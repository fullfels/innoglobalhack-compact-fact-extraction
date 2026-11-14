# Comparable A/B/C reports

Install optional analysis dependencies: `pip install -r requirements-analysis.txt`.

```bash
python -m src.experiments.compare --gold /private/data/val.jsonl --evaluator /private/data/case_metrics.py \
 --run A=experiments/A-eval --run B=experiments/B-eval --run C=experiments/C-eval \
 --benchmark A=experiments/A-cpu --benchmark B=experiments/B-cpu --benchmark C=experiments/C-cpu \
 --out experiments/ABC-cpu-report
```

Evaluation bundles must be completed, share dataset/evaluator hashes, pass preflight and match official recomputation. Resource bundles are optional: unknown values remain null, never invented. Present resource bundles must have matching model IDs, hardware/device and identical batch/repeat/input/warmup protocol. Raw/summary hashes and latency recomputation are checked; benchmark input features must agree with the evaluation dataset. CPU and CUDA comparisons must be separate. Model configs may differ; INT8/INT4 are not assumed. Missing GPU measurements are not zero-cost claims.

Outputs: comparison.csv/json; pairwise.json for every unordered pair (delta_pp=100*(B-A), byte changes/reductions, median-latency speedup); fact_errors.jsonl; errors_by_attribute.json; pairwise_facts.jsonl using gold indices, not prediction array positions. Official match_facts determines matches. Unmatched gold/prediction remain missing/extra. Auxiliary polarity/value near-misses reuse official object matching but do not change official totals. Temporal day matching, span intersection and per-attribute state error use the supplied unchanged evaluator's semantics. Known evaluator object/state limitations remain unchanged; this report is not independent semantic adjudication.

Five graphs plus the CSV table: fact recall × latency; fact recall × peak RAM; temporal × peak RAM; state × peak RAM; fact-recall/latency Pareto (min latency, max recall). Axes have explicit metric/unit names. Missing coordinates are omitted with a not-measured annotation. Synthetic/mock inputs are labelled in graph titles and never presented as Qwen experiments. --no-plots creates tabular/error reports without matplotlib. Empty input does not generate fake results.

Pipeline benchmark now saves feature digests and summary hash for comparison provenance. Configurations should run in fresh isolated processes with shared hardware conditions, the same train-derived warmup and full-val quality. Statistical confidence intervals and semantic manual error review remain team tasks. No actual A/B/C model runs are claimed.

## Audit hardening

Quality/resource pairing requires identical per-run model config and artifact identity, compatible mode/device, benchmark_valid=true, and complete cold/warmup/warm raw protocol. Peaks are checked against raw resource records and sampled RSS; impossible or contradictory CPU/GPU measurements are rejected. Real runs additionally require model_config.device, a 64-hex artifact.hash, and evaluation_config.comparison_protocol with input_policy and postprocessing identifiers. That protocol/postprocessing must match across configurations; model/precision configuration itself may differ between A/B/C. Legacy incomplete resource bundles must be re-run, not patched into apparently verified measurements.

fact_counts.json explicitly records official TP/FP/FN by call and attribute; matched temporal/grounding errors remain TP for fact precision/recall, not full semantic correctness. span_evaluated distinguishes unavailable optional span scoring. Auxiliary near-misses do not alter official counts. CI-only evaluator_protocol_double exercises rejection gates without shipping organizer code; actual official scoring is tested separately with OFFICIAL_EVALUATOR.
