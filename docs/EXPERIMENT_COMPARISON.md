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
