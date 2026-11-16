# Archived experiment results

Selected evidence from the official383 and temporal runs supplied on 16 November 2026. JSON and JSONL files are preserved byte-for-byte; PUBLISHED_RESULTS_MANIFEST.json records source ZIP hashes and the files published here.

## Official validation: 383 calls

- `official383/predictions.jsonl`: original partial-serving predictions, exactly 383 unique call IDs; no input turns or literal source quotes.
- `official383/official_val383/metrics.json`: original normalized/serving metrics, acceptance counts and timing scope.
- `official383/inference_config.json` and `generation_manifest.json` under `official_val383/`: model identity, adapter hash, dependencies and inference settings.
- `official383/baseline_recomputed.json`: organizer baseline, evaluator 2.1.
- The original final_report.json is already published as `../official383_original.json`.

The original serving result is precision 0.6736, recall 0.4363, temporal 0.3260 and state 0.4559. It does not pass the official quality gates. The separately documented tariff-catalog replay changes state to 0.4575; the predictions here are the original serving result, before that replay. See `../official383_catalog_replay.json` and `../FINAL_REPORT.md`.

To independently recompute the original result, obtain the unchanged organizer 2.1 package, keeping catalog.json next to val.jsonl, then run:

```sh
python3 /path/to/official/case_metrics.py /path/to/official/val.jsonl /path/to/this/directory/official383/predictions.jsonl
```

Expected evaluator SHA256: `863f87737dd1108e772250967f43697fa2411b3394c190d192b7ff66666259c3`.

## Temporal corrective experiment

`temporal/` contains the original training configuration, 256-update log, loss/token/reload checks, selection reports and per-checkpoint aggregate metrics. Selection retained the original parent adapter: selected_updates=0, selected_original_parent=true, observed_inner_improvement=false. This experiment is not the source of a newly improved final adapter. Its inner selection/guard metrics do not constitute an independent official validation result.

The selected adapter SHA256 is `577bc8184877de05386244791bee2ecaeb5dd894e86df749cd31ed270dbab98b`, shared with the official383 inference run. The model descriptor is preserved at `temporal/provenance/selected_model_manifest.json`. Model files named there are external artifacts and are not bundled here.

## Scope and storage

Both source ZIPs explicitly exclude model weights. No optimizer state, weights, tokenizer binaries, source transcripts, per-call raw generation records or full training payloads were added by this publication. The raw ZIPs remain part of the separate team artifact bundle.

Each `source_results_manifest.json` is the original inventory of its full source ZIP; some listed raw files are intentionally not published. Use PUBLISHED_RESULTS_MANIFEST.json for the exact inventory and hashes of this Git evidence subset. It does not hash itself or this explanatory README.

No training or model generation was performed while publishing these results. Recomputed metrics use the saved predictions only.
