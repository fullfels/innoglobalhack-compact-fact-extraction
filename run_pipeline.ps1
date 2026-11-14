New-Item -ItemType Directory -Force -Path "data/train_ready_v1" | Out-Null

 = "scripts"

Write-Host "=== Step 1: Audit raw data ===" -ForegroundColor Cyan
python scripts/audit.py data/train.jsonl data/val.jsonl data/baseline_pred.jsonl catalog.json

Write-Host "
=== Step 2: Analyze train patterns ===" -ForegroundColor Cyan
python scripts/patterns.py data/train.jsonl data/val.jsonl data/baseline_pred.jsonl catalog.json

Write-Host "
=== Step 3: Test fast postprocess baseline ===" -ForegroundColor Cyan
python scripts/postprocess_test.py data/val.jsonl data/baseline_pred.jsonl catalog.json data/fixed_baseline_pred.jsonl

Write-Host "
=== Step 4: Clean train and parse dates ===" -ForegroundColor Cyan
python scripts/prepare_data.py data/train.jsonl data/val.jsonl catalog.json data/train_clean.jsonl

Write-Host "
=== Step 5: Full postprocessor and state chains ===" -ForegroundColor Cyan
python scripts/full_postprocess.py data/val.jsonl data/baseline_pred.jsonl catalog.json data/final_fixed_pred.jsonl

Write-Host "
=== Step 6: Build SFT dataset and ASR augmentation ===" -ForegroundColor Cyan
python scripts/build_training_sets.py data/train_clean.jsonl data/val.jsonl data/train_ready_v1/sft_train.jsonl data/train_ready_v1/sft_val.jsonl data/train_ready_v1/val_adv.jsonl

Write-Host "
=== ALL STEPS SUCCESSFULLY COMPLETED! ===" -ForegroundColor Green
