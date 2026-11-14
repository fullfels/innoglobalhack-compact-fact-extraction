# Первый Qwen inference: проверяемая передача и запуск

Этот runbook использует существующие ManagedAdapter, Pipeline, FastAPI и experiment CLI. Конкретный Qwen loader/bridge поставляет ML Engineer. Пока артефактов нет, команды real inference не выполняются, T20 остаётся открытой. Конфигурации, веса, входы и результаты хранятся вне Git.

## 1. Проверенный код и merge gate

Совместная ветка: `fs/deep-integration-audit`; включает #46–#51. SHA и результаты: [INTEGRATION_STATUS](INTEGRATION_STATUS.md). Для локального smoke можно использовать эту ветку, не меняя main. Для командного принятия нужны peer review и явное разрешение владельца на merge. Технический аудит и зелёный CI не заменяют peer approval.

Безопасный последовательный порядок: #46 → #47 → #48 → #49 → #50 → #51. После #47 retarget #50 на main; после #48 retarget #49; после #49 retarget #51. До merge каждого дочернего PR проверить актуальный base/head, diff, peer approval, конфликты и новый CI. Сохранить общую историю обычным merge, без force push. Проверочная ветка не является вторым umbrella PR.

## 2. Минимальная передача от ML Engineer

Передать приватный локальный bundle или утверждённое хранилище, не веса через Git:

```text
qwen-export/
  checkpoint/              # все shards/config; либо base + PEFT adapter вместе
  tokenizer/               # vocabulary/config/special tokens/chat template
  files.sha256             # SHA256 engine assets, относительные пути
  handoff.json             # публичная идентичность, provenance и команды
  model.json               # experiment-safe inference config, без credentials
  evaluation.json          # evaluator hash + comparison protocol
  requirements-real.lock   # версии engine/runtime; проверенный профиль CPU/CUDA
```

Названия checkpoint/tokenizer файлов определяет настоящий engine. Ни одна отсутствующая часть не подменяется скачиванием большой модели. `files.sha256` покрывает все shards/base/adapter/tokenizer, их engine configs и runtime lock. Не включать сам files.sha256, handoff.json или experiment configs model.json/evaluation.json, которые ссылаются на aggregate hash: это создало бы циклический hash. Experiment configs отдельно сохраняются runner в manifest. Относительные пути остаются внутри bundle, без внешних symlinks. Проверка до исполнения bridge:

```bash
cd /private/qwen-export
shasum -a 256 -c files.sha256
shasum -a 256 files.sha256
```

Для этой передачи второй hash используется как aggregate `artifact.hash` в private config и `artifact_sha256` ManagedAdapter. Это convention передачи, не новая схема Qwen и не изменение ModelResources. При ином уже согласованном ML manifest явно указать, какие неизменные bytes хэшируются; во всех компонентах использовать один и тот же hash.

Минимальное содержание handoff.json (значения в угловых скобках должны быть заменены реальными):

```json
{
  "model_id": "<public-model-id>",
  "revision": "<immutable-checkpoint-or-export-revision>",
  "artifact_sha256": "<64-hex-aggregate-hash>",
  "license": "<license-and-provenance>",
  "engine": "<actual-engine-and-version>",
  "device": "<cpu-or-cuda:N>",
  "dtype": "<actual-dtype-or-quantization>",
  "context_tokens": null,
  "parameters": null,
  "manifest_id": "<immutable-export-id>",
  "api_factory": "<approved-module>:<api_factory>",
  "evaluation_factory": "<approved-module>:<evaluation_factory>",
  "benchmark_factory": "<approved-module>:<benchmark_factory>"
}
```

`context_tokens` здесь placeholder: передать фактическое положительное значение; неизвестные параметры — null, не выдуманные нули. Дополнительно нужны: deterministic decode/seed, input/window/overlap policy, truncation/context overflow policy, stop tokens, offline reload command, release callback, fatal/OOM/restart policy, один features-only smoke JSON и ожидаемые shape/source assertions. Не требуется заранее обещать semantic gold answer.

Bridge code/dependencies должны быть проверены как доверенный код команды. Не включать произвольный remote code/pickle из непроверенного артефакта. Реальную сигнатуру engine FS не определяет.

### Общие интерфейсы и три точки сборки

- `Extractor.extract(CallInput) -> CandidateBatch`; bridge `(CallInput, ModelResources) -> CandidateBatch` сохраняет исходные idx/speaker/text и не читает gold.
- `ModelResources(model_path, tokenizer, inference_config)` хранит приватные ресурсы. Loader `(ModelResources) -> LoadedModel(inference=bridge, close=release)` возвращает завершённую загрузку. Если падает до return, сам очищает частичные аллокации.
- `ManagedAdapter(..., descriptor=ModelDescriptor(model_id, 'real', engine), revision=..., artifact_sha256=..., dtype=..., context=..., parameters=..., manifest_id=...)` публикует только публичные metadata. Ready устанавливается после успешной загрузки, release идёт после физического inference.
- `api_factory()` без аргументов возвращает `Pipeline(adapter, ordered_normalizers)`. **Не загружать повторно на запрос:** lifespan вызывает adapter.load один раз.
- `evaluation_factory(config)` возвращает Pipeline; существующий runner сам вызывает load, требует ready и делает close в finally.
- `benchmark_factory(config)` возвращает `BenchmarkTarget(pipeline, device, synthetic=False, parameters=..., checkpoint_paths=(...), close=adapter.close)`. **Эта factory включает adapter.load до return:** именно её вызов измеряется как load. Каждый cold trial создаёт новый adapter/session; singleton недопустим. Device приходит в config и обязан соблюдаться. Checkpoint paths включают base+adapter/tokenizer, а не только маленький adapter.

Один ML construction helper может обслуживать эти три входа; отдельные pipelines/схемы не нужны. Bridge синхронный и возвращает завершённый CandidateBatch, не coroutine/future; engine-specific async/CUDA completion обеспечивает ML.

CandidateBatch содержит `candidates`, `warnings`, `status='ok'`; CandidateFact — `fact`, `object_key?`, `evidence_turns`, `temporal_expression?`, `confirmation`, `provenance`, `warnings`. Полный Fact и enum — в `src/contracts/models.py`. Ошибки load/inference/decode/context/OOM — PipelineError с module/code/error_type, не успешный пустой batch. Пустой успешный результат допускается.

## 3. Передача от DE

Нужны реализации `Normalizer.normalize(CandidateBatch, CallInput) -> CandidateBatch` в порядке catalog → temporal → state → polarity → grounding. Catalog и его hash передаются при сборке, без mutable global state между calls. Передать fixtures aliases/ambiguity, даты и границы года, отрицания/agreements, historical state transitions/multiobject и Unicode source.

Rejected не экспортируется; unknown допускается обычным утверждениям, но agreement требует confirmed и client evidence принятия. Семантическую проверку принятия выполняют ML/DE. Temporal anchor — started_at; сохранять timezone, не выдумывать точную дату при ambiguity. State reconciliation закрывает только несовместимое состояние того же объекта, сохраняя историю. Source.turn_idx существует, speaker совпадает; span — Python code points `[start,end)`, quote точно равен срезу. Для API при ненадёжной привязке span/quote оба null. Revalidation выполняется после каждого этапа.

Если DE stages отсутствуют, ML factory возвращает Pipeline с пустым tuple. Для такого прогона **явно использовать raw-model experiment**: `--raw-model` в HTTP smoke, `experiment_id` с raw-model, `model_config.postprocessing='raw-model/no-de/v1'` и такой же identifier в evaluation protocol. Mode остаётся real только при настоящем engine. Это не завершённый DE pipeline и не выполнение T20.

## 4. Один запрос через FastAPI

После проверки checksums и peer review ML code создать окружение с переданными engine dependencies. Из корня repo задать реальные значения:

```bash
export MWS_MODE=real
export MWS_PIPELINE_FACTORY='<approved-module>:<api_factory>'
export MWS_MODEL_ID='<public-model-id>'
export MWS_ARTIFACT_SHA256='<64-hex-aggregate-hash>'
export MWS_RUN_DIR="$PWD/experiments/first-qwen-<unique-run-id>"
export MWS_GOLD='/private/organizer/val.jsonl'
export MWS_EVALUATOR='/private/organizer/case_metrics.py'
export MWS_MODEL_CONFIG='/private/qwen-export/model.json'
export MWS_EVAL_CONFIG='/private/qwen-export/evaluation.json'
export MWS_EVAL_FACTORY='<approved-module>:<evaluation_factory>'
export MWS_BENCH_FACTORY='<approved-module>:<benchmark_factory>'
.venv/bin/uvicorn src.api.app:app --host 127.0.0.1 --port 8000 --workers 1
```

`MWS_MODEL_ID` — ожидаемое имя для проверки, descriptor остаётся источником истины. Factory/loader получают приватную конфигурацию через согласованный ML механизм. Production image пока содержит только backend dependencies: до real Docker ML должен предоставить engine/runtime profile и проверенную сборку; одного /models mount недостаточно. В Compose сейчас не передаётся MWS_PIPELINE_FACTORY — для real deployment явно добавить его и engine module/config через private override либо утверждённый real image. Не помещать credentials в Docker context.

В другом терминале из repo (с теми же export): подготовить deterministic первые 10 calls. Это локальная копия исходных records для smoke, не изменение split/train/val. Проверить официальный hash до копирования.

```bash
.venv/bin/python - <<'PY'
import json, os
from pathlib import Path
from src.experiments.common import rows
from src.experiments.evaluate import features
root=Path(os.environ['MWS_RUN_DIR']); root.mkdir(parents=True,exist_ok=False)
source=Path(os.environ['MWS_GOLD'])
records=rows(source); assert len(records)==383
lines=source.read_bytes().splitlines(keepends=True)
assert len(lines)==383
(root/'smoke-gold.jsonl').write_bytes(b''.join(lines[:10]))
calls=[features(row) for row in list(records.values())[:10]]
(root/'one-input.json').write_text(calls[0].model_dump_json()+'\n')
(root/'smoke-inputs.jsonl').write_text(''.join(c.model_dump_json()+'\n' for c in calls))
(root/'smoke-ids.json').write_text(json.dumps([c.call_id for c in calls]))
print('Prepared 10 deterministic calls; API inputs contain no gold facts')
PY
.venv/bin/python -m scripts.real_smoke \
  --url http://127.0.0.1:8000 --input "$MWS_RUN_DIR/one-input.json" \
  --model-id "$MWS_MODEL_ID" --artifact-sha256 "$MWS_ARTIFACT_SHA256" \
  --out "$MWS_RUN_DIR/one-http"
```

При отсутствующих DE stages добавить `--raw-model`. Утилита не загружает модель: проверяет работающий API, real readiness/identity/hash, metadata, исходный CallInput, output schema/grounding и сохраняет `prediction.json`, `response.json`, `model_info.json`, `manifest.json`, `validation_report.json`. Не перезаписывает каталог. Ошибка/mock/неверный hash/чужой call_id не дают успешный prediction export. Загрузка один раз проверяется lifecycle tests и ML loader counters/logs; HTTP metadata само по себе не доказывает фактическую природу engine.

Остановить HTTP процесс штатно перед offline evaluation/benchmark, чтобы не держать две копии модели на GPU. HTTP timeout не отменяет native inference: дождаться drain/release. Зависший engine требует согласованного restart, не принудительного освобождения slot.

## 5. Smoke 10 → validation 383

`model.json` должен содержать явный `device`, experiment-safe decode/window config, `artifact: {hash, revision, dtype, manifest_id}` и postprocessing identifier. API/evaluation/benchmark используют одну и ту же конфигурацию. `evaluation.json`:

```json
{
  "expected_evaluator_sha256": "dffec9b8b20c5a8a101a33a27b1672f2b6195f00ab60f021a1c14b31eeeacabd",
  "timeout_sec": 300,
  "comparison_protocol": {
    "input_policy": "<approved-window-and-input-policy-v1>",
    "postprocessing": "<approved-de-catalog-and-rules-v1-or-raw-model/no-de/v1>"
  }
}
```

Hash выше относится к проверенному оригинальному evaluator текущей команды. При другой официальной версии сначала согласовать hash, не отключать pin ради успешного запуска.

```bash
.venv/bin/python -m src.experiments.evaluate \
  --gold "$MWS_RUN_DIR/smoke-gold.jsonl" --factory "$MWS_EVAL_FACTORY" \
  --evaluator "$MWS_EVALUATOR" --out "$MWS_RUN_DIR/smoke-10-eval" \
  --model-id "$MWS_MODEL_ID" --experiment-id 'first-qwen-smoke-10' \
  --dataset-version 'official-val-v1/first-10' --mode real \
  --model-config "$MWS_MODEL_CONFIG" --evaluation-config "$MWS_EVAL_CONFIG"
```

Продолжать только после exit 0, manifest complete, validation_report.valid=true, ровно 10 исходных IDs, отсутствия fallback/parse errors и ручного просмотра нескольких источников. Успешный shape не является достаточным качеством; неожиданные пустые facts должны быть диагностированы ML. Для raw run идентификатор должен содержать raw-model.

Baseline пересчитать отдельно неизменённым скриптом:

```bash
.venv/bin/python "$MWS_EVALUATOR" "$MWS_GOLD" /private/organizer/baseline_pred.jsonl \
  --json "$MWS_RUN_DIR/baseline-recomputed.json"
.venv/bin/python -m src.experiments.evaluate \
  --gold "$MWS_GOLD" --factory "$MWS_EVAL_FACTORY" --evaluator "$MWS_EVALUATOR" \
  --out "$MWS_RUN_DIR/full-383-eval" --model-id "$MWS_MODEL_ID" \
  --experiment-id 'first-qwen-full-383' --dataset-version 'official-val-v1' --mode real \
  --model-config "$MWS_MODEL_CONFIG" --evaluation-config "$MWS_EVAL_CONFIG" \
  --published-baseline /private/organizer/baseline_metrics.json \
  --recomputed-baseline "$MWS_RUN_DIR/baseline-recomputed.json"
```

На 383 calls runner создаёт predictions.jsonl, metrics.json, manifest.json, validation_report.json, evaluation.log и раздельные published/recomputed baseline. Публиковать fact_precision, fact_recall, temporal_recall, state_accuracy, grounding_turn и nullable grounding_span из фактического metrics.json. Нельзя продолжать с failed manifest или удалять ошибочные calls для красивого числа. Gold допускается только в scorer process: runner передаёт extractor/normalizers исключительно allowlisted CallInput. Не давать ML bridge путь к gold и не монтировать gold в serving container.

Известно: 81 noisy optional gold type диагностируется без изменения исходного файла; baseline имеет обратный интервал и не проходит строгую проверку predictions. Не чинить его ради runner: прямой original-script пересчёт остаётся отдельной diagnostic baseline. Полный hash val: `fb0dd73bba4b642ca18538751ce4a64d40a569aa7edd1a2a2b595af3c6b6ca83`.

## 6. A/B/C после первой модели

До обучения B/C оценить размеры модели, токенизацию/окна, один pilot, VRAM и доступный GPU budget с ML. Наличие трёх labels не требует трёх full trainings: конфигурации должны соответствовать фактическим экспортам/режимам и согласованному плану. Никакие metrics не заполнять заранее.

Для каждой A/B/C использовать одинаковые val/evaluator/input/postprocessing, seed/protocol и hardware. Model/precision могут отличаться и фиксируются в отдельных immutable configs. Raw/no-DE и полный DE pipeline сравнивать отдельными сериями. Сначала quality, затем в отдельном свежем процессе benchmark; explicit train-derived warmup inputs предоставляет DE/ML, не выбирать удобные val calls.

```bash
.venv/bin/python -m src.experiments.benchmark \
  --factory "$MWS_BENCH_FACTORY" --inputs "$MWS_GOLD" \
  --warmup-inputs /private/approved/train-derived-warmup-features.jsonl \
  --config "$MWS_MODEL_CONFIG" --device cpu --cold-trials 3 --repeats 3 \
  --experiment-id A-cpu --out experiments/A-cpu-bench
```

Повторить только на реально поддерживаемых устройствах. CUDA: `--device cuda:N`, отдельная серия, `--exclusive-gpu` только при настоящем выделении одного GPU. Не запускать одновременно HTTP/evaluation/benchmark. Cold — новая session, не очищенный OS cache; RSS — текущий процесс, VRAM — PyTorch reserved peak, GPU time — allocated wall time при exclusive flag. Собирать raw samples, summary/hardware/manifest, quality, model/checkpoint size, latency/p95, throughput и nullable RAM/VRAM/GPU-seconds.

```bash
.venv/bin/python -m src.experiments.compare \
  --gold "$MWS_GOLD" --evaluator "$MWS_EVALUATOR" \
  --run A=experiments/A-eval --run B=experiments/B-eval --run C=experiments/C-eval \
  --benchmark A=experiments/A-cpu-bench --benchmark B=experiments/B-cpu-bench \
  --benchmark C=experiments/C-cpu-bench --out experiments/ABC-cpu-report
```

Benchmark config должен в точности совпадать с соответствующим evaluation model_config, включая device/artifact. Все raw phases должны быть валидны. Отсутствующие измерения остаются null. После первой настоящей успешной серии обновить [INTEGRATION_STATUS](INTEGRATION_STATUS.md) реальными результатами, не закрывать T20 из fake/protocol tests.
