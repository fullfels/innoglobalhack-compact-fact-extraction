# Deep code review, fixes and integration audit

Дата: 2026-11-14. Репозиторий: `fullfels/innoglobalhack-compact-fact-extraction`.

## A. Executive summary

Проверены PR #46–#51, включая реальные зависимости веток. Подтверждённые проблемы исправлены в наиболее ранних подходящих ветках; исправления переданы зависимым веткам обычными merge-коммитами, без переписывания истории. `main` не изменена, PR не объединялись и не закрывались.

Основные дефекты касались достоверности экспериментальных артефактов, проверки результатов benchmark, сопоставимости quality/resource bundles и владения ресурсами при cancellation/close. Исправлены все подтверждённые в этом аудите P1 и локальные P2. Новых подтверждённых P0 не обнаружено. После повторного просмотра исправлений и совместной проверки открытых P0/P1 не осталось в проверенном объёме.

На совместной ветке: **137 Python tests passed, 0 failed, 0 skipped; 5 JavaScript tests passed**. Выполнены Docker build/startup/healthcheck, sentinel-проверка контекста и образа, HTTP smoke, Compose validation, browser smoke и synthetic E2E до comparison. Фактический CI всех шести PR зелёный; ссылки ниже.

Инфраструктурный путь готов к передаче ML/DE. Настоящая Qwen, её accuracy/latency/VRAM и семантика ещё не переданных DE rules не проверены. T20 (первый настоящий end-to-end inference) остаётся открытой задачей.

## B. Review scope

### Ветки и SHA

Все PR оставались `OPEN` при заключительной проверке. Base/head проверены через GitHub, а не взяты только из постановки задачи.

| PR | Base | Head branch | Начальный SHA | SHA после исправлений |
|---|---|---|---|---|
| #46 | main | fix/review-p1-integration | `4bc39594ab49c6fb9e7b7e36ded28f9781f54813` | тот же |
| #47 | main | fix/review-followups | `34b8c7fc5cbacd5928b8b79073ad9033d1b0e18a` | `73239025e9efa87d401b6912f5fa2ae08d33bb06` |
| #48 | main | fs/evaluation-runner | `7c5443431c25447de50ac2345b2b95ab78eb1e6d` | `898477addbe79019a6e29370772f1c36d90d46ce` |
| #49 | fs/evaluation-runner | fs/benchmark-collector | `b9582064e796dede881d1f5708f54ac52f0b9507` | `b74a3e168c47c2bbe262b37f51306a5ff2b533df` |
| #50 | fix/review-followups | fs/real-integration-readiness | `85ff88c0995db11c514eb85cae587fd69e4e4cba` | `6eadbd83644ae307236e7fcbee22a85fe98d9dc6` |
| #51 | fs/benchmark-collector | fs/experiment-comparison | `bec5002140bc59afd7b3da568afdc32ca896379d` | `26f5c235c2f009e0db390b7a418676215007246c` |

`main`: `c8911a99936de3d7a06c5327e55abcc5fa28694e`, без изменений. Совместный снимок: `fs/deep-integration-audit`, код протестирован на `b4569379148ec4953148e51befda10364d22bc96`. Эта ветка включает все перечисленные head; это проверочный снимок, отдельный дублирующий PR не создан. Последующий коммит добавляет только этот отчёт.

Начальный Git status был проверен; неизвестные untracked-файлы не добавлялись. При фиксации использованы явные списки файлов. Зоны обучения/первичной подготовки данных и оригинальные материалы организаторов не изменены.

### Код и проверки

- `src/experiments/{common,evaluate,benchmark,compare}.py`: JSON/JSONL, fingerprints, CLI, evaluator protocol, формулы и provenance.
- `src/contracts/models.py`, `src/pipeline/{adapters,service,lifecycle}.py`: входы, candidates, rejected/confirmation, normalizers, output gate, ресурсы.
- `src/api/{app,config,worker}.py`: readiness, ошибки, privacy, timeout, slot ownership, startup/shutdown.
- `frontend/app.js`, frontend tests: stale response, изменение ввода во время запроса, call_id, таблица/source/JSON.
- `.dockerignore`, `Dockerfile`, Compose, `.github/workflows/ci.yml`, командные инструкции/CODEOWNERS и handoff.
- Все Python tests; Node tests; реальный Docker sentinel; неизменённый официальный evaluator; отдельный synthetic HTTP→JSONL→evaluation→benchmark→comparison.

Недоступны настоящие checkpoint/tokenizer/ML bridge, DE implementations, CUDA GPU и их production deployment. Управляемые fake adapters и fake CUDA hooks проверяют управление/формулы, а не производительность Qwen.

## C. Findings

Номера строк относятся к исправленному совместному снимку `b456937`; причина и воспроизведение описывают поведение до исправления. Для P1 использованы падавшие до исправления regression tests.

| ID | Severity | Component / location | Problem | Root cause | Fix | Status |
|---|---|---|---|---|---|---|
| D01 | P2 | `evaluate.py:19`, `common.py:13` | Неизвестные поля prediction и переполнение JSON float могли проходить preflight | Не было верхнего allowlist; стандартный float parser допускает `1e999` | Строгий prediction envelope и finite JSON decoder | Исправлено #48 |
| D02 | P1 | `evaluate.py:81`, `evaluate.py:108` | Manifest мог перестать соответствовать конфигурации/проверенным данным; неверный count evaluator принимался | Общие mutable config, недостаточная проверка целостности после subprocess и aggregate counts | Deep copy; pin/check hashes; counts; invalidation metrics при ошибке; effective artifact metadata | Исправлено #48 |
| D03 | P1 | `benchmark.py:134` | Валидный по форме ответ для чужого call_id считался успешным | Проверялась схема без равенства исходного CallInput и полного source gate | Revalidation, exact CallInput equality, grounding validation | Исправлено #49 |
| D04 | P1 | `benchmark.py:105`, `benchmark.py:188` | Close failure оставлял complete summary; неудачный промежуточный close вызывался повторно | Ownership сохранялся во время callback, cleanup был после записи success | Ownership снимается до callback; once-only cleanup; failure invalidates artifacts | Исправлено #49 |
| D05 | P1 | `benchmark.py:71`, `benchmark.py:153` | Неудачный warmup и смена модели между cold trials не исключали benchmark | Учитывались только warm errors; identity не сравнивалась между trials | Все phase_errors, benchmark_valid, стабильные config/identity/artifact | Исправлено #49 |
| D06 | P1 | `compare.py:73`, `compare.py:145` | Quality и resources разных конфигураций/протоколов могли попасть на один график | Недостаточное связывание configs, artifact, mode, device, postprocessing | Строгое per-run matching и общий quality protocol; provenance для real | Исправлено #51 |
| D07 | P1 | `compare.py:98`, `compare.py:113` | Изменённый peak RAM или неполные raw phases принимались при согласованных hashes | Summary проверялся по timing, но не по raw resource/protocol records | Raw RSS/resource peaks, phase coverage, device/GPU formulas, finite values | Исправлено #51 |
| D08 | P1 | `worker.py:10`, `worker.py:69` | Отменённый close возвращал управление до окончания native inference | Cancellation прерывала shield waiter, не underlying thread | Drain до физического завершения даже при повторных cancellation | Исправлено #47 |
| D09 | P1 | `app.py:51`, `app.py:64` | Cancellation startup оставляла loader thread/session без cleanup | Load ожидался вне гарантированного lifecycle cleanup | Shielded load + finally; drain loader/worker/close; безопасный поздний error log | Исправлено #50 |
| D10 | P1 | `lifecycle.py:50`, `lifecycle.py:58` | Прямой adapter.close мог освободить ресурсы во время inference | Inference и resource release не разделяли lock | Общий lock, stopping signal, защита reentrant close | Исправлено #50 |
| D11 | P2 | `lifecycle.py:33`, `lifecycle.py:39` | Некорректный LoadedModel bridge мог утечь после успешной аллокации | Ошибка проверки handle не освобождала возвращённый session | Cleanup доступного handle; failed state и safe code/type | Исправлено #50 |

Дополнительное усиление в этих же локальных исправлениях: coroutine не считается завершённым benchmark inference; неконечный timer и отсутствующий объявленный checkpoint отклоняются; TP/FP/FN явно отделены от temporal/grounding correctness; optional span имеет флаг доступности. Это проверяется тестами и не меняет официальный matcher.

## D. Fixes, reproduction and re-review

### D01–D02: evaluation (#48)

До исправления: prediction с неизвестным top-level ключом принимался; factory меняла вложенный config в caller/manifest; подмена проверенного gold после preflight и JSON evaluator с `n_calls=999` не препятствовали complete. Regression tests сначала воспроизвели эти сценарии, затем прошли после исправлений.

Тесты: `test_unknown_prediction_fields_rejected`, `test_config_factory_cannot_rewrite_recorded_manifest`, `test_evaluator_must_not_change_validated_inputs`, `test_mismatched_metric_counts_rejected`, `test_ambiguous_or_nonfinite_json_rejected`, `test_expected_evaluator_hash_is_enforced` в `tests/test_evaluation_runner.py:136` и далее. Существующие tests продолжают проверять bad JSON/duplicates/coverage/attributes/polarity/dates/grounding, минимальный официальный record, отсутствие gold leakage, release при ошибке и реальный evaluator.

Коммиты:

- `aa2fc8fe35fc913924ac1afc3b89e6fc4dee30c2` — validation/provenance/error artifacts.
- `03c960628d1923f2e3768a0ee414979f09ba0622` — effective artifact metadata для последующего pairing.
- `1f25261e8b3f4e2024c3ac27fd21cd76d1b2957a` — изоляция тестового subprocess spy.
- `898477addbe79019a6e29370772f1c36d90d46ce` — документация.

При повторной проверке нового теста обнаружен собственный тестовый дефект: глобальный spy subprocess перехватывал также `git rev-parse` и создавал синтетический файл `HEAD` в корне checkout. Spy изолирован через подмену `identity`; удалён только доказанно созданный этим тестом файл с фиктивными метриками. Это не production bug и не чужой файл. Итоговая рабочая копия чиста.

### D03–D05: benchmark (#49)

Падавшие tests возвращали prediction с чужим call_id, бросали исключение из final/intermediate close, ломали warmup и меняли descriptor между cold trials. Корень исправлен на границах output/ownership/protocol, без новой реализации модели.

Тесты `tests/test_benchmark_collector.py:83` и далее: `test_foreign_prediction_is_not_schema_success`, `test_cleanup_failure_invalidates_completed_manifest_once`, `test_intermediate_cleanup_is_not_repeated_on_failure`, `test_failed_warmup_invalidates_resource_comparison`, `test_changed_model_between_cold_trials_rejected`, `test_unawaited_async_pipeline_is_rejected_without_success`, `test_nonfinite_clock_cannot_create_complete_benchmark`, `test_declared_missing_checkpoint_is_error`.

Коммиты: `9e1b8df088904c9431b6604901fe9c2e4cbea4cc` (код/tests), `b74a3e168c47c2bbe262b37f51306a5ff2b533df` (docs). Upstream #48 включён обычными merge-коммитами, а не дублированием исправлений.

Повторная проверка: успешные latency samples отделены от ошибок, throughput использует фактическое measured wall window; close failure делает artifacts failed и выполняется один раз. CPU не получает выдуманные VRAM/GPU-seconds. CUDA sync/formула проверены только fake runtime test.

### D06–D07: comparison (#51)

Parameterized regression сначала показал приём шести несовместимых bundles: config, protocol, resource peak, device, mode, failed samples. Теперь comparison отклоняет их до создания отчёта. Проверяются raw phases и summary, а не только самосогласованные hashes.

`tests/test_experiment_comparison.py:75`: `test_incompatible_quality_resource_reports_rejected` (6 случаев), `test_official_fact_counts_are_order_independent`, `test_real_comparison_requires_provenance_not_only_label`. Реальный official integration test дополнительно меняет temporal/grounding/polarity/value и проверяет пять графиков и tamper rejection. CI-only `tests/evaluator_protocol_double.py` — явно обозначенный собственный protocol double, не копия/замена официального evaluator.

Коммиты: `15cb48302d5d5d628852d8682a221762da1d8ebe`, `b3a03b91d72968c6d49d5c3325376a248e7ef484`, `c82fa1244715e9e2fbc63cdfd1d909539b6f512b`, `26f5c235c2f009e0db390b7a418676215007246c`.

Повторно проверены `delta_pp=100*(B-A)`, `speedup=latency_A/latency_B`, отсутствие деления на ноль, null для неизвестных измерений, min-latency/max-recall Pareto, official match_facts вместо позиции в JSON. TP по official matcher не означает отсутствие temporal/grounding ошибки; auxiliary near-misses не меняют TP/FP/FN. Особенности official state/object matching сохранены и обозначены.

### D08–D11: worker / lifecycle (#47 → #50)

До исправления управляемые Events воспроизводили ранний выход отменённого close, освобождение session во время прямого inference, утечку invalid bridge и loader, завершившийся уже после отмены startup. Новый drain и общий resource lock сохраняют владельца до завершения физической работы.

`tests/test_detached_errors.py:94`: `test_cancelled_shutdown_still_drains_physical_work` (включая повторную cancellation). `tests/test_managed_integration.py:72` и далее: `test_close_waits_for_direct_adapter_inference`, `test_invalid_loaded_bridge_releases_allocated_session`, `test_cancelled_startup_waits_for_loader_and_releases_resources`, `test_stopping_health_and_concurrent_close_are_safe`, `test_cancelled_startup_failure_is_still_logged`.

Коммит #47: `73239025e9efa87d401b6912f5fa2ae08d33bb06`. Коммиты #50: `40687b4991c33f3f6e4f22147e8bb686cd8b82c1`, `6eadbd83644ae307236e7fcbee22a85fe98d9dc6`; #47 включён обычным merge.

Повторная проверка включает single initialization, 429 при занятом физическом slot, 504 без отмены native inference, shutdown drain до release, safe logging поздней ошибки, stopping/failed readiness 503, concurrent close once. Reentrant close из resource callback явно отвергается вместо deadlock. Не реализовано недоказанное восстановление CUDA/OOM.

### Сохранённые R01–R03 / F01–F09 и security review

- Detached failure логируется один раз с request_id/module/code/type; активный waiter не создаёт двойной записи. Транскрипт и внутреннее сообщение исключения не выводятся.
- Docker sentinel использует только фиктивные строки: вложенные env/keys/credentials/checkpoints исключены из context и image; runtime sources/assets остаются доступны.
- Frontend сбрасывает результат при смене ввода, игнорирует ответ старой revision и чужой call_id; таблица/source highlight/JSON работают в браузере.
- Сохранены nested revalidation, rejected filtering, confirmation policy, даты/timezone/интервалы, Unicode/source grounding и честные mock/real descriptors.
- Просмотрены subprocess calls: списки аргументов без `shell=True`; HTTP не принимает factory/evaluator/model paths. Нет generic pickle/model deserialization. Factory и evaluator остаются доверенным локальным исполняемым кодом, не допускается загрузка произвольного пользовательского Python.
- HTTP body ограничен 256 KiB, turns — 500; output/диагностика не раскрывают private model path/tokenizer. Frontend выводит значения через textContent. Служебные endpoints не имеют аутентификации и предназначены для текущего локального demo deployment; публичная публикация требует отдельной настройки доступа.

## E. Test results

### Локальный совместный снимок

В окружении Python 3.12.14 выполнено:

```bash
OFFICIAL_EVALUATOR=/path/to/original/case_metrics.py \
MWS_DOCKER_TESTS=1 MPLCONFIGDIR=/private/tmp/mws-matplotlib \
PYTHONDONTWRITEBYTECODE=1 .venv/bin/pytest -q -p no:cacheprovider
node --test tests/frontend.test.cjs
docker build -t mws-deep-review:integration .
docker compose config --quiet
python scripts/smoke.py http://127.0.0.1:18089
```

| Проверка | Фактический результат |
|---|---|
| Полный pytest | **137 passed, 0 failed, 0 skipped**, 5.61 s; 2 upstream deprecation warnings (Starlette/AnyIO, matplotlib/pyparsing) |
| Frontend Node tests | **5 passed**, 0 failed |
| Docker build | PASS, production Dockerfile, mock mode |
| Docker startup/healthcheck | PASS, контейнер healthy; host binding `127.0.0.1:18089` |
| Context/image synthetic sentinel | PASS внутри полного pytest; настоящие credentials не использованы |
| HTTP smoke | PASS: health, model/info, assets, live extraction, source |
| Compose configuration | PASS |
| Browser smoke | PASS: два примера, таблица, source highlight, JSON, очистка старого результата |
| Synthetic full pipeline | PASS: HTTP extraction → fixture normalization → output validation → prediction-only JSONL → official evaluation → CPU mock benchmark → A/B/C comparison + 5 plots |

Image manifest digest: `sha256:935954d97e0c1b0332bdb634ca213ea0f4a47335a33043adf2af601b4d733263`. Synthetic driver и экспериментальные outputs сохранены только во временном локальном каталоге, не в Git и не в Docker image. Контейнеру fixture E2E передан только read-only plugin, gold-файл не монтировался.

### Официальные файлы / 383 calls

Использован оригинальный внешний `case_metrics.py`, SHA256 `dffec9b8b20c5a8a101a33a27b1672f2b6195f00ab60f021a1c14b31eeeacabd`. SHA256 исходного `val.jsonl`: `fb0dd73bba4b642ca18538751ce4a64d40a569aa7edd1a2a2b595af3c6b6ca83`. Проверены повторно после запусков; файлы не менялись и не публиковались.

Runner успешно обработал **383 calls / 1845 facts** в отдельно маркированном `gold-projection-NOT-a-model`, mode=synthetic: проекция эталона проверяет формат/проводку полного корпуса, **не inference и не ML accuracy**. Optional noisy type/span/quote не переносились в эту искусственную prediction fixture; оригинальный gold передавался evaluator без изменений. Все 81 gold `object.type=other` сохранены как диагностические warnings, соответствующие calls не удалены. Предсказания с явно неверным type по-прежнему отвергаются.

Оригинальный baseline с обратным интервалом предсказуемо отвергнут строгим runner: exit 2, manifest failed, metrics not_evaluated. Для диагностики оригинальный permissive evaluator отдельно пересчитал исходный baseline без его исправления. Published и recomputed baseline сохранены раздельно:

| Метрика | Published | Recomputed original script |
|---|---|---|
| fact precision | 0.2074 | 0.2189 |
| fact recall | 0.2358 | 0.2488 |
| temporal recall | 0.4161 | 0.4205 |
| state accuracy | 0.3483 | 0.3194 |
| grounding turn | 0.6759 | 0.6906 |

Расхождение не замаскировано под результат модели. Причину различия опубликованного baseline должен подтвердить организатор/владелец исходного запуска.

### Remote CI

Проверены реальные completed/success workflows на окончательных SHA:

| PR | CI run |
|---|---|
| #46 | [43506815453](https://github.com/fullfels/innoglobalhack-compact-fact-extraction/actions/runs/43506815453) |
| #47 | [43574180620](https://github.com/fullfels/innoglobalhack-compact-fact-extraction/actions/runs/43574180620) |
| #48 | [43574182143](https://github.com/fullfels/innoglobalhack-compact-fact-extraction/actions/runs/43574182143) |
| #49 | [43574181199](https://github.com/fullfels/innoglobalhack-compact-fact-extraction/actions/runs/43574181199) |
| #50 | [43574181137](https://github.com/fullfels/innoglobalhack-compact-fact-extraction/actions/runs/43574181137) |
| #51 | [43574181799](https://github.com/fullfels/innoglobalhack-compact-fact-extraction/actions/runs/43574181799) |

CI отдельных PR имеет меньший scope, чем объединённый локальный снимок. Например, #51: **117 passed, 2 skipped**, Docker/HTTP smoke PASS; два теста с внешним official evaluator пропущены, поскольку материалы организаторов не передаются в GitHub. Новые comparison rejection gates всё равно исполняются с ясно обозначенным protocol double. Локально original evaluator предоставлен, поэтому пропусков нет. Предупреждение runner о Node runtime старых версий Actions не мешает выполнению, обновление Actions оставлено отдельной задачей.

## F. PR status

Ready to merge означает техническую рекомендацию в проверенном объёме, а не выполненное одобрение от имени другого участника или merge.

| PR | Review | Fixes | CI | Ready to merge |
|---|---|---|---|---|
| [#46](https://github.com/fullfels/innoglobalhack-compact-fact-extraction/pull/46) | Workflow/handoff проверены | Новых code defects не подтверждено | success | Да, после командного review; branch protection остаётся внешним ограничением |
| [#47](https://github.com/fullfels/innoglobalhack-compact-fact-extraction/pull/47) | R01–R03 + shutdown | D08 | success | Да |
| [#48](https://github.com/fullfels/innoglobalhack-compact-fact-extraction/pull/48) | Evaluation/383-call compatibility | D01–D02 | success | Да; согласовать ранее введённый приём empty ASR turns с DE/ML |
| [#49](https://github.com/fullfels/innoglobalhack-compact-fact-extraction/pull/49) | Timing/resources/identity/errors | D03–D05 | success | После #48 и обновления base |
| [#50](https://github.com/fullfels/innoglobalhack-compact-fact-extraction/pull/50) | Managed lifecycle/readiness | D09–D11 | success | После #47 и обновления base; T20 отдельно |
| [#51](https://github.com/fullfels/innoglobalhack-compact-fact-extraction/pull/51) | Compatibility/math/errors/plots | D06–D07 | success | После #49 и обновления base |

Зависимые ветки обновлены; force push не применялся. После фактического объединения родителей необходимо retarget/recheck оставшиеся PR: статус текущего CI не заменяет проверку нового base.

## G. Integration readiness

| Компонент | Готовая точка / точные требования | Что ещё предоставить |
|---|---|---|
| Evaluation | `python -m src.experiments.evaluate`; factory(config) → Pipeline; CallInput allowlist; prediction-only JSONL; strict preflight; unchanged official evaluator; artifact/config hashes | Утверждённый evaluator hash, реальные predictions/config/manifest без секретов |
| Benchmark | factory(config + device) → BenchmarkTarget(pipeline, device, synthetic, parameters, checkpoint_paths, close); sync completed Pipeline.run; explicit warmup/repeats/cold_trials; raw/summary/hardware | ML engine/device handling, корректный close, base+adapter/tokenizer file list, реальные CPU/GPU runs |
| A/B/C | Complete evaluation/benchmark bundles; совпадающие per-run config/artifact/device; общий input/postprocessing protocol; hashes и raw records | Реальные A/B/C artifacts и одинаковый benchmark protocol; incomplete legacy bundles требуется перезапустить |
| Lifecycle | ManagedAdapter(ModelResources, loader, descriptor, public metadata); loader → LoadedModel(inference, close); load once; readiness/cleanup/drain | Настоящий loader, очистка частичной аллокации при ошибке до return, engine fatal-error/restart policy |
| Qwen interface | Extractor.extract(CallInput) → CandidateBatch; bridge(CallInput, ModelResources) → CandidateBatch; ModelResources(model_path, tokenizer, inference_config); ModelDescriptor(model_id, real, engine) | Экспорт checkpoint/base+adapter, tokenizer/special tokens, dependencies, revision/hash/license, dtype/context/device/seed/stop/window policy, bridge/parser, approved real fixture |
| DE interface | Normalizer.normalize(CandidateBatch, CallInput) → CandidateBatch; ordered tuple: catalog → temporal → state → polarity → grounding → final gate | Реализации/catalog+hash, fixtures и ambiguity policy; rejected никогда не экспортировать; agreement только confirmed с клиентским evidence; unknown разрешён обычным утверждениям |
| Real Docker | `MWS_MODE=real`, trusted `MWS_PIPELINE_FACTORY=module:function`, one worker, read-only /models, отсутствие mock fallback | Проверенные engine dependencies/base image/profile, offline reload, CPU/GPU serving acceptance |

Контракты Extractor, Normalizer, CandidateBatch, ModelResources, ModelDescriptor и bridge не изменены этим аудитом. Empty ASR Turn.text — уже существовавшее в начальном #48 совместимое расширение входа; индексы не перенумеровываются. Состояние stopping отражается приложением/adapter signal без расширения общего `Extractor.state` enum.

DE: source.turn_idx должен существовать, speaker соответствовать реплике; span — Python code points `[start,end)`, quote равен срезу. При ненадёжном span рекомендуется span/quote=null. Temporal anchor=started_at, не выдумывать UTC для naive времени и точную дату для ambiguity; valid_to >= valid_from. State reconciliation закрывает только взаимоисключающее состояние того же объекта и сохраняет историю/multiobject. `model_copy` не заменяет revalidation.

ML: штатный пустой CandidateBatch допустим; load/inference/decode/context/OOM ошибки нельзя превращать в facts=[]. Использовать PipelineError со стабильным module/code/error_type без секретного message. Metadata generic wrapper лишь публикует; фактические hashes/tokenizer/checkpoint должен проверить engine-specific loader. Нативный async engine должен завершать свою работу внутри синхронного bridge.

Историческая таблица blockers в `INTEGRATION_HANDOFF.md` из #46 отражает состояние до #48/#49. Для текущего состояния evaluation/benchmark/lifecycle использовать этот раздел и `EVALUATION_RUNNER.md`, `BENCHMARK_COLLECTOR.md`, `EXPERIMENT_COMPARISON.md`, `REAL_INTEGRATION_READINESS.md` вместе с `INTEGRATION_V1.md`.

## H. Remaining risks

**Подтверждённые нерешённые code defects P0/P1 в проверенном объёме:** нет. Это не доказательство отсутствия ошибок в несуществующих ML/DE components.

- Настоящий Qwen inference, accuracy, parsing robustness, checkpoint/tokenizer compatibility, real CUDA synchronization/memory/latency/OOM не проверены. Fake hooks не являются GPU benchmark.
- HTTP timeout ограничивает ожидание ответа, а не исполнение native engine. Hung inference сохраняет slot и может задерживать graceful shutdown. Нужны cooperative engine stopping либо отдельный процесс/проверенная restart policy; фиктивного OOM recovery нет.
- Loader, упавший до возврата LoadedModel, отвечает за собственную частичную аллокацию. Нельзя освободить неизвестные generic wrapper ресурсы.
- RSS — сэмплирование текущего процесса каждые 20 ms, не process tree и не гарантированный абсолютный peak. CUDA reserved peak — PyTorch allocator, не NVML/device-wide VRAM. Cold trial — новая target instance, не холодный OS cache/новый процесс. Публиковать эти ограничения вместе с результатами.
- Реальная совместимость hardware/protocol зависит от честного factory/manifest и одинаковой нагрузки/выделения GPU. Hashes гарантируют целостность файлов относительно manifest, а не достоверность произвольного внешнего производителя. Config/artifact не должны содержать credentials.
- Известные особенности/ограничения official object/state matching и расхождение published baseline остаются внешними; evaluator и исходные данные не исправлялись.
- Branch protection для текущего private repository недоступна в текущем GitHub плане (ранее API 403); target configuration не равна применённой защите. Нужен подходящий план либо командная review discipline. Username DE всё ещё не подтверждён; ML `QFa1` имеет write. Нельзя выдумывать CODEOWNERS/collaborator account.
- Нет проверки публичного deployment с authentication/rate limiting; текущий контейнер тестировался через loopback. Engine-specific безопасная загрузка checkpoint должна исключать недоверенный pickle/remote code по умолчанию.
- CI не содержит официальных private organizer files; соответствующие проверки воспроизводятся локально. Actions Node runtime deprecation и upstream library warnings остаются некритичными задачами сопровождения.

## I. Next actions

1. Командный review #46 и #47; объединять только по отдельному указанию владельца. #46 независим; #47 — родитель #50.
2. Объединить #48, retarget #49 на обновлённую main и повторить CI; затем #49 → retarget/recheck #51 → #51. Не squash/rebase общую зависимую историю без согласованной процедуры.
3. После #47 retarget/recheck #50 и выполнить командный review. Интеграционный снимок не объединять вторым параллельным umbrella PR.
4. ML передаёт экспорт/manifest/tokenizer/dependencies/loader/bridge/real fixture; DE — ordered normalizers/catalog/semantic fixtures. FS подключает их существующим factory и проверяет первый offline+HTTP real inference. T20 закрывать только по настоящему результату.
5. FS запускает full-val evaluation, затем отдельные сопоставимые CPU/GPU benchmarks A/B/C и comparison. Проверить process isolation/cooperative stop и real OOM policy до долговременного serving. После этого обновить реальные measured figures; synthetic artifacts не использовать как ML результаты.

Ни один открытый PR не был автоматически объединён. Веса, raw dataset, секреты, временные driver/results и материалы организаторов не добавлялись в Git.
