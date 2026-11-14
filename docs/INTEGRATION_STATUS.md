# Integrated system status — 2026-11-14

Последняя проверка: [QWEN_ACCEPTANCE_STATUS.md](QWEN_ACCEPTANCE_STATUS.md). PR #52 на 4958f94 получил официальный APPROVE после независимых mutation tests; появились draft #53 bridge и #54 pilot. Отдельная FS-ветка codex/qwen-acceptance включает bridge и metadata fix #50 (56303b0), full suite 183 passed. Реальных артефактов/reload/inference пока нет. Предшествующие отчёты ниже сохраняются как история.

Дополнение после появления PR #52: [PR52_INTEGRATION_REVIEW.md](PR52_INTEGRATION_REVIEW.md). Архивный 10-call Qwen smoke независимо пересчитан; request changes по source provenance. Новый FS verifier и 7 tests не меняют ML source или serving contracts. Текущий full local suite — 154 passed, 0 skipped; real serving и T20 по-прежнему не готовы без ML package. Данные основного отчёта ниже относятся к предшествующему snapshot.

## Итог и границы

Проверен единый snapshot #46–#51 в `fs/deep-integration-audit`. Все актуальные PR heads входят в его историю (проверено merge-base после authenticated fetch). `main` остаётся `c8911a99936de3d7a06c5327e55abcc5fa28694e`. Открытые PR не объединялись, их heads/base не менялись этой задачей. Force push, переписывание истории и добавление неизвестных файлов не применялись.

Пользователь подтвердил: настоящие ML-артефакты ещё не переданы. Дополнительно проверены repo, ignored artifacts, Downloads, стандартный Hugging Face cache и все remote branches/Actions artifacts: checkpoint/tokenizer/bridge не обнаружены, `src/ml` и `src/normalizers` отсутствуют, GitHub artifacts — 0. Путь в ином хранилище не был предоставлен. **Qwen не подключена; T20 не выполнена.**

Инфраструктура работает совместно. Новых подтверждённых production integration bugs при повторной проверке не найдено. Исправлены устаревшие инструкции, утверждавшие, что evaluation/benchmark/lifecycle/HTTP deadline ещё не реализованы. Интерфейсы Extractor/Normalizer/CandidateBatch/ModelResources/ModelDescriptor не изменены; ML/DE implementation paths, training и официальный evaluator не тронуты.

## GitHub gate

| PR | Base | Head SHA | GitHub conflicts | CI | Peer approval |
|---|---|---|---|---|---|
| [#46](https://github.com/fullfels/innoglobalhack-compact-fact-extraction/pull/46) | main | `4bc39594ab49c6fb9e7b7e36ded28f9781f54813` | нет | [success](https://github.com/fullfels/innoglobalhack-compact-fact-extraction/actions/runs/43506815453) | отсутствует |
| [#47](https://github.com/fullfels/innoglobalhack-compact-fact-extraction/pull/47) | main | `73239025e9efa87d401b6912f5fa2ae08d33bb06` | нет | [success](https://github.com/fullfels/innoglobalhack-compact-fact-extraction/actions/runs/43574180620) | отсутствует |
| [#48](https://github.com/fullfels/innoglobalhack-compact-fact-extraction/pull/48) | main | `898477addbe79019a6e29370772f1c36d90d46ce` | нет | [success](https://github.com/fullfels/innoglobalhack-compact-fact-extraction/actions/runs/43574182143) | отсутствует |
| [#49](https://github.com/fullfels/innoglobalhack-compact-fact-extraction/pull/49) | fs/evaluation-runner | `b74a3e168c47c2bbe262b37f51306a5ff2b533df` | нет | [success](https://github.com/fullfels/innoglobalhack-compact-fact-extraction/actions/runs/43574181199) | отсутствует |
| [#50](https://github.com/fullfels/innoglobalhack-compact-fact-extraction/pull/50) | fix/review-followups | `6eadbd83644ae307236e7fcbee22a85fe98d9dc6` | нет | [success](https://github.com/fullfels/innoglobalhack-compact-fact-extraction/actions/runs/43574181137) | отсутствует |
| [#51](https://github.com/fullfels/innoglobalhack-compact-fact-extraction/pull/51) | fs/benchmark-collector | `26f5c235c2f009e0db390b7a418676215007246c` | нет | [success](https://github.com/fullfels/innoglobalhack-compact-fact-extraction/actions/runs/43574181799) | отсутствует |

Все шесть PR `OPEN`, не draft, mergeable=true; review submissions и inline review threads пусты. Существующий technical audit/comments не заменяют независимого peer approval. CONTRIBUTING требует другого участника review; shared contracts также требуют затронутых DE/ML reviewers. main имеет protected=false: командное правило действует даже без серверной защиты. Не обходить его через admin/self approval.

После peer review и отдельного разрешения владельца: **#46 → #47 → #48 → #49 → #50 → #51**. Это один допустимый последовательный порядок: #46 независим; #50 зависит только от #47, а #51 — от #49. После #47 retarget #50, после #48 retarget #49, после #49 retarget #51 на main. Перед каждым merge повторно проверить base/head, минимальность diff, конфликтность, актуальность approval и CI; после merge — CI main. Обычные merge-коммиты сохраняют общую зависимую историю. Никаких merge/retarget автоматически не выполнено.

Именно этот порядок дополнительно смоделирован через merge-tree/commit-tree на временных Git objects без изменения refs: все шесть шагов без конфликтов; итоговые src/frontend/Docker/Compose/CI совпадают с проверенным snapshot. Симуляция не является merge PR и не отменяет повторную проверку после новых commits/reviews.

## Что добавлено для первого real запуска

`scripts/real_smoke.py` — проверка уже работающего API, без загрузки/подмены модели:

- Требует features-only CallInput, ожидаемые public model_id и artifact SHA256.
- Проверяет real/ready в health и model/info, совпадение metadata/manifest и отсутствие mock.
- Сверяет исходный CallInput, output schema и source grounding через существующие контракты/output gate.
- При отсутствии DE stages допускает экспорт только с явным `--raw-model`; такой manifest не означает полного pipeline.
- Сохраняет prediction.json/response.json/model_info.json/manifest.json/validation_report.json в новый каталог. Ошибки не превращаются в успешные facts; содержимое исключений/транскрипт не печатаются в CLI logs.

Код/tests: `59364a1cca66ad8905e64d047f5ca9e18490e436`. Десять новых protocol tests проверяют успешный export без повторного inference, отказ от overwrite, mock/unready/hash/call_id/metadata/manifest/gold/grounding rejection и raw-model policy. HTTP doubles в tests **не являются настоящей моделью**.

Проверяемый handoff и команды: [FIRST_REAL_INFERENCE.md](FIRST_REAL_INFERENCE.md). Уточнены именно существующие factory signatures: API — без аргументов с load в lifespan; evaluation — config с load/close в runner; benchmark — config/device с load внутри измеряемой factory и close callback. Существующая документация обновлена адресно, без новой архитектуры.

## Фактические проверки

До дополнений повторно проверен исходный integrated code: 137 Python passed, 0 skipped. После real smoke/tests:

```bash
OFFICIAL_EVALUATOR=/path/to/original/case_metrics.py \
MWS_DOCKER_TESTS=1 MPLCONFIGDIR=/private/tmp/mws-matplotlib \
PYTHONDONTWRITEBYTECODE=1 .venv/bin/pytest -q -p no:cacheprovider
node --test tests/frontend.test.cjs
python -m compileall -q src scripts
docker build -t mws-qwen-readiness:integration .
docker compose config --quiet
python scripts/smoke.py http://127.0.0.1:18090
```

| Проверка | Результат |
|---|---|
| Full Python suite | **147 passed, 0 failed, 0 skipped**, 3.34 s, 2 upstream deprecation warnings |
| Новые real smoke protocol tests | **10 passed**, 0 failed |
| Frontend tests | **5 passed**, 0 failed; stale results/input revision/call_id/JSON сохранены |
| Imports / compileall / три CLI help | PASS: app, contracts, lifecycle, evaluation, benchmark, comparison |
| Docker build | PASS, текущий production Dockerfile, без весов/обучения |
| Mock startup/healthcheck | PASS: healthy, loopback port 18090 |
| HTTP mock E2E | PASS: health/model info/assets/live extraction/source |
| Real mode без артефактов | PASS: health 503/unloaded, model/info 200 с real/is_mock=false/unloaded, extraction 503 без prediction/fallback |
| Real smoke против mock/unavailable real | Оба ожидаемо exit 2; failed manifests, prediction не экспортирован |
| Docker sentinel / Compose | PASS; sentinel внутри полного pytest, только фиктивные credentials/weights |
| Synthetic full E2E | PASS: HTTP → fixture normalization → validation → JSONL → original evaluator → mock CPU benchmark → comparison + 5 plots |
| Настоящий real HTTP inference | **Не выполнялся: нет ML-артефактов** |
| Real validation 10/383 calls | **Не выполнялась: нет ML-артефактов** |
| Real GPU/VRAM/latency/OOM | **Не измерялись** |

Образ: `sha256:f4b90c098f3a6f7d153caf76c5f220629ff037cb1449f31ab9902b6bee014e08`. Runtime production modules не менялись этой задачей. Новый real smoke запускается с host Python, а не включается автоматически в production image.

Предыдущий snapshot CI: [43575142965](https://github.com/fullfels/innoglobalhack-compact-fact-extraction/actions/runs/43575142965), success, 135 Python passed/2 external official-file tests skipped, 5 JS passed и Docker/HTTP PASS. CI нового real smoke commit: [43576401909](https://github.com/fullfels/innoglobalhack-compact-fact-extraction/actions/runs/43576401909), **completed/success: 145 passed, 2 skipped**, 5 JS passed и Docker/HTTP PASS. Два external official-file tests пропущены: официальные материалы не загружаются в GitHub; локально они выполнены. Следующий documentation-only commit не изменяет протестированный runtime/helper код.

## Validation preparation и метрики

Подготовлены первые 10 исходных records отдельным private smoke-gold.jsonl и features-only one-input.json/smoke-inputs.jsonl. Исходные train/val не менялись, оригинальный порядок и idx сохранены. Это **подготовка входов**, а не Qwen smoke. Gold не передавался API. Приватные artifacts находятся только во временном локальном каталоге, путь сообщается в итоговом ответе, содержимое не добавляется в Git.

Отдельно повторён synthetic 383-call **format-only** check с gold projection, явно `gold-projection-NOT-a-model`: он проверяет совместимость полного корпуса/evaluator, не качество извлечения. Все 81 gold optional type anomalies остаются диагностикой без удаления calls. Значения этой искусственной проверки не приводятся как model metrics.

Оригинальный evaluator `dffec9b8b20c5a8a101a33a27b1672f2b6195f00ab60f021a1c14b31eeeacabd`, val `fb0dd73bba4b642ca18538751ce4a64d40a569aa7edd1a2a2b595af3c6b6ca83`. Исходные baseline predictions и train также проверены по hash и не изменены.

Повторный прямой пересчёт оригинального baseline сохранён раздельно с published JSON:

| Метрика | Published baseline | Recomputed baseline | Qwen |
|---|---|---|---|
| fact_precision | 0.2074 | 0.2189 | не измерена |
| fact_recall | 0.2358 | 0.2488 | не измерена |
| temporal_recall | 0.4161 | 0.4205 | не измерена |
| state_accuracy | 0.3483 | 0.3194 | не измерена |
| grounding_turn | 0.6759 | 0.6906 | не измерена |
| grounding_span | null | null | не измерена |

Официальный baseline с обратным интервалом не исправлялся. Пересчёт выполнен permissive original script отдельно от строгой валидации model predictions; расхождение published/recomputed не маскируется.

## Зависимости и следующие действия

1. **Peer review:** другой участник проверяет #46–#51 и shared contracts. ML account подтверждён как QFa1; DE username всё ещё нужен. После approvals владелец отдельно разрешает merge. Существующее разрешение на snapshot не разрешает merge main.
2. **ML:** приватный export/base+adapter, полный tokenizer/chat template/special tokens, checksums/immutable revision/license, deterministic inference config, engine dependencies/profile, context/window/stop policy, loader/release/bridge, три factory entrypoints и real fixture. Точный минимальный bundle и проверка — в runbook. Без этого не исполнять чужой непроверенный код/десериализацию и не скачивать модель автоматически.
3. **DE:** ordered normalizers, catalog+hash и ambiguity policy, semantic fixtures для negation/agreement/temporal/state/multiobject/Unicode. Семантические правила FS не реализовывал. До передачи возможен только явно raw-model real эксперимент, когда появится модель.
4. **FS + ML:** checksums/offline load once → один HTTP real запрос/metadata/prediction → smoke 10 → full 383. После каждого gate проверить реальные artifacts/ошибки; не пропускать сложные calls. T20 закрыть только по настоящему E2E с согласованной постобработкой.
5. **A/B/C:** сначала pilot и бюджет GPU; затем реальные конфигурации с одинаковыми val/evaluator/postprocessing/hardware protocol. Последовательно quality → свежий процесс benchmark → comparison. Не начинать три полных обучения и не заполнять отсутствующие metrics нулями.

Нативный hung engine всё ещё требует cooperative stop/process isolation/restart policy; HTTP timeout не освобождает физический slot. Real Docker требует ML runtime image/dependencies и явного factory/config, а /models mount не заменяет интеграцию. Эти зависимости не скрываются fake implementations.
