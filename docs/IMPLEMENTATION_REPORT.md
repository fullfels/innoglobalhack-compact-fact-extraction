# Результат первого E2E прототипа

Проверки выполнены 13 ноября 2026 (МСК). Исходный docs/ был незакоммичен; существующие файлы не менялись. Оригинальные schema.md и README прочитаны из Downloads, hash schema совпал с аудитом. Dataset/ML/evaluator не изменены. Коммит не создавался.

## Файлы

```text
src/contracts/models.py          CallInput, Fact, CandidateBatch, HTTP envelope
src/pipeline/adapters.py         Protocols, mock, injected/pending real adapters
src/pipeline/service.py          orchestration и output grounding gate
src/api/{app,config}.py          FastAPI, config, JSON errors/logs, single worker
frontend/{index.html,style.css,app.js}
tests/test_system.py             26 unit/API/integration tests
scripts/smoke.py                 live HTTP E2E
Dockerfile, compose.yaml, .dockerignore
requirements{,-dev}.{txt,lock}   direct dependencies и полный lock
pytest.ini, .gitignore
.github/workflows/ci.yml         PR tests + Docker smoke
README.md                       запуск
 docs/INTEGRATION_V1.md          handoff ML/DE
 docs/mock-demo.png              browser evidence
```

## Фактически выполнено

- Python 3.12.14, pytest: **26 passed**, 0.23s; один upstream DeprecationWarning Starlette/AnyIO BlockingPortal.
- Схемы: forbidden gold, invalid dates/roles/idx/noise/duration, duplicate indices, interval/polarity, roundtrip и сохранение timezone.
- API: health/info/extract, неверный JSON, body limit, unsupported mock, real-not-ready, masked inference error, 429 занятого worker.
- Integration: injected model bridge → test normalizer → output gate, точное Unicode grounding и official required fields.
- Docker: финальная сборка с base digest и locked dependencies успешна. Контейнер без весов стартовал, HTTP smoke прошёл; health дополнительно вызван изнутри контейнера.
- Live smoke: health, model/info, HTML/JS, extraction и точный source — PASS.
- Browser E2E: выбор первого примера по умолчанию → extraction → два факта → клик по факту/подсветка реплики → raw JSON. Screenshot сохранён.
- `compileall`, `docker compose config --quiet` — PASS.
- GitHub Actions workflow создан; удалённый CI run не запускался. Сам compose up не запускался: образ проверен через docker run на отдельном порту 18080.
- Official evaluator не запускался: это synthetic mock, метрики обучения/val не заявляются. Соответствие формату проверено по оригинальной Markdown-схеме, Pydantic и output validation.

## Режимы и границы

Реально работают HTTP, validation, orchestration, UI, JSON/logs, Docker и injected adapter contract. Mock extraction — буквальное сопоставление трёх синтетических реплик. Дата факта в mock берётся из started_at как fixture convention. Это не NLP и не обещание качества. ID/type объекта пока null — разрешено минимальным official prediction.

Real Qwen inference, tokenizer/prompt/decoding, загрузка весов, DE catalog/time/state/polarity/grounding rules отсутствуют. При real без адаптера — 503. Metadata, context, dtype, manifest реальной модели пока null. Отдельно нужны bounded queue, stopping/timeout и lifecycle real worker (T19); текущее поведение queue=0 документировано в model/info. Смонтированный model volume не заменяет ML bridge.

## Backlog

- T16: завершена реализация и тесты.
- T17: завершён UI и manual browser smoke.
- T18: Dockerfile, lock, CI workflow реализованы; local build/startup/smoke проверены, remote CI ожидает push.
- T02: реализация contract+fixtures готова, принятие DE/ML ещё не состоялось, формальный DoD не закрыт.
- T01, T19, T20, T26 и последующие не объявляются завершёнными.

## Следующие пять задач FS

1. Получить review контракта v1 от DE/ML и закрыть T02; уточнить официальный deadline (T01).
2. Реализовать lifecycle и bounded queue/timeout worker под фактический engine/stopping контракт (T19).
3. Подключить ML bridge и DE этапы, проверить реальный checkpoint через API/UI (T20).
4. Создать benchmark collector: cold/warm, RSS/VRAM, hardware и raw samples (T26).
5. Подготовить real Docker profile, offline reload и проверку выбранного экспорта (T34).

## Исправления review F01–F09 (ветка fix/review-p1-integration)

Этот раздел уточняет состояние первоначального отчёта выше.

- F01: revalidate_instances=always на всех shared моделях; повторная проверка CandidateBatch на границах этапов и CallPrediction перед сериализацией. model_copy с polarity=0, bad attribute/date/object/source отвергается. Ошибка — 503 без prediction/meta.schema_valid.
- F02: rejected отфильтровывается всегда. unknown допустим для обычных утверждений, agreement требует confirmed и существующий client evidence. Выявление семантики принятия/предложения остаётся ML/DE; политика уточнена в INTEGRATION_V1.
- F03: ModelDescriptor определяет model_id/mode/engine; несовместимая Settings.mode приводит к ошибке сборки. Mock не может быть маркирован real обычным create_app. Legacy без descriptor допустим только как custom-mock.
- F04: InferenceWorker удерживает task независимо от HTTP waiter через shield, отмена запроса не освобождает слот. Повторные запросы до завершения физической операции получают 429. Lifespan shutdown запрещает новую работу и ожидает текущую.
- F05: InjectedAdapter стартует unloaded, загрузчик переводит его в loading/ready/failed через set_state. Pending real pipeline больше не ready. Самих Qwen loader и engine тут нет.
- F06: ошибки этапов сохраняют module/error_type в безопасных логах, без текста исключения/транскрипта. HTTP содержит только module/code/request_id.
- F07: numeric timestamps для valid_from/to запрещены; timezone и naive datetime сохраняются.
- F08: баннер получает режим и состояние из /health, при недоступности сообщает неизвестный режим.
- F09: grounding tests проверяют PipelineError и конкретный code/module.

Файлы исправлений: src/contracts/models.py; src/pipeline/adapters.py; src/pipeline/service.py; src/api/app.py; новый src/api/worker.py; frontend/index.html; frontend/app.js; tests/test_system.py; новый tests/test_review_regressions.py; README.md; docs/INTEGRATION_V1.md; этот отчёт. DE/ML pipeline, train/val и evaluator не изменены.

Проверено после исправлений:

- pytest: **66 passed**, 0.27s; исходные 26 случаев сохранены и проходят, добавлено 40 случаев. Один upstream Starlette/AnyIO DeprecationWarning.
- Новые проверки: порча вложенных моделей через model_copy; даты; confirmation/agreement; преобразования двумя последовательными normalizers; ошибки и безопасные diagnostics; пустой batch; mode/lifecycle; invalid model output; Unicode/nonzero span; cancellation, повторный запрос и draining shutdown. Отмена проверена на worker и непосредственно на FastAPI endpoint.
- compileall src/tests/scripts — успешно.
- Docker build mws-review-fixes:mock — успешно, base/dependency слои из cache.
- Отдельный Docker run на 18083; scripts/smoke.py — PASS; health изнутри контейнера — ready=true, mode=mock. docker compose config — успешно.
- Официальный case_metrics.py прочитал временный predictions JSONL. Один синтетический совпадающий факт дал единичные метрики: это проверка совместимости формата, не качества модели. Исходные official файлы не менялись.
- Remote CI и real Qwen inference не запускались.

Git: remote отсутствует, до изменений репозиторий не имел ни одного коммита, все существующие файлы были untracked. Создана запрошенная локальная ветка; чужие файлы не добавлялись в индекс и не коммитились, PR не создавался.

Остаётся: review уточнённой confirmation policy DE/ML, Qwen bridge/loader, model manifest metadata, cooperative stopping или process isolation для зависшего native inference, реальные benchmark/eval и real Docker. Graceful shutdown ждёт inference; для зависшего engine ожидание может быть неограниченным. Принудительный kill не является graceful shutdown. Очередь остаётся 0, без Redis/Celery. T19 частично реализована; T20/T26 не объявляются закрытыми.
