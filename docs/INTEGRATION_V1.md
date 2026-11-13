# Реализованный интеграционный контракт v1

Источник: официальный `/Users/aleksandrkulakov/Downloads/schema.md` (§1–5), прочитан вместе с оригинальным README; схема соответствует SHA256 `8435f897c4d848a696c2bc5aecf061914d251296ffb2e0312f2d2d907436484b` из аудита. Исходные документы в репозитории отсутствовали. Их содержимое не заменено новой документацией.

`src/contracts/models.py` — исполняемый контракт. CallInput: call_id, started_at, optional duration_sec/series/noise_level, 1–500 immutable turns (idx уникален, может быть непоследовательным). Порядок массива сохраняется. Gold и неизвестные поля запрещены. Метаданные даты не меняют timezone.

Fact: object{name,id?,type?}, attribute enum из официальной схемы, свободный value, polarity ±1, speaker, valid_from/to, source{turn_idx,span?,quote?}. Output gate проверяет интервал, существование исходной реплики, speaker, Unicode Python offsets и точное quote. При отсутствии надёжного span — span/quote оба null. CallPrediction сохраняет CallInput и добавляет facts; HTTP envelope содержит prediction и meta. Пустые facts допустимы только при успешном inference, ошибки не маскируются.

CandidateFact оборачивает Fact: object_key?, evidence_turns, temporal_expression?, confirmation=unknown/confirmed/rejected, provenance, warnings. CandidateBatch содержит candidates/warnings/status=ok. Ошибки — исключения PipelineError(code,status,module), не facts. Confidence не придумывается. Незавершённый или частичный batch не представляется успешным.

## ML Engineer

`Extractor.extract(call: CallInput) -> CandidateBatch` — общий interface. `InjectedAdapter(ModelResources(model_path, tokenizer, inference_config), inference)` принимает **нашу bridge-функцию** `(CallInput, ModelResources) -> CandidateBatch`. Это не сигнатура фактической модели: ML реализует bridge после передачи артефактов, сам выполняет prepare/tokenize/decode/parse, гарантирует исходные idx и отсутствие gold. Backend не импортирует Transformers и не зависит от GPU. Проверка формы результата выполняется адаптером; grounding — pipeline.

Для сборки приложения: `create_app(Settings(mode='real', model_id='public-id'), Pipeline(adapter, normalizers))`. InjectedAdapter стартует в unloaded. Loader вызывает set_state("loading"), затем set_state("ready") только после успешной загрузки; при неудаче — set_state("failed"). Одно наличие pipeline больше не означает ready. ModelDescriptor(model_id, mode, engine) — источник metadata; несовпадение Settings.mode с descriptor.mode запрещено. Model path и inference config остаются приватными. Конкретный Qwen bridge, model load, tokenizer, stopping, windows и metadata manifest остаются до handoff ML. Mount `/models:ro` в compose подготовлен, но сам по себе не подключает модель.

Передать ML: все shared модели, adapter Protocol, fixtures тестов; получить checkpoint/revision/hash/license, tokenizer, inference config, bridge, context/stop policy и проверенную загрузку.

## Data Engineer

`Normalizer.normalize(batch: CandidateBatch, call: CallInput) -> CandidateBatch`. Pipeline принимает ordered tuple адаптеров этапов. DE может обернуть согласованные normalize_catalog/time/reconcile_state/validate_polarity/ground в этот интерфейс; catalog передаётся при конструировании объекта. Original turns frozen, без global state. Изменения и предупреждения сохраняются в provenance/warnings. По умолчанию normalizers отсутствуют, ответ явно сообщает об этом. FS реализовал только проверку output, а не DE правила.

Передать DE: CandidateBatch/Fact/CallInput, unicode source tests, null-span policy; получить normalizers, catalog hash и temporal/polarity fixtures. Порядок: catalog → time → state → polarity → grounding → output validation.

## Ограничения первого прототипа

Worker один, очередь 0 с 429 вместо предварительно предложенной очереди 8. Это сознательное упрощение для mock E2E; перед real integration нужны очередь ≤8, queue timeout ≤30s и cooperative stopping/изоляция worker для inference timeout. Timeout asyncio не добавлен: он не останавливает native/GPU вызов. Реальная загрузка и health lifecycle требуют артефактов. Нет измерений качества/ресурсов Qwen и заявлений о соответствии performance gates.

## Backlog

T16 реализована и покрыта тестами. T02: контракт и fixtures реализованы, формальное принятие DE/ML ещё требуется. T17: UI реализован, статус browser smoke см. IMPLEMENTATION_REPORT. T18: Docker mock, статус реального smoke см. отчёт. T19/T20 (real worker/интеграция) не закрыты одним лишь interface. Исходные tasks.csv и backlog не менялись без review.


## Уточнения после review F01–F09

Все shared Pydantic модели используют revalidate_instances=always: model_copy(update=...) не считается доказательством корректности. CandidateBatch повторно проверяется после extractor и каждого normalizer, CallPrediction — перед экспортом. Ошибки не возвращают meta.schema_valid=true. Числовые timestamps valid_from/valid_to запрещены; разрешены datetime и ISO datetime string (valid_to также null).

Политика confirmation: rejected никогда не экспортируется. unknown для service_state/tariff/ticket_status/action допускается как утверждение, но не как операторское предложение/гипотеза; выявить такое предложение и выставить rejected либо удалить кандидата обязаны extractor/DE polarity stage. confirmed означает подтверждённое по контексту утверждение, а не gold label. agreement экспортируется только с confirmed и evidence_turns, содержащим существующую клиентскую реплику принятия. Семантическую проверку принятия («да, подходит», а не «нет») выполняет ML/DE; backend проверяет статус и наличие клиентского evidence, не подменяет NLP правилом на любую реплику клиента. Отфильтрованные кандидаты отмечаются warning. Пустой успешный batch допустим.

Extractor.extract и Normalizer.normalize сохраняют прежние сигнатуры. Для serving extractor публикует descriptor и state. Legacy extractor без descriptor разрешён только в mock с именем custom-mock; real сборка требует явного descriptor. InjectedAdapter использует real descriptor (можно передать ModelDescriptor('qwen-A','real','engine-name')), state изначально unloaded. Программная метка Settings.model_id не переименовывает фактический адаптер; MWS_MODEL_ID не является источником идентичности.

InferenceWorker владеет task, HTTP coroutine ожидает его через shield. Отмена HTTP-ожидания не отменяет inference и не освобождает slot. Повторный запрос получает 429 до физического завершения. Graceful lifespan shutdown прекращает приём и ждёт текущую работу. Принудительный kill процесса не гарантирует завершение; зависший native inference требует будущего stopping/process isolation. Timeout не реализован искусственным снятием slot. Ошибка этапа сохраняет module/error_type в безопасном JSON log без сообщения исключения/транскрипта; HTTP содержит только module/code/request_id.

Новые условия confirmation и descriptor требуют review DE/ML перед подключением. Статусы являются сигналом загрузчика, а не проверкой наличия файлов на диске при каждом запросе. Веса, реальная загрузка, engine-specific stopping, inference timeout, очередь и model manifests остаются задачами real integration.
