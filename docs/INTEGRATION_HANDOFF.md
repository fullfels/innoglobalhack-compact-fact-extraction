# Handoff первого real E2E

Исполняемые схемы: `src/contracts/models.py`. Детальная семантика: [INTEGRATION_V1](INTEGRATION_V1.md). Это требования к передаче, а не утверждение о готовых ML/DE компонентах.

## Начало работы

Принять приглашение GitHub, клонировать private repo, создать ветку от main:

```bash
git clone https://github.com/fullfels/innoglobalhack-compact-fact-extraction.git
cd innoglobalhack-compact-fact-extraction
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.lock
pytest -q
```

DE: `data/<task>`, ML: `ml/<task>`. Связать PR с существующим T-ID; один другой участник review, contracts — затронутые DE/ML и FS. Не переносить raw данные, каталог организаторов, evaluator или веса в Git без подтверждённых прав распространения. Получать отдельно, проверять hashes. .env и токены не коммитить.

## ML / QFa1

- `Extractor.extract(call: CallInput) -> CandidateBatch`; call содержит только features. Сохранять started_at, original idx/speaker/text, не читать gold. Минимальные official facts обернуть в CandidateFact.
- `InjectedAdapter(ModelResources(model_path, tokenizer, inference_config), bridge, descriptor=ModelDescriptor(model_id, 'real', engine))`. Bridge `(CallInput, ModelResources) -> CandidateBatch` — наш interface, не сигнатура Qwen. Загруженный engine удерживать в bridge/loader, не загружать каждый запрос.
- Loader вызывает `set_state('loading')`, после успешной загрузки и smoke — `ready`, при load failure — `failed`. Исходное состояние `unloaded`. Пока state не ready, /health и /extract возвращают 503. Для переключения конфигурации A/B/C создавать отдельную согласованную сборку; не загружать три копии в HTTP worker.
- ModelDescriptor — источник публичной идентичности; mode приложения обязан совпадать. Пути/tokenizer/config приватны. Поля artifact/hash/context/dtype/manifest в /model/info ещё нужно связать с ML manifest.
- Generation/parse/context overflow/OOM не превращать в facts=[]. Использовать PipelineError со стабильным code/module; неожиданные ошибки оборачиваются этапом с error_type без текста исключения в HTTP/logs. Штатный пустой CandidateBatch разрешён.
- CandidateBatch: candidates + warnings + status=ok. CandidateFact: fact, object_key, evidence_turns, temporal_expression, confirmation, provenance, warnings. Даты ISO datetime; не numeric timestamps. Span/quote при ненадёжной привязке — оба null.
- Confirmation: unknown для утверждений, rejected для опровергнутых/неподтверждённых предложений; agreement требует confirmed и evidence_turns со ссылкой на принятие клиентом. Присутствие клиентской реплики не заменяет семантическую проверку её содержания.

Ожидаются: exported checkpoint/base+adapter; tokenizer и special tokens; revision/SHA256/license/provenance; inference dependencies и CPU/GPU profile; context/window/stop policy; deterministic inference config и seed; bridge/loader; manifest A/B/C; один real fixture и команда reload без сети. Training код/targets остаются в зоне ML/DE.

## DE / username ожидается

- `Normalizer.normalize(batch: CandidateBatch, call: CallInput) -> CandidateBatch`. FS передаёт ordered tuple в `Pipeline(adapter, normalizers)`.
- Порядок: catalog → temporal → state → polarity → grounding → общий output gate. Каталог передавать в конструктор этапа, не менять original turns и не использовать global state между calls.
- Catalog aliases не должны склеивать близкие объекты; сохранять free-text action/agreement.
- Temporal: якорь started_at, naive timezone не переводить в UTC; ambiguity — warning/trace, не выдуманная точная дата. valid_to >= valid_from, согласованная timezone.
- State: закрывать только взаимоисключающее состояние того же объекта; сохранять исторические интервалы и multiobject.
- rejected никогда не экспортируется. unknown допускается для обычных утверждений, не для agreement. Для agreement выставлять confirmed только по принятой клиентом договорённости и дать client evidence_turns; иначе исключение из prediction с warning.
- Source.turn_idx существует, speaker совпадает; span — Python code points исходного текста [start,end); quote точно равен срезу. Нет надёжного span — span/quote оба null.
- Не считать model_copy подтверждением валидности: границы pipeline и финальный output выполняют revalidation.

Ожидаются normalizers + fixtures для aliases, dates/границ года, negation/agreements, state transitions/multiobject и Unicode grounding; catalog hash; ambiguity policy и trace. Веса и primary dataset отдельно от Git.

## Точки интеграции и blockers

| Компонент | Готовая точка | Зависимость / что отсутствует |
|---|---|---|
| Qwen | InjectedAdapter → Pipeline | ML bridge, engine/dependencies, checkpoint/tokenizer |
| DE rules | ordered tuple Normalizer | Реализации, catalog, semantic fixtures |
| Lifecycle | descriptor/state + create_app + InferenceWorker | Реальный loader, stopping/recovery; graceful shutdown ждёт native inference |
| Benchmark | Pipeline.run(call) | FS collector ещё не реализован: cold/warm, RSS/VRAM, CUDA sync, hardware и raw samples |
| Offline JSONL | CallInput(features) → Pipeline.run → prediction.model_dump(mode='json') | Runner ещё не реализован: gold отделить до pipeline, preflight ID/duplicates, serialize prediction без meta, evaluator отдельно |
| Real Docker | read-only /models volume | Реальные dependencies, factory entrypoint, offline reload и CPU/GPU profiles |

Сигнатуры extraction/normalization сохраняются. Очередь пока 0 (429), concurrency=1, отмена HTTP не освобождает физический slot. Зависший native engine требует stopping/process isolation; безопасный inference timeout ещё не реализован.

Acceptance: артефакт загружен из экспортированного каталога → real bridge → DE stages → валидный output → HTTP/UI с is_mock=false и корректным model_id → offline prediction IDs/schema preflight → evaluator → benchmark. Fake bridge используется только в тестах, не является acceptance real модели.
