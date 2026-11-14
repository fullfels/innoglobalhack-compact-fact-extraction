# Qwen: bridge и controlled LoRA-пилот (T09/T10)

Сборка проверяется против FS integration snapshot `640dc29e6fc49869110fd291e3d445210d9d0aeb` (`fs/deep-integration-audit`). Этот snapshot содержит ManagedAdapter/lifecycle; старый main ещё не содержит их. Изменения не заменяют shared contracts и не создают HTTP-сервер.

## Готовность

Код и синтетические regression tests подготовлены. Результаты GPU не предзаполнены: они появляются только после выполнения `train.py`. Пилот/основной SFT/официальные метрики считать выполненными только по сохранённым артефактам реального запуска. Предыдущие 10 val-звонков остаются архивным smoke до SFT.

## Данные

Используется отдельный закрытый пакет `mts_train_ready_v2.zip`: 3375 train / 366 inner-dev; pilot32/8. Официальный val383 не используется для обучения. Файлы корпуса, каталог, evaluator, ZIP и веса в Git не входят.

ML использует явное представление system prompt из `prompting.py`: текущий момент наблюдения отличается от неизвестной даты исторического события. Исходный пакет/targets/splits не переписываются. В training_config сохраняются хеш эффективного prompt, исходного manifest и исполнявшихся файлов. Один и тот же эффективный prompt используется до/после обучения, сохраняется в экспортированном артефакте.

## Проверки без GPU

```text
python -m pytest -q ml/qwen/test_bridge.py tests/test_archived_provenance.py
python -m py_compile ml/qwen/train.py ml/qwen/runtime.py
```

Тесты содержат только независимо составленные фиктивные примеры, проверяют контракты и ошибки, не качество модели. `tests/test_ml_qwen_bridge.py` обеспечивает сбор этих тестов стандартным CI.

## Пилот

Установить `ml/qwen/requirements.txt`, сохранив CUDA PyTorch среды. Затем из корня проекта:

```text
python -u -m ml.qwen.train --data-root /content/mts_train_ready_v2 --out /content/runs/pilot-unique --phase pilot --steps 64
```

Закреплены `Qwen/Qwen3.5-0.8B` и revision `2fc06364715b967f1860aea9cf38778875588b17`. LoRA FP32 по умолчанию, r16/alpha32, только linear-модули language_model; visual tower не обучается. AdamW, gradient clipping, checkpointing, seed42; autocast и GradScaler в FP32 выключены. Квантования в первом контроле нет. Первый FP16-пилот остановлен: в журнале первых 20 шагов 9 пропусков из-за нечисловых градиентов, dev-loss после16 вырос0,46142→0,61658. Это не успешный пилот. FP32 до повторного пилота проверяется отдельным numerics_probe: 3 конечных backward/update, фактическое изменение весов, самый длинный train-backward без обновления. Устойчивость FP32 не объявляется доказанной до реального результата.

```text
python -u -m ml.qwen.numerics_probe --data-root /content/mts_train_ready_v2 --out /content/runs/numerics-unique
python -u -m ml.qwen.train --data-root /content/mts_train_ready_v2 --out /content/runs/pilot-fp32-unique --phase pilot --steps 64 --precision float32 --learning-rate 5e-5
```

Проверяется совпадение completion-only prefix и EOS; loss на входных токенах запрещён. `logits_to_keep` вычисляет logits только для суффикса ответа и предыдущей позиции; его loss сравнивается с обычным masked causal-LM loss на реальной модели перед обучением. Это экономия памяти, а не изменение цели обучения.

Парная диагностика до/после использует первые 2 train и 2 dev (`--baseline-count`), без отбора по правильности ответа. После обучения дополнительно проверяются все 32 train / 8 dev; сравнивать метрики разных объёмов напрямую нельзя. Это проверка процесса, не репрезентативная оценка. Протокол сокращён после реального наблюдения: базовая модель потратила 121,58 с, исчерпала 1536 токенов и стала перечислять каталог; исходный частичный прогон сохраняется отдельно, завершённым обучением не называется. На 64 optimizer steps получаются измерения последних 50, loss и CUDA allocated/reserved. Дополнительно проверяется самый длинный train-пример с backward без изменения весов. Сохраняются best/last, tokenizer, исходники параметров, dependencies/hardware, private raw records, predictions, evaluator metrics, pilot.json и тест нового base+adapter reload.

Основной SFT разрешается кодом только после успешных технических gates, снижения dev-loss, улучшения train fact recall до ≥0,5, многoфактовых ответов, отсутствия ошибок на train32 и успешного reload/long-call memory. Это заранее заданные рабочие gates пилота, не новые правила организаторов. Если они не пройдены, сохраняется неудача; она не называется успешным обучением.

## Основной SFT

```text
python -u -m ml.qwen.train --data-root /content/mts_train_ready_v2 --out /content/runs/main-unique --phase main --pilot-report /content/runs/pilot-unique/pilot.json --budget-hours 4 --full-eval
```

Один epoch всего train, подбор checkpoint по inner-dev loss. Лимит 4 часа — внутреннее ограничение ожидаемого времени обучения по измеренному пилоту с 20% запасом, не гарантия и не официальный ресурсный порог. Генерация и полная оценка занимают дополнительное время.

`--full-eval` сохраняет zero-adapter base-предсказания на полном val до обучения и выбранный checkpoint на том же val после. Декодирование/prompt/catalog одинаковы. Все вызовы присутствуют в официальном prediction JSONL. Failure остаётся `status=failed,facts=null` в raw record, а в evaluator-view даёт ноль фактов и явно считается в failed_generations; это не успешный пустой ответ.

Отчёт различает:

- `metrics`: сырые факты модели до DE-этапов;
- `serving_metrics_without_de_normalizers`: факты после действующего общего Pipeline, включая фильтрацию неподтверждённых agreements;
- число сопоставленных `n_temporal`, fact recall, temporal recall и state accuracy, а также structural failures и неразрешённые agreements.

Улучшение raw extraction не является доказательством качества HTTP-serving до подключения DE и real acceptance.

## Контракт для Full-stack

```text
MWS_PIPELINE_FACTORY=ml.qwen.runtime:factory
MTS_MODEL_DIR=/models/export
MTS_MODEL_DEVICE=cpu
```

Factory возвращает существующий Pipeline с ManagedAdapter. Loader выполняется один раз, проверяет manifest/хеши, локально загружает pinned base и adapter, tokenizer, prompt и каталог, делает структурный smoke. Lifecycle/health/error semantics остаются у FS. Веса base должны быть предварительно размещены в HF cache в том же runtime; serving использует local_files_only=True и не скачивает их при HTTP-запросе. Adapter/tokenizer/prompt/catalog берутся из export; manifest содержит реальные параметры и hashes.

`QwenBridge(engine, temporal_resolver=..., confirmation_resolver=...)` принимает engine с `generate(CallInput)->str` и возвращает существующий CandidateBatch. Вход allowlist исключает gold; original idx/speaker/text и timezone сохраняются. Неизвестное время — QWEN_UNRESOLVED_TIME; resolver DE может вернуть только обоснованную ISO datetime. Дата звонка не подставляется автоматически. Speaker/source mismatch вызывает ошибку; speaker не переписывается. Quote-only выравнивается только при единственном точном совпадении в исходной реплике; при неоднозначной позиции quote/span оба null с warning. Сдвиг source к другой реплике запрещён.

### Что требуется от DE перед real acceptance

1. Обоснованная temporal policy: если valid_from неизвестно, `temporal_resolver(raw_fact, call)` возвращает подтверждённую дату либо оставляет кандидата неразрешённым/ошибкой; никакого механического started_at fallback.
2. Семантическое подтверждение agreement: `confirmation_resolver(candidate, call)` возвращает тот же Fact с проверенными confirmation/evidence_turns. Должно проверяться содержание принятия, его связь с предложением, отрицания и альтернативные предложения. Простое присутствие клиентской реплики не подходит.
3. Catalog/state/polarity/grounding normalizers включаются FS в установленном порядке; их отсутствие явно отражается warning.

Без resolver agreement остаётся unknown; общий Pipeline не экспортирует его. Bridge не выдаёт `confirmed` по одному слову «да» и не объявляет этот пробел решённым. Точный object.id между звонками и исправление невыводимой исторической разметки остаются отдельными задачами команды.

Необходимы отдельные real HTTP/Docker acceptance и CPU/GPU benchmarks. Фиктивный engine в unit-тесте не закрывает эти проверки. Квантование/три точки ресурсов и анализ temporal/state ошибок выполняются после реального SFT.

## Первичные источники реализации

- Transformers v5.3.0: [официальный Qwen3.5 forward и logits_to_keep](https://github.com/huggingface/transformers/blob/v5.3.0/src/transformers/models/qwen3_5/modeling_qwen3_5.py).
- PEFT: [LoRA и выбор целевых модулей](https://huggingface.co/docs/peft/en/developer_guides/lora).
