# PR #52 — independent review and serving handoff

Дата: 2026-11-14. Проверен head `9701bbbb0d44ec5f73d6e83723dc780d0419a19a`, автор QFa1, base main `c8911a99936de3d7a06c5327e55abcc5fa28694e`.

## Результат review

[Request changes](https://github.com/fullfels/innoglobalhack-compact-fact-extraction/pull/52#pullrequestreview-5903446525) оставлен по одному воспроизведённому P2: verification исходников не исполняется. PR остаётся архивным ML smoke, не production bridge. Ни один PR не объединён, T07/T08/T20 не закрыты. Файлы ML Engineer не редактировались.

| PR | Head | Base | Проверенный CI | Conflicts |
|---|---|---|---|---|
| #46 | 4bc3959 | main | success | нет |
| #47 | 7323902 | main | success | нет |
| #48 | 898477a | main | success | нет |
| #49 | b74a3e1 | fs/evaluation-runner | success | нет |
| #50 | 6eadbd8 | fix/review-followups | success | нет |
| #51 | 26f5c23 | fs/benchmark-collector | success | нет |
| #52 | 9701bbb | main | [success, 43579594200](https://github.com/fullfels/innoglobalhack-compact-fact-extraction/actions/runs/43579594200) | нет |

Все семь PR открыты; других открытых ML/DE PR при проверке нет. main не сдвинулась. #46–#51 уже входят в fs/deep-integration-audit. #52 проверен отдельно во временном каталоге вместе с контрактами и существующим evaluator runner; архивный source не добавлен в serving и не объединён в snapshot. Командные peer review и отдельное разрешение владельца остаются merge gates. После исправления #52 повторить проверку нового SHA; зелёный generic mock CI не проверяет происхождение ML outputs.

## Provenance: что подтверждено

`predict.py`, `common.py` и notebook в Git имеют LF; hashes исполненного архива в manifest/provenance соответствуют CRLF. Это объяснимое изменение только EOL: восстановление CRLF дало **точное совпадение** для всех трёх. `prompt.txt` совпал без преобразований. Никакой другой normalisation (strip/comments/JSON reformatting) не применялась.

Но `verify_metrics.py:21–27` проверяет только official input hashes. Независимое воспроизведение: добавить строку `# independent tamper sentinel` в source/common.py временной копии → verify_metrics.py всё равно exit 0 / verified=true. README/verification.json утверждают source verification, которую этот воспроизводимый verifier не выполняет. Сохранённые fingerprints не нужно менять; ML должен добавить проверку и ясно описать Git bytes versus executed EOL.

FS добавил отдельную неисполняющую ML проверку и 7 synthetic tests:

```bash
python -m scripts.verify_archived_provenance --archive /path/to/PR52/experiment
python /path/to/PR52/experiment/verify_metrics.py --data /path/to/original/CaseData
```

Первый шаг проверяет predict/common/prompt/notebook, выдаёт оба hashes и exact-bytes либо git-LF-to-executed-CRLF policy. Содержательная mutation любого из четырёх файлов отклоняется; prompt не нормализуется. Тесты проверяют LF/CRLF, exact bytes, четыре mutations и запрет скрытого изменения EOL prompt. Второй шаг — существующий ML verifier метрик. PR #52 ещё должен связать provenance verification со своими инструкциями/verification claims; наличие FS helper само по себе не исправляет PR.

Эта проверка доказывает только согласованность файлов относительно сохранённых manifests. Приватные input/results ZIP отсутствуют локально: их digest, raw per-call logs, GPU execution, latency/VRAM и источник самого manifest независимо не аутентифицированы. Notebook без outputs/execution counts; по содержимому это архивный standalone probe. Его upload cell проверяет official data hashes, а не исполняемый source ZIP целиком.

## Метрики и входы

Обе таблицы независимо пересчитаны оригинальным case_metrics.py. Список совпадает с **первыми 10 исходными val call_id** без duplicates/missing/extra. Qwen predictions: 8 facts, gold: 54, state cases: 16. Существующий строгий `src.experiments.evaluate` также принял сохранённые predictions и воспроизвёл metrics; mode=unknown, experiment_id=pr52-saved-predictions-recompute, model_id=archived-Qwen3.5-0.8B-not-serving. Это пересчёт сохранённых данных, не новая генерация.

| Metric | Archived Qwen | Baseline на тех же 10 calls |
|---|---|---|
| fact_precision | 0.2500 | 0.1613 |
| fact_recall | 0.0370 | 0.1852 |
| temporal_recall | 0.0000 | 0.4000 |
| state_accuracy | 0.0625 | 0.3750 |
| grounding_turn | 0.5000 | 0.7000 |
| grounding_span | null | null |

Summary сообщает 8 accepted/2 failed; пустые facts этих двух вызовов — явно обозначенное соглашение исторического experiment, не допустимый API fallback. Raw records отсутствуют, поэтому причину каждого failure и timing statistics можно сверить только с переданным ML отчётом, не заново вычислить из 10 per-call records. p50 исходного кода выбирает верхний средний sample при чётной длине; это не интерполированная statistical median. Для реальных сравнений использовать общий benchmark collector.

На этом диагностическом срезе temporal loss к same-slice baseline — 0.4000, state loss — 0.3125: качество не даёт оснований утверждать прохождение gates. Формальную приёмку по full validation/hidden test на 10 calls делать нельзя. Сопоставимых ресурсных измерений teacher нет; ускорение/экономия не доказаны.

Проверены prompt builder и независимая synthetic sentinel fixture: в model_input передаются только call_id/started_at/turns; gold facts не попадают в messages. Preview с gold создаётся уже после генерации для приватного просмотра. Dataset/evaluator/catalog originals в PR отсутствуют, notebook outputs очищены, больших весов нет. Проверка известных private-key markers не нашла совпадений; это ограниченная проверка, не абсолютная гарантия отсутствия секретов. Prediction JSONL — model outputs, не raw input transcript. Официальные оригиналы не менялись.

## Serving: точные блокеры

Архивный decode_and_validate не подключён и не должен импортироваться production backend:

- Он допускает valid_from=null; serving Fact требует ISO datetime. Не подставлять started_at автоматически: ML/DE должны дать обоснованную дату либо явную ошибку/согласованную ambiguity policy.
- Quote без span и span без quote допускаются историческим parser, но строгий API gate требует согласованную пару либо оба null.
- Confirmation/evidence отсутствуют в historical output. Agreement требует confirmed и клиентское evidence принятия; нельзя автоматически пометить всё confirmed.
- Multiple facts обрабатываются циклом; return идёт только после проверки всего списка. Speaker/source mismatch отклоняется. Object ID, extra/nested fields и JSON duplicate-key policy требуют общей Pydantic revalidation.
- Archived failure → facts=[] остаётся только в историческом scorer protocol. Реальный bridge возвращает CandidateBatch только при успехе, иначе PipelineError; decode/load/context/OOM не маскируются.

Нужна передача ML в своей зоне src/ml: loader, inference bridge и factory module (имена файлов согласовать с ML). Loader(ModelResources) → LoadedModel(inference, close); bridge(CallInput, ModelResources) → CandidateBatch. API factory() → Pipeline; evaluation factory(config) → Pipeline; benchmark factory(config/device) включает load до return BenchmarkTarget и отдаёт close callback. Loader загружает один раз, не при каждом запросе. Descriptor real, revision/hash/context/dtype/manifest относятся к фактически загруженному artifact; ModelResources и tokenizer не публикуются HTTP.

От ML нужны локальные exported checkpoint shards/config (либо base+LoRA adapter), tokenizer/vocabulary/chat template/special tokens, files.sha256 и revision/license/provenance, dependency lock с проверенным CPU/CUDA profile, context/stop/window policy и offline reload instructions. Одна ссылка Qwen/HF revision не является локальным проверенным checkpoint. Ничего не скачивалось и обучение не запускалось.

От DE нужны catalog → temporal → state → polarity → grounding normalizers с normalize(CandidateBatch, CallInput), catalog/hash, fixtures aliases/ambiguity/year boundaries/negation/agreement/history/multiobject/Unicode. Реализаций src/normalizers пока нет. Их бизнес-правила FS не дублировал.

## Выполненные проверки и следующий запуск

- Полный integrated pytest с original evaluator и Docker sentinel: **154 passed, 0 skipped**, 2 upstream deprecation warnings, 5.95 s.
- Frontend: **5 passed**.
- Новые provenance tests: **7 passed**; отдельно actual PR #52 files согласованы по EOL, а mutation воспроизведённо отклонена новым helper.
- Archived metrics verifier: exit 0; строгий existing evaluation runner: complete/valid, все 10 calls. Gold/predictions/metrics/manifest/validation_report сохранены только локально во временном каталоге.
- Mock production Docker build/startup, Compose и HTTP smoke: **PASS**; real Docker ещё требует ML runtime и factory.
- Настоящий inference/новая latency/VRAM/full-383: **не выполнялись**. T20 не выполнена. Сохраняются load-once, timeout/slot ownership и cancellation tests existing snapshot.

Исправление архивного provenance может идти параллельно разработке serving package. Для serving принять проверенный ML package, затем [FIRST_REAL_INFERENCE](FIRST_REAL_INFERENCE.md): offline reload → один HTTP запрос с real metadata и schema/source gate → 10-call диагностика с generation/schema/date/source error counts → 383 calls original evaluator. До передачи DE только явно raw-model experiment. На каждом этапе сохранять artifacts и не пропускать ошибочные calls. A/B/C — после первого рабочего real pipeline и бюджета ресурсов, с одинаковыми dataset/evaluator/postprocessing/hardware protocol; никаких трёх обучений автоматически.

Итог: архитектурные точки готовы; исторические 10-call metrics воспроизведены; serving Qwen пока не подключена. Запустить один настоящий транскрипт сейчас нельзя без checkpoint/tokenizer/runtime/loader/bridge. Полный DE pipeline дополнительно зависит от normalizers. Включать исторический parser для обхода этих зависимостей нельзя.
