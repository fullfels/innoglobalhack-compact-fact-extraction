# Полный backlog

Источник истины по задачам — [tasks.csv](tasks.csv); нижеследующие карточки сгенерированы из тех же записей. Один owner; reviewer не совладелец. P0 — состав MVP/доказательства кейса, P1 — только после P0, P2 — отменять первым. Все deadlines МСК; 16.11 12:00 — внутреннее допущение, официальный срок неизвестен. Оценки — human active hours; SFT/benchmark wall-clock дополнительно ограничены roadmap gates. Если GPU задержалась, сначала переключить план, а не скрывать недовыполнение.

Плановая активная загрузка DE: 17.5 ч P0; дополнительные P1/P2 отдельно.

Плановая активная загрузка ML: 14.5 ч P0; дополнительные P1/P2 отдельно.

Плановая активная загрузка FS: 19.0 ч P0; дополнительные P1/P2 отдельно.

## T01 — Фиксировать требования и регламент

Сохранить требования, официальный дедлайн и baseline resource questions.

- Owner: FS; reviewer: ML; приоритет: P0; оценка: 0.5 ч.
- Зависимости: нет; deadline: 2026-11-13 23:30 +03:00.
- Deliverable: `decision_log.md`.
- DoD: Сверить пять критериев с PDF; неизвестное явно отмечено.
- Проверка: review sources. Риск: R10/R17 (см. 12_RISKS.md).

## T02 — Контракт v1 и fixtures

Утвердить CallInput, Fact, Candidate и ExtractResponse, mock/real interface.

- Owner: FS; reviewer: DE/ML; приоритет: P0; оценка: 1 ч.
- Зависимости: T01; deadline: 2026-11-14 09:00 +03:00.
- Deliverable: `contracts-v1 и fixtures`.
- DoD: Один пример round-trip без facts в features; все участники приняли.
- Проверка: contract review. Риск: R11 (см. 12_RISKS.md).

## T03 — QA исходного корпуса

Проверить hashes, структуру, даты, полярность, source refs, quotes и spans.

- Owner: DE; reviewer: ML; приоритет: P0; оценка: 1.5 ч.
- Зависимости: нет; deadline: 2026-11-14 09:30 +03:00.
- Deliverable: `data_audit.json`.
- DoD: Все 3926+383 звонка проверены; нарушения перечислены, raw неизменен.
- Проверка: сверка с audit/source_audit.json. Риск: R15 (см. 12_RISKS.md).

## T04 — Leakage и inner split

Сгруппировать account/series, exact и near duplicates, seed42 split.

- Owner: DE; reviewer: ML; приоритет: P0; оценка: 1 ч.
- Зависимости: T03; deadline: 2026-11-14 10:30 +03:00.
- Deliverable: `split_ids.json`.
- DoD: Ноль account leakage; спорные близкие шаблоны рассмотрены.
- Проверка: пересечение множеств и review. Риск: R16 (см. 12_RISKS.md).

## T05 — Train-ready v1

Стабильная сериализация targets, masking поля, ненадёжные spans отдельно.

- Owner: DE; reviewer: ML; приоритет: P0; оценка: 1.5 ч.
- Зависимости: T02;T04; deadline: 2026-11-14 12:00 +03:00.
- Deliverable: `train_ready.jsonl+manifest`.
- DoD: Принят ML, hashes и преобразования сохранены; val неизменен.
- Проверка: 20 sample review и schema. Риск: R15 (см. 12_RISKS.md).

## T06 — Аудит имеющейся Qwen

Проверить configs, params, adapter/base, provenance/license, prior SFT.

- Owner: ML; reviewer: FS; приоритет: P0; оценка: 1 ч.
- Зависимости: нет; deadline: 2026-11-14 09:30 +03:00.
- Deliverable: `qwen_audit.json`.
- DoD: Проверены 11 пунктов из запроса; unknown не выдуманы.
- Проверка: metadata+smoke review. Риск: R01/R02 (см. 12_RISKS.md).

## T07 — Hardware и engine smoke

Загрузка на целевом устройстве, память и русские JSON fixtures.

- Owner: ML; reviewer: FS; приоритет: P0; оценка: 0.75 ч.
- Зависимости: T06; deadline: 2026-11-14 10:00 +03:00.
- Deliverable: `hardware.json`.
- DoD: Есть рабочий engine либо явный fallback и отсутствие GPU.
- Проверка: 20-call smoke. Риск: R03/R11 (см. 12_RISKS.md).

## T08 — Baseline команды и исходная base

Сравнить доступную Qwen с base одинаковым prompt на inner-dev.

- Owner: ML; reviewer: DE; приоритет: P0; оценка: 1 ч.
- Зависимости: T02;T07; deadline: 2026-11-14 11:00 +03:00.
- Deliverable: `team_baseline/`.
- DoD: Метрики/JSON rate, revisions и отсутствие доступной base объяснено.
- Проверка: official evaluator+schema. Риск: R01 (см. 12_RISKS.md).

## T09 — Pilot и training budget

32-example overfit, 50-step timing, masking, token lengths/overflow.

- Owner: ML; reviewer: DE; приоритет: P0; оценка: 1.25 ч.
- Зависимости: T05;T07; deadline: 2026-11-14 13:00 +03:00.
- Deliverable: `pilot.json+train_config`.
- DoD: ETA и peak измерены, один основной run укладывается в G2.
- Проверка: loss/decoded targets/timer. Риск: R04 (см. 12_RISKS.md).

## T10 — Основной SFT

Один LoRA/QLoRA по pilot, seeds и eval schedule.

- Owner: ML; reviewer: DE; приоритет: P0; оценка: 1.5 ч.
- Зависимости: T09; deadline: 2026-11-14 18:00 +03:00.
- Deliverable: `adapter+best/last checkpoint`.
- DoD: Обученный checkpoint и журнал доступны; OOM/early stop отражены.
- Проверка: inner-dev eval и reload. Риск: R04/R06 (см. 12_RISKS.md).

## T11 — Первая полная валидация A

383 звонка, preflight IDs и official evaluator.

- Owner: ML; reviewer: DE; приоритет: P0; оценка: 0.75 ч.
- Зависимости: T10; deadline: 2026-11-14 19:00 +03:00.
- Deliverable: `A/predictions+metrics`.
- DoD: Все ID и schema; gates показаны по двум baseline.
- Проверка: full-val и hash audit. Риск: R06/R18 (см. 12_RISKS.md).

## T12 — Каталог и alias rules

Нормализовать после extraction, не склеивать близкие объекты.

- Owner: DE; reviewer: ML; приоритет: P0; оценка: 1.5 ч.
- Зависимости: T02;T03; deadline: 2026-11-14 14:00 +03:00.
- Deliverable: `catalog_normalizer+tests`.
- DoD: Сохранены free-text values и нет global text replacement.
- Проверка: fixtures tariffs/objects. Риск: R07 (см. 12_RISKS.md).

## T13 — Временной normalizer

Якорь started_at, past/future weekdays, month/time ambiguity.

- Owner: DE; reviewer: ML; приоритет: P0; оценка: 2 ч.
- Зависимости: T02;T03; deadline: 2026-11-14 16:00 +03:00.
- Deliverable: `date_normalizer+tests`.
- DoD: ISO и trace, ambiguous policy явно задана.
- Проверка: date fixtures из 10. Риск: R07 (см. 12_RISKS.md).

## T14 — Polarity и подтверждения

Контекст не/да нет/не работает и клиентское подтверждение.

- Owner: DE; reviewer: ML; приоритет: P0; оценка: 1 ч.
- Зависимости: T02;T03; deadline: 2026-11-14 17:00 +03:00.
- Deliverable: `polarity_rules+tests`.
- DoD: Отрицательный симптом не превращён в отрицание факта.
- Проверка: trap fixtures. Риск: R07 (см. 12_RISKS.md).

## T15 — State и grounding

Объектные интервалы и привязка к raw turn/span.

- Owner: DE; reviewer: ML; приоритет: P0; оценка: 2 ч.
- Зависимости: T12;T13; deadline: 2026-11-14 18:30 +03:00.
- Deliverable: `state_grounding+tests`.
- DoD: Два объекта не смешиваются; quote literal либо null.
- Проверка: state/Unicode fixtures. Риск: R15 (см. 12_RISKS.md).

## T16 — Схемы и mock API

/extract /health /model/info и ошибки, is_mock.

- Owner: FS; reviewer: ML; приоритет: P0; оценка: 2 ч.
- Зависимости: T02; deadline: 2026-11-14 11:00 +03:00.
- Deliverable: `API mock endpoints`.
- DoD: Mock runnable до SFT, API не принимает gold.
- Проверка: API contract tests. Риск: R11 (см. 12_RISKS.md).

## T17 — Минимальный UI

Ввод/пример, факты, source highlight, JSON и timeline.

- Owner: FS; reviewer: DE; приоритет: P0; оценка: 2 ч.
- Зависимости: T16; deadline: 2026-11-14 12:00 +03:00.
- Deliverable: `demo page`.
- DoD: Проходит mock сценарий, роли/полярность видны.
- Проверка: manual UI smoke. Риск: R14 (см. 12_RISKS.md).

## T18 — Docker mock и CI

Чистый mock startup, pinned env и tests на PR.

- Owner: FS; reviewer: ML; приоритет: P0; оценка: 1.5 ч.
- Зависимости: T16; deadline: 2026-11-14 12:00 +03:00.
- Deliverable: `Dockerfile+CI+lock`.
- DoD: Один документированный запуск, health/extract проходят.
- Проверка: clean build/smoke. Риск: R12 (см. 12_RISKS.md).

## T19 — Real adapter и worker

Model load lock, bounded queue, timeout и resource config.

- Owner: FS; reviewer: ML; приоритет: P0; оценка: 1.5 ч.
- Зависимости: T07;T16; deadline: 2026-11-14 16:00 +03:00.
- Deliverable: `real adapter+queue`.
- DoD: Модель не дублируется workers; ошибки видны.
- Проверка: load/queue/timeout tests. Риск: R11 (см. 12_RISKS.md).

## T20 — Первая real интеграция

Загрузить первый checkpoint A через общий pipeline и HTTP; финальный serving export проверяется отдельно.

- Owner: FS; reviewer: ML; приоритет: P0; оценка: 1 ч.
- Зависимости: T11;T19; deadline: 2026-11-14 20:00 +03:00.
- Deliverable: `E2E run`.
- DoD: Один реальный запрос с model_id и источником.
- Проверка: real E2E fixture. Риск: R11 (см. 12_RISKS.md).

## T21 — Normalizer ablation

На inner-dev сравнить без правил и с ними; freeze postprocessing.

- Owner: ML; reviewer: DE; приоритет: P0; оценка: 0.75 ч.
- Зависимости: T11;T12;T13;T14;T15; deadline: 2026-11-14 20:00 +03:00.
- Deliverable: `ablation_report.json`.
- DoD: Регрессии отключены одинаково во всех A/B/C.
- Проверка: delta official+diagnostics. Риск: R07 (см. 12_RISKS.md).

## T22 — Экспорт A и hash

Base/adapter/merged, tokenizer/prompt/deps/checksums.

- Owner: ML; reviewer: FS; приоритет: P0; оценка: 0.75 ч.
- Зависимости: T11;T21; deadline: 2026-11-14 20:30 +03:00.
- Deliverable: `A serving artifact`.
- DoD: Reload в отдельном процессе, совпали schema и predictions.
- Проверка: export reload smoke. Риск: R11 (см. 12_RISKS.md).

## T23 — Конфигурация B

INT8 из A, smoke или Plan B use_cache=false.

- Owner: ML; reviewer: FS; приоритет: P0; оценка: 0.75 ч.
- Зависимости: T22; deadline: 2026-11-15 09:00 +03:00.
- Deliverable: `B artifact/config`.
- DoD: Работает на целевом engine, точное отличие описано.
- Проверка: 20-call smoke. Риск: R08/R09 (см. 12_RISKS.md).

## T24 — Конфигурация C

INT4 из A, smoke или Plan B beams=2.

- Owner: ML; reviewer: FS; приоритет: P0; оценка: 0.75 ч.
- Зависимости: T22; deadline: 2026-11-15 09:30 +03:00.
- Deliverable: `C artifact/config`.
- DoD: Нет нового SFT, конфигурация воспроизводима.
- Проверка: 20-call smoke. Риск: R08/R09 (см. 12_RISKS.md).

## T25 — Три сравнительных full-val runs

Одинаковый frozen pipeline/val/evaluator, включая A после freeze.

- Owner: ML; reviewer: DE; приоритет: P0; оценка: 1 ч.
- Зависимости: T22;T23;T24; deadline: 2026-11-15 11:00 +03:00.
- Deliverable: `A_B_C evaluation bundle`.
- DoD: Три набора по 383 ID; raw и final schema rates, hashes.
- Проверка: comparative preflight. Риск: R13 (см. 12_RISKS.md).

## T26 — Benchmark collector

Warm/cold timers, CUDA sync, RSS/VRAM, latency/throughput.

- Owner: FS; reviewer: ML; приоритет: P0; оценка: 2 ч.
- Зависимости: T19; deadline: 2026-11-14 18:00 +03:00.
- Deliverable: `bench harness+raw format`.
- DoD: Контрольный run сохраняет raw samples и hardware.
- Проверка: timer/memory smoke. Риск: R10 (см. 12_RISKS.md).

## T27 — CPU benchmark A/B/C

Один host, одинаковые inputs и 3 повтора; fallback subset общий.

- Owner: FS; reviewer: ML; приоритет: P0; оценка: 1.5 ч.
- Зависимости: T23;T24;T26; deadline: 2026-11-15 11:30 +03:00.
- Deliverable: `cpu resources reports`.
- DoD: Все ограничения измерений описаны, нет CPU/GPU mix.
- Проверка: raw samples review. Риск: R13 (см. 12_RISKS.md).

## T28 — GPU benchmark A/B/C

Одна GPU без обучения параллельно, batch1+throughput profile.

- Owner: FS; reviewer: ML; приоритет: P0; оценка: 1.5 ч.
- Зависимости: T23;T24;T26; deadline: 2026-11-15 13:00 +03:00.
- Deliverable: `gpu resources reports`.
- DoD: GPU-sec/1000 и peak с единицами, либо missing reason.
- Проверка: 3 repeat protocol. Риск: R03/R10 (см. 12_RISKS.md).

## T29 — Диагностические срезы

Отрицания, transitions, traps, длинные и multiobject.

- Owner: DE; reviewer: ML; приоритет: P0; оценка: 1.5 ч.
- Зависимости: T05; deadline: 2026-11-15 10:00 +03:00.
- Deliverable: `slice_ids+definitions`.
- DoD: Детерминированные IDs/определения, не подмена full-val.
- Проверка: distribution review. Риск: R16 (см. 12_RISKS.md).

## T30 — Попарный error analysis

Official matching по gold indices, A/B,A/C,B/C и ручной review.

- Owner: DE; reviewer: ML; приоритет: P0; оценка: 2 ч.
- Зависимости: T25;T29; deadline: 2026-11-15 13:00 +03:00.
- Deliverable: `pairwise_errors.csv`.
- DoD: 30–50 расхождений + shared-correct sanity, категории/источники.
- Проверка: sample adjudication. Риск: R13 (см. 12_RISKS.md).

## T31 — Дельты и графики

Δpp, speedup, size/memory savings, Pareto и оба baseline.

- Owner: ML; reviewer: DE; приоритет: P0; оценка: 1.5 ч.
- Зависимости: T25;T27;T28;T30; deadline: 2026-11-15 15:00 +03:00.
- Deliverable: `comparison.csv+6 graphs`.
- DoD: Все values traceable к manifest, нет вымышленных baseline ресурсов.
- Проверка: formula+plot review. Риск: R10 (см. 12_RISKS.md).

## T32 — Robustness и data report

Train-dev perturbations с QA, static vs temporal ошибки.

- Owner: DE; reviewer: ML; приоритет: P0; оценка: 2 ч.
- Зависимости: T15;T29; deadline: 2026-11-15 16:00 +03:00.
- Deliverable: `stress fixtures+data_report`.
- DoD: Hidden-test успех не заявлен, labels проверены.
- Проверка: robustness suite. Риск: R16 (см. 12_RISKS.md).

## T33 — Финальный выбор и fallback

Gate→Pareto→стабильность; сохранить прошлую A.

- Owner: ML; reviewer: FS; приоритет: P0; оценка: 0.5 ч.
- Зависимости: T31;T32; deadline: 2026-11-15 17:00 +03:00.
- Deliverable: `selection.md`.
- DoD: Объяснён выбор каждой метрикой/ресурсом и unknown.
- Проверка: decision review. Риск: R06/R08 (см. 12_RISKS.md).

## T34 — Real Docker и freeze

Выбранные экспортированные веса, offline restart, CPU/GPU profiles.

- Owner: FS; reviewer: ML; приоритет: P0; оценка: 1.5 ч.
- Зависимости: T20;T33; deadline: 2026-11-15 18:00 +03:00.
- Deliverable: `release image+digest`.
- DoD: Clean machine health/extract, pinned artifacts, schema 100%.
- Проверка: Docker E2E. Риск: R12 (см. 12_RISKS.md).

## T35 — Финальная валидация экспорта

Повторить full-val именно поставляемой версии.

- Owner: ML; reviewer: DE; приоритет: P0; оценка: 0.75 ч.
- Зависимости: T33;T34; deadline: 2026-11-15 18:30 +03:00.
- Deliverable: `final predictions+metrics`.
- DoD: Нет качества от старого in-memory checkpoint.
- Проверка: hash+full-val. Риск: R11 (см. 12_RISKS.md).

## T36 — Идеи развития и блок данных

Реальные ограничения, self-training/verification, attrs/языки.

- Owner: DE; reviewer: ML; приоритет: P0; оценка: 1 ч.
- Зависимости: T30;T32; deadline: 2026-11-15 18:30 +03:00.
- Deliverable: `slides data/errors/future`.
- DoD: Не выдаёт будущие функции за реализованные.
- Проверка: slide review. Риск: R14 (см. 12_RISKS.md).

## T37 — ML отчёт и слайды

Обучение, три точки, benchmark условия, выбор.

- Owner: ML; reviewer: DE; приоритет: P0; оценка: 1 ч.
- Зависимости: T31;T35; deadline: 2026-11-15 19:00 +03:00.
- Deliverable: `ML slides+report`.
- DoD: Каждая цифра со своим run, оба baseline.
- Проверка: report audit. Риск: R14 (см. 12_RISKS.md).

## T38 — Презентация и backup video

Собрать историю и записать real demo 60–90с.

- Owner: FS; reviewer: DE; приоритет: P0; оценка: 1.5 ч.
- Зависимости: T34;T36;T37; deadline: 2026-11-15 20:00 +03:00.
- Deliverable: `deck+video`.
- DoD: Открывается offline, replay подписан.
- Проверка: timed demo. Риск: R14 (см. 12_RISKS.md).

## T39 — Проверка данных сдачи

Hashes/splits/error CSV и инструкции DE.

- Owner: DE; reviewer: ML; приоритет: P0; оценка: 0.5 ч.
- Зависимости: T32;T36; deadline: 2026-11-16 09:00 +03:00.
- Deliverable: `data checklist signed`.
- DoD: Raw/val не изменены, docs совпадают.
- Проверка: manifest review. Риск: R15 (см. 12_RISKS.md).

## T40 — ML bundle сдачи

Веса/код/config/license/metrics/manifests A/B/C.

- Owner: ML; reviewer: FS; приоритет: P0; оценка: 0.5 ч.
- Зависимости: T35;T37; deadline: 2026-11-16 09:00 +03:00.
- Deliverable: `model bundle checklist`.
- DoD: Пути/хэши/команды пригодны другому участнику.
- Проверка: artifact audit. Риск: R11 (см. 12_RISKS.md).

## T41 — Репетиция и чистая поставка

Защита, вопросы, clean startup, комплект сдачи.

- Owner: FS; reviewer: ML/DE; приоритет: P0; оценка: 1.5 ч.
- Зависимости: T38;T39;T40; deadline: 2026-11-16 12:00 +03:00.
- Deliverable: `submission package`.
- DoD: Регламент и все обязательные файлы проверены.
- Проверка: rehearsal+clean smoke. Риск: R14/R17 (см. 12_RISKS.md).

## T42 — Live comparison UI

Сохранённые A/B/C ответы, manifests, без трёх моделей в памяти.

- Owner: FS; reviewer: ML; приоритет: P1; оценка: 1 ч.
- Зависимости: T25;T17; deadline: 2026-11-15 16:00 +03:00.
- Deliverable: `comparison tab`.
- DoD: Не блокирует core demo, результаты явно offline.
- Проверка: UI smoke. Риск: R13 (см. 12_RISKS.md).

## T43 — Paired bootstrap

Account-group resampling с одинаковыми draws для всех.

- Owner: ML; reviewer: DE; приоритет: P1; оценка: 1 ч.
- Зависимости: T25; deadline: 2026-11-15 16:00 +03:00.
- Deliverable: `confidence intervals`.
- DoD: Повторные ID не потеряны через dict, seed сохранён.
- Проверка: bootstrap sanity. Риск: R13 (см. 12_RISKS.md).

## T44 — ASR augmentation

Train-only шум с сохранением semantics/source map.

- Owner: DE; reviewer: ML; приоритет: P1; оценка: 1 ч.
- Зависимости: T05; deadline: 2026-11-14 16:00 +03:00.
- Deliverable: `augmented train v2`.
- DoD: Никакой val contamination, labels проверены.
- Проверка: sample QA. Риск: R16 (см. 12_RISKS.md).

## T45 — Второй короткий SFT/абляция

Только при наличии времени после успешной A, иначе отменить.

- Owner: ML; reviewer: DE; приоритет: P2; оценка: 2 ч.
- Зависимости: T10;T44; deadline: 2026-11-15 11:00 +03:00.
- Deliverable: `optional checkpoint`.
- DoD: Не отнимает budget трёх обязательных точек.
- Проверка: inner-dev eval. Риск: R04/R13 (см. 12_RISKS.md).
