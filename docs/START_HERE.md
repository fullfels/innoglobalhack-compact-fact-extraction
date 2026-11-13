# С чего начать

Это план реализации, а не реализованное приложение. Подготовлены 15 Markdown-документов, 45 задач в CSV и воспроизводимый аудит десяти источников. Обучение и разработка основных компонентов не запускались. Основные результаты аудита: [01_CASE_ANALYSIS.md](01_CASE_ANALYSIS.md); измеренные данные и hashes — [source_audit.json](audit/source_audit.json).

## Ключевое решение

Одна компактная Qwen, один case-SFT и общая постобработка; A=BF16/FP16, B=INT8, C=INT4 при поддержке hardware. Это три режима одного checkpoint, а не три независимых обучения и не сокращение логических параметров квантованием. Резерв — три serving/decoding режима из 08. Сначала контракт и mock E2E, затем real-модель. Неподтверждённый ресурсный baseline — отдельная проблема доказательств, не повод остановить разработку.

## Читать по порядку

1. [01 — кейс и аудит](01_CASE_ANALYSIS.md), [14 — неизвестное](14_OPEN_QUESTIONS.md): факты, противоречия, допущения.
2. [02 — спецификация](02_TECHNICAL_SPEC.md), [03 — архитектура](03_ARCHITECTURE.md): согласовать контракт v1.
3. [04 — ответственность](04_TEAM_RESPONSIBILITIES.md), [05 — roadmap](05_ROADMAP.md), [06 — backlog](06_BACKLOG.md), [tasks.csv](tasks.csv): взять задачи своей зоны.
4. [07 — ML](07_ML_STRATEGY.md), [08 — три конфигурации](08_MODEL_EXPERIMENTS.md), [09 — сравнение](09_COMPARISON_PROTOCOL.md).
5. [10 — тесты и метрики](10_TESTING_AND_METRICS.md), [11 — ресурсы](11_BENCHMARKING.md), [12 — риски](12_RISKS.md), [13 — demo](13_DEMO_AND_PITCH.md).

## Первые 2–4 рабочих часа DE

- [ ] 0–45 мин: прочитать README/schema/catalog и аудит. Вход: все raw JSONL. Выход: подтверждение stats, список аномалий; готово, когда проверены source/quote/date и 10 спорных примеров, raw не изменён (T03).
- [ ] 45–90 мин: account grouping+near duplicate review. Вход: train/val и series. Выход: inner split IDs+seed42; готово при нуле account leakage (T04).
- [ ] 90–150 мин: train-ready v1 по контракту. Вход: только train для targets. Выход: JSONL+transform manifest, отдельно ненадёжные spans; готово после review ML 20 примеров (T05).
- [ ] 150–240 мин: fixtures dates/polarity и catalog aliases. Вход: train/catalog, исходные turns. Выход: однозначные expected values и ambiguity policy; не править официальный val (T12–T14).

DE не пишет training loop и не подбирает lr/rank. Дополнительно его собственная зона: нормализация, QA, error analysis, данные/ошибки в презентации.

## Первые 2–4 рабочих часа ML

- [ ] 0–60 мин: получить путь имеющейся Qwen; проверить configs/base/adapters/params/license/prior SFT. Выход: qwen_audit с unknown полями; готово после решения keep/replace (T06).
- [ ] 60–105 мин: hardware/engine load и 20 русских fixtures. Вход: checkpoint, целевой CPU/GPU. Выход: hardware.json, peak и schema rate; готово если есть рабочий путь или конкретный fallback (T07).
- [ ] 105–165 мин: исходная Qwen/base на train-derived dev, baseline evaluator повторить. Выход: manifests и метрики; готово без утечки facts в prompt (T08).
- [ ] 165–240 мин: принять train-ready, tokenizer lengths/masking, 32-example overfit и 50-step pilot. Выход: training config, ETA, memory; готово при обоснованном budget одного SFT (T09).

Дальше один SFT, экспорт A, B/C без retraining, полный val и графики. Не начинать три обучения до проверки бюджета.

## Первые 2–4 рабочих часа FS / Tech Lead

- [ ] 0–30 мин: зафиксировать реальное T0/официальную сдачу, проверить roadmap. Выход: decision log и список внешних вопросов; готово при явных unknown, без ожидания ответов для независимых задач (T01).
- [ ] 30–75 мин: общий CallInput/Fact/Candidate/envelope и fixtures. Вход: schema.md/02/03. Выход: контракт v1, принятый DE/ML (T02).
- [ ] 75–165 мин: FastAPI mock /extract,/health,/model/info и ошибки. Вход: fixtures, без весов. Выход: работающий API и tests; is_mock=true обязателен (T16).
- [ ] 165–240 мин: Docker mock smoke и UI skeleton с примером, таблицей/источником/JSON. Выход: первый проверяемый E2E; дальнейший UI finish в G1 (T17/T18).

FS не берёт на себя обучение, очистку, ML графики и весь анализ ошибок. Его зона — архитектура, контракты, API/UI, интеграция, эксплуатационные замеры и финальная сборка защиты.

## Контрольные точки

14.11 12:00 mock E2E; 14.11 20:00 real обученная A; 15.11 13:00 три полных точки; 15.11 18:00 freeze; 16.11 10:00 clean startup; внутренний пакет 16.11 12:00 МСК. Точная официальная сдача неизвестна — проверить раньше остальных организационных задач. При более позднем фактическом старте пересчитать ёмкость, снять P2/P1.

## Критическая оценка

1. Успеть реально для узкого MVP при рабочей GPU и старте 14.11 утром; полный исследовательский scope без сокращений не гарантирован. До внутренней сдачи около 61 календарного часа от начала аудита, работа не непрерывная.
2. Самый рискованный этап — пригодный checkpoint+hardware+один SFT с temporal/state качеством; внешнее доказательство ресурса baseline тоже может остаться недоступным.
3. Нужен один bounded LoRA/QLoRA SFT после pilot, стартово один epoch; фактическое число шагов/часов определяют lengths, speed и inner-dev, не выдуманная оценка.
4. Три сопоставимых режима получить реалистично без трёх checkpoints; одинаковое качество не делает эксперимент неудачным.
5. Не pruning неизвестной Qwen. Если она мала и подходит — адаптация и precision режимы; если велика/неизвестна — готовый компактный checkpoint. Три размера дороже и не обязательны.
6. Первый минимум — mock API+Docker+один fixture; первый содержательный ML минимум — компактная real-модель→валидные facts→источник→official eval до последнего дня.
7. Наибольший ожидаемый вклад: quality gates и доказанная экономия, 100% schema, три честные точки с error analysis, reproducible image и ясная защита. За конкретные баллы обещать нельзя.
8. Снять сначала pruning/distillation, второй SFT, live multi-model UI, сложный orchestration, расширенную augmentation, затем bootstrap. Не снимать обязательное сравнение, schema/IDs checks, базовый Docker и репетицию.
