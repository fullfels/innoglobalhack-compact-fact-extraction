# Протокол сравнения качества и ответов

## Freeze

Вход full-val: 383 звонка, hash из source_audit.json. Один список call_id, одинаковый порядок исполнения, hardware и postprocessing. Seed=42, greedy для основного precision-эксперимента; отдельные режимы документировать. Зафиксировать evaluator hash из 01. Из prediction нельзя извлекать входы: использовать исходный val, предварительно удалить facts и связанные gold-only поля. Не подключать ground truth к retry или normalizers.

До запуска evaluator: JSONL parse, отсутствие duplicate IDs, set(pred IDs)==set(val IDs), ровно один output на звонок, корректная схема, типы, source idx, ISO и bounds. Ошибочные/пропущенные outputs учитывать в coverage и failure rate; не исключать сложные звонки из знаменателя. Сам evaluator не ловит все эти нарушения.

## Сравнение на уровне факта

Для каждой конфигурации и call_id вызвать неизменный `match_facts(gold_facts,pred_facts)`. Каждому gold присвоить устойчивый индекс `(call_id,gold_index)` и сохранить соответствие pred_index/score. Не сравнивать JSON-строки или просто множества value: порядок полей несущественен, одинаковое значение может относиться к разным состояниям.

Для каждой пары A/B, A/C, B/C вычислить множества matched gold indices: intersection=правильно извлечены обеими; left-only=только первой; right-only=только второй; neither=gold минус union. Здесь «правильно» относится к official fact matching, а не автоматически к датам/источнику. Для matched отдельно записать date_from/date_to correct, turn correct, span overlap и semantic object check. Так видны факты, у которых содержание верно, а временная часть потеряна.

False positives — все несопоставленные pred indices; false negatives — несопоставленные gold. Их попарное diagnostic matching по атрибуту/сходству value с ослаблением polarity использовать только для объяснения «неверная полярность», не как TP official. Сохранять original indices, не сериализовать по одному value. При равных fuzzy scores официальный greedy tie зависит от исходного порядка: predictions сохранять в установленном inference-порядке, evaluator не пересортировывать ради улучшения. Permutation sensitivity отражать отдельно.

DE вручную размечает 30–50 расхождений с приоритетом temporal, отрицаний и нескольких объектов; дополнительно 10 обоих-правильных для sanity. Проверить все редкие polarity=-1 val как диагностический срез, без обучения на нём. Категории multi-label: FP trap/unconfirmed, FN, wrong object, polarity/scope, start/end date, lost state transition, wrong source, catalog collision, truncation, parse/schema, annotation concern. Спорный gold — annotation concern, но official score остаётся исходным.

Формат error CSV: pair, call_id, gold_index, pred_index_left/right, attribute, category, correct_left/right, date_left/right, source_left/right, short evidence, reviewer, notes. Не публиковать полный диалог в презентации без необходимости. Автоматический отчёт связывает каждую строку с run manifest.

## Дельты и неопределённость

Для метрики m: Δpp(B,A)=100×(mB−mA). Относительное изменение %=100×(mB/mA−1) отдельно, при mA=0 — null/reason. Размер reduction%=100×(1−sizeB/sizeA); memory аналогично; speedup=latencyA/latencyB; GPU saving%=100×(1−GB/GA). Для качества/стоимости: GPU-seconds per 1000 / Q, только Q>0, помечать внутренним индексом. Числа в долях и pp не смешивать.

Optional paired bootstrap 1000 повторов seed=42, resampling целых account groups с сохранением всех звонков группы и одинаковых выборок для A/B/C; метрики пересчитывать по фактам внутри выборки, не средним per-call. Переименовывать повторно выбранные ID, чтобы dict не удалил bootstrap repeats. Это неопределённость на val, не гарантия hidden-test. Temporal denominator меняется между моделями: рядом обязательно matched count и fact recall, дополнительно temporal-correct/all-gold.

## Baseline

Сравнить quality A/B/C с пересчитанным baseline одним evaluator; дополнительно колонка опубликованной базы и обе delta. Не скрывать расхождение. Параметры, CPU/GPU latency и memory baseline неизвестны. Из baseline_pred нельзя восстановить runtime. Качество baseline уже измерено, зато speedup к baseline пока null. Не сравнивать время локальной Qwen с временем чтения baseline JSONL.

## Таблица отчёта

| Метрика | Baseline опубликован | Baseline пересчитан | A | B | C |
|---|---:|---:|---|---|---|
| Parameters | неизвестно | неизвестно | не измерено | не измерено | не измерено |
| Fact precision | 0.2074 | 0.2189 | не измерено | не измерено | не измерено |
| Fact recall | 0.2358 | 0.2488 | не измерено | не измерено | не измерено |
| Temporal recall | 0.4161 | 0.4205 | не измерено | не измерено | не измерено |
| State accuracy | 0.3483 | 0.3194 | не измерено | не измерено | не измерено |
| Grounding turn | 0.6759 | 0.6906 | не измерено | не измерено | не измерено |
| Grounding span | null | null | не измерено | не измерено | не измерено |
| CPU / GPU mean,p50,p95 | неизвестно | неизвестно | не измерено | не измерено | не измерено |
| RAM / VRAM peak | неизвестно | неизвестно | не измерено | не измерено | не измерено |
| GPU-seconds/1000 | неизвестно | неизвестно | не измерено | не измерено | не измерено |

«Не измерено» здесь фиксирует стадию проекта, а не пустой шаблон вместо плана. После run values поступают автоматически из manifest/metrics/resources, не руками по памяти.
