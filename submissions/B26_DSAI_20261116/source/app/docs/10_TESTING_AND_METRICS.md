# Тестирование и точная семантика метрик

## Что реально делает case_metrics.py

Fact precision=M/P, recall=M/G по всем фактам, где M — число greedy one-to-one пар. Attribute равен точно; polarity сравнивается знаком (>0), поэтому внешняя схема обязана ограничить ±1. Value: совпадение строк после strip/lower либо жадное выравнивание токенов в порядке, порог 0.85; токен совпадает по SequenceMatcher≥0.75. Ё заменяется на е при токенизации. Это не semantic similarity.

В object matching есть существенная особенность: `" ".join(name)` вставляет пробел между символами, затем берётся первый токен. В большинстве русских названий сравнение фактически сводится к первому символу имени, а не всему имени. Порог NAME_SIM=0.80; условие `(osim>=NAME_SIM or not osim)` также пропускает zero similarity, а пустое tokenized имя даёт osim=1. Поэтому object matching очень слаб. Не использовать эту особенность для подмены object.name; выводить правильные объекты и показывать отдельный semantic object diagnostic.

Пары сортируются по vsim×osim, greedy занимает каждый gold/pred один раз; ties зависят от порядка. Даты, speaker, object.id/type и grounding в fact matching не участвуют. Совпавший fact может иметь ошибочный источник или дату.

Temporal recall=число matched facts с правильными ОБЕИМИ датами / M. Проверяются первые 10 символов valid_from/valid_to, null равен null. Учитываются все matched факты, не только переходы. Пропущенные факты из denominator исключены. Таким образом, название не соответствует recall по всем временным gold. n_temporal=M, а не количество всех gold с датой. Отдельной temporal_precision в возвращаемом JSON нет, хотя docstring её упоминает.

State: gold группируется по `(object.id,attribute)` для service_state/tariff/ticket_status с valid_to=null; берётся максимальный строковый valid_from. Но pred candidates фильтруются только по attribute и valid_to=null, БЕЗ object filter и без polarity. Один последний pred может оцениваться против нескольких gold objects. Валидность актуальности относительно started_at дополнительно не проверяется. Это ошибка/ограничение метрики, не желательное поведение системы. Наша reconciliation остаётся по реальному объекту, а дополнительная диагностика — object-aware.

Grounding turn=правильный source.turn_idx / M. Span denominator включает только matched с правильным turn и непустыми spans ОБЕИХ сторон. Успех span — любое непустое пересечение, не IoU и не exact match. Возвращается null, если таких пар нет. Не выдавать высокий span score на маленьком покрытии за качественное общее grounding; показать coverage и exact substring diagnostics.

Пустые predictions: precision=recall=0; temporal и grounding=null; state=0 при n_state>0. Пропущенные call_id воспринимаются как пустые facts. Лишние pred call_id вообще не обходятся; duplicates в CLI молча перезаписываются dict comprehension. Поэтому обязательна preflight проверка IDs/числа строк, а не только evaluator exit code. Плохие типы могут вызвать исключение; schema validation делать до запуска.

Официальный скрипт не менять. Обсудить особенности с организаторами; при новой версии сохранить старую, зафиксировать новый hash и пересчитать ВСЕ A/B/C+baseline, не смешивать старые и новые результаты.

## Дополнительные диагностики

F1 по P/R; temporal_correct/all_gold (штрафует пропуски); object-aware state; state transition recall; speaker accuracy; span coverage/exact quote rate; rates JSON parse, output schema и final pipeline success; raw vs postprocessed quality; доли truncation/retry/OOM. Все отдельно от official metrics. Macro per-call и micro по фактам не смешивать.

## Матрица тестов реализации

| Группа | Owner | Конкретная проверка | Условие прохождения |
|---|---|---|---|
| Data schema | DE | JSONL, unique IDs, attr/polarity/date, turn refs, spans | Отчёт всех нарушений, исходники не изменены |
| Leakage | DE | account split, hash exact и near duplicates | Ноль account пересечений inner-train/dev; спорные шаблоны рассмотрены |
| Dates | DE | вчера, прошлый месяц, weekday, после 14, year rollover | Детерминированные expected ISO; ambiguity имеет warning |
| Polarity | DE | «не работает», «не использую», «да нет, Плюс» | Правильный scope, отвергнутый тариф не создаётся |
| States | DE | два service objects, смена тарифа, equal dates, future agreement | Только релевантный объект закрывается; устойчивый tie rule |
| Grounding | DE | Unicode, ASR, quote до шума, отсутствующий idx | exact span либо null; исходный текст сохранён |
| Training smoke | ML | 32 examples, assistant mask/padding, reload adapter | Нет утечки target, loss адекватен, inference загружается |
| Model validation | ML | все 383 ID + evaluator | Метрики и denominators сохранены, failures не удалены |
| API | FS | valid/invalid input, duplicate idx, oversize, unloaded/queue/timeout | 200/422/413/503/429/504 по контракту, нет скрытых ошибок |
| Integration | FS | mock и real через общий pipeline | envelope/schema одинаковы, is_mock правильный |
| Docker | FS | clean startup с mounted model, health, один extract, offline restart | Нет скачивания неизвестной latest версии при каждом запуске |
| Performance | FS | холодный/тёплый, sync, memory, input strata | Raw samples воспроизводимы, нагрузка и batch указаны |
| Comparative | ML | count/ID/hash evaluator/config equality | Одинаковые 383 calls, три manifests, нет смешанных runs |
| Robustness | DE | зашумлённые train-dev копии, ловушки, длинные звонки | Labels проверены, degradation reported, original val неизменен |

Date fixtures: для 2026-06-27 «вчера» → 2026-06-26T00:00:00; подтверждённая встреча «во вторник» → 2026-06-30T00:00:00 при future intent. «Со вторника» → ближайший прошлый вторник с явно принятой конвенцией same-day. «В прошлом месяце» не имеет точного дня: проверить принятые gold конвенции по train и документировать, а не выдавать первый день за общеязыковой факт. «После двух» без день/ночь/даты требует контекста; пример ожидается с warning при неоднозначности.

## Уровни проверки

Каждый PR: unit/contracts/mock, без GPU. Каждый handoff: один real fixture и загрузка артефакта. G2/G3: полный val и preflight. G4/G6: real Docker E2E, schema 100%, official metrics экспортированной версии. Один завершённый run не перезапускать без изменения версии или обнаруженной ошибки.

## Срезы

DE формирует по train-derived правилам: отрицания, переходы, agreement confirmation, несколько объектов, operator traps, относительные даты, длинные звонки, неподтверждённые гипотезы. Val slices используются для анализа, не изменения эталонов. Искусственный adversarial benchmark отдельно от official val; часть samples проверить вручную, поскольку удаление «не» меняет label. Скрытый adversarial недоступен, устойчивость на нём пока не доказана.
