# Risk register и резервные сценарии

Вероятности качественные оценки до hardware pilot, не статистика. Влияние: В — угроза сдаче/критерию, С — локальная задержка.

| ID | Риск | Вероятность / влияние | Ранний сигнал | Mitigation | Fallback и trigger | Owner |
|---|---|---|---|---|---|---|
| R01 | Существующая Qwen не подходит | Средняя / В | Нет config/base, плохой JSON/RU | Аудит до 14.11 10:00 | Официальная компактная Qwen, один SFT | ML |
| R02 | Превышен размер | Средняя / В | total params>1.5B | Считать serving params, не adapter | Не pruning; готовый компактный checkpoint | ML |
| R03 | GPU недоступна | Средняя / В | Нет подтверждённого хоста к 10:00 14.11 | Проверить доступ/окно/VRAM первым делом | CPU inference, доступное команде GPU окно; при отсутствии честно неполная GPU часть | ML |
| R04 | SFT слишком долгий | Средняя / В | Pilot ETA после G2 | ETA по шагам, checkpoints | Один epoch/лимит шагов, last stable; убрать второй run | ML |
| R05 | Невалидный JSON | Высокая / В | Raw schema rate низкая | Краткий target, masked loss, валидация | Проверенный structured decoding если поддержан; помеченный failure без выдуманных facts | ML |
| R06 | Качество ниже базы | Средняя / В | Inner-dev temporal/state слабы | Разбор traps/dates, ablation | Последний лучший A, не обещать threshold; отчёт gap | ML |
| R07 | Normalizer ухудшает | Высокая / С | Ablation дала падение | Conservative rules+trace | Feature flag off для вредного правила, одинаково A/B/C | DE |
| R08 | Квантование теряет temporal | Средняя / В | B/C full-val ниже gates | Одинаковые условия, pairwise dates | Финал A, B/C остаются отрицательным научным результатом | ML |
| R09 | Нет трёх checkpoints | Высокая / С | Budget не позволяет | Не планировать три SFT | Три режима одной модели из 08 | ML |
| R10 | Нет baseline resources | Высокая / В | Организаторы не дали сведения | Рано запросить точный protocol/params | Quality baseline + собственная кривая; resource criterion unverified | FS |
| R11 | Конфигурации несовместимы | Средняя / В | Разные output schemas/runtime | Общий Extractor, smoke до full-val | Один engine, три режима Plan B | FS |
| R12 | Docker не запускается | Средняя / В | Mock smoke/real load fail | Build 14.11, pinned deps, mounted weights | CPU profile, последний рабочий image, backup video с честной маркировкой | FS |
| R13 | Не хватает времени сравнить | Средняя / В | Нет A/B/C к 15.11 13:00 | Зарезервировать полдня eval | Убрать косметику, общий benchmark subset, full quality оставить | ML |
| R14 | Не хватает подготовки защиты | Средняя / В | Слайды пусты к G4 | Разделы пишут DE/ML заранее | Freeze 18:00, короткая защита и записанное demo | FS |
| R15 | Разметка не совпадает с ASR | Высокая / В | Audit mismatch counts | Не учить недостоверные spans, review спорных labels | turn grounding+null span; unchanged val | DE |
| R16 | Подгонка под val/шаблоны | Средняя / В | Изменения после каждого val | Account inner-dev, ограниченные confirm runs | Freeze на train-dev, robustness report | DE |
| R17 | Deadline раньше допущения | Средняя / В | Нет точного регламента | Уточнить первым делом | Сдвиг release на T−6h, снять P1/P2 | FS |
| R18 | Метрики baseline расходятся | Уже подтверждён / В | 459 vs 435 matched | Сохранить hashes и оба набора | Консервативные gates 01, просьба о версии evaluator | ML |

## Сценарии

Plan A: одна обученная компактная модель BF16/FP16 + INT8 + INT4, общая постобработка, полный val и CPU/GPU отчёт. Это основной план, а не три полноценные SFT.

Plan B: одна обученная модель, три режима serving/decoding на совместимом runtime. Переход после 60 минут нерешённой несовместимости квантования. Меньше novelty, зато сравнение воспроизводимо; отличия качества могут быть нулевыми.

Plan C: стабильный рабочий extractor, законченные API/Docker, измерения доступных режимов и честный gap report. Если SFT не завершён — исходная компактная модель как demo, прямо указать отсутствие обязательного обученного артефакта. Replay/video — backup демонстрации, не замена реальным predictions для score. При возможности три режима сохраняются даже в Plan C.

Не экономить время за счёт фальсификации замеров, удаления сложных val examples, правки evaluator или выдачи mock за real.
