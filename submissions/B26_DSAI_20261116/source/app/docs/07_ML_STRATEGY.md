# Стратегия ML: аудит → один SFT → три режима

## Аудит существующей Qwen (владелец ML)

В доступном проекте веса не найдены. Получить у команды путь к артефакту; не сканировать весь пользовательский диск. Прочитать config.json, tokenizer/chat_template, generation_config, model index, adapter_config/adapter_model, model card/license и training metadata. Зафиксировать architecture/model_type, base_model_name_or_path, revision, число слоёв/hidden/vocab, dtype, sharding и checksum. Число параметров считать по архитектуре/тензорам с корректным учётом tied weights; отдельно total/active для MoE, trainable adapters и serving total. Не выдавать число LoRA-параметров за размер модели.

Проверить safetensors/другой формат, полный checkpoint или только адаптер; наличие точной base revision, мержился ли старый adapter. Прежний fine-tuning нельзя восстановить надёжно только из весов: неизвестные данные/цель отметить unknown. Проверить происхождение данных и разрешённость использования дообученной до хакатона версии у организаторов. При неизвестном происхождении безопаснее официальный исходный checkpoint; это техническое решение, не утверждение о запрете.

Загрузить на целевом hardware, записать CPU/RAM/GPU/VRAM/driver/CUDA/torch/transformers/PEFT и engine. Проверить русские отрицания, подтверждения, даты, JSON на 20–30 train-derived fixtures; затем inner-dev. Сравнить доступную командную Qwen с точной исходной base без case-SFT, одинаковым prompt. Если она >1.5B, повреждена, требует неизвестного base или непригодна для дальнейшего SFT — не тратить время на pruning.

## Выбор

Кандидат по умолчанию при провале аудита — официальный Qwen3-0.6B, известный существующий компактный decoder; его доступность загрузки и совместимость команды ещё проверить. Подтверждение семейства и non-thinking режима: [model card](https://huggingface.co/Qwen/Qwen3-0.6B). Блокировать reasoning в chat template (`enable_thinking=False` для поддерживающего его checkpoint); для иных Qwen проверять шаблон, не добавлять этот аргумент вслепую. Greedy — наше решение для воспроизводимости извлечения, не рекомендация model card для всех задач.

Qwen3-1.7B существует, но конфликтует с ориентиром ≤1.5B; только исследовательская точка после разрешения, не default. Не предполагать наличие модели ровно 1.0B. Encoder на данном шумном корпусе требует отдельных labels и не является быстрым бесплатным fallback.

## Данные и targets

DE создаёт immutable raw manifest, train-ready JSONL и внутренний group split ~90/10 по account_id; одиночные звонки группируются префиксом call_id. seed=42; все окна одной истории в одной группе. Сохранить точные ID списки, доли по атрибутам/полярности/длине и SHA256. Train-only dev — выбор prompt, правил и гиперпараметров; официальный val — ограниченное подтверждение frozen A/B/C, не источник дополнительных training targets.

Input: started_at, исходные turn idx/speaker/text и краткая онтология. Target: facts в стабильном порядке по source.turn_idx/attribute/value, JSON без reasoning и служебных полей; assistant-only loss, prompt и padding masked. Значения free text допустимы. Не обучать модель угадывать account ID по служебному идентификатору; object.id можно добавлять детерминированно после извлечения. Исходные реплики сохранять; механическое исправление ASR перед извлечением создаёт ошибки grounding.

DE проверяет source bounds, цитаты, странные даты, пустые звонки и near duplicates; сомнительные targets помечает, а не массово переписывает. Val не чистить и не исправлять. Для train span/quote, которые не совпадают с исходником, исключать только ненадёжный supervision этих полей; source.turn_idx и семантический факт сохранять. Все изменения отдельным transform manifest. Нормализация к каталогу после семантического извлечения, не глобальной подстановкой по тексту.

## Pilot и budget

1. Токенизировать train/dev: распределения input/output и доля overflow; выбрать context 4096 как начальную гипотезу, не гарантированный лимит.
2. Overfit 32 качественных примеров за 50–100 шагов: падает ли loss и получается ли валидный JSON; это инженерная диагностика, не оценка качества.
3. Pilot 50 шагов после прогрева с измерением tokens/sec, peak memory и времени validation/save. ETA = оставшиеся шаги × медиана секунд/шаг + eval/export buffer 30–60 минут.
4. Если ETA не укладывается в G2, уменьшить training budget, не запускать три обучения. Нижняя граница полезного объёма определяется inner-dev, не обещанием «часа достаточно».

## Начальная конфигурация SFT (гипотеза для pilot)

Один LoRA run: r=16, alpha=32, dropout=0.05; начать с q_proj/v_proj, наличие модулей проверить; learning_rate=1e-4, warmup_ratio=0.03, weight_decay=0.01, max_grad_norm=1, AdamW; microbatch=1, accumulation=16, effective batch=16. BF16 только при поддержке hardware, иначе FP16 с корректным scaler либо другой dtype. Epochs=1, до 2 только при улучшении inner-dev и наличии времени. seed/data_seed=42, shuffle order и версии сохранить. Eval каждые 200 шагов либо раз в ~30 минут; сохранять best+last и tokenizer/prompt hash. Не включать packing до проверки границ и масок targets.

Ранги/lr — стартовые значения, не найденный оптимум. Если memory pressure — сначала context/микробатч, gradient checkpointing; затем QLoRA при проверенной поддержке. QLoRA — 4-bit frozen base + обучаемые adapters, не full fine-tune 4-bit весов. [PEFT quantization](https://huggingface.co/docs/peft/en/developer_guides/quantization) описывает сочетание PEFT и квантования; поддержка конкретного backend проверяется локально. Не включать quantization на macOS как будто это CUDA.

Память весов: N×bytes/param только нижняя оценка; LoRA, gradients, optimizer, activations, KV и allocator добавляются. Для 0.6B BF16 веса порядка 1.2 GB в десятичных единицах, но это не VRAM требования. Все значения RAM/VRAM получить pilot. Отсутствие GPU → CPU inference и поиск доступного GPU окна, не обещание полного CPU обучения.

## Выбор checkpoint и абляции

На inner-dev сравнить исходную Qwen, case-SFT без правил, case-SFT+catalog/date, затем +state/polarity/grounding. DE отвечает за ошибки правил, ML — сравнительную валидацию. Полный факторный перебор не нужен. Отключить правило, если оно ухудшает temporal/state либо создаёт ложные факты; одинаковая frozen постобработка A/B/C. Oversampling редких отрицаний/ASR augmentation — P1 после основной модели, только train и с проверкой labels; не синтезировать noise так, чтобы смысл и polarity менялись без переразметки.

Выбор: quality gates → schema → fact P/R → стабильность → ресурсы → воспроизводимость. При несопоставимых преимуществах Pareto и решение по 08. Не выбирать по одному loss.

## Экспорт

model_id, base revision, adapter/merged artifact, tokenizer и chat template, prompt, normalizer/catalog hashes, generation config, dependency lock, license/provenance, параметр count, SHA256 и инструкция CPU/GPU. Сначала merge в поддерживаемом dtype (если выбран merged путь), затем quantization; сохранить ссылку на исходный A. Валидировать именно экспортированную/квантованную версию, а не предыдущий in-memory объект. Не терять старую рабочую A до проверки новой.
