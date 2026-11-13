# Data Engineer workspace

Raw data и производные datasets не входят в Git. Получите десять официальных файлов у команды/организаторов и разместите в `data/raw/` (ignored). Публичное распространение не разрешено автоматически даже при private репозитории.

Проверка целостности из корня:

```bash
python docs/audit/reproduce_audit.py --source data/raw --out /tmp/source-audit.json
```

Сравните SHA256 с `docs/audit/source_audit.json`; JSON/Markdown источники и PDF в Git не копируются. `data/manifests/` — версии и hashes без текстов; `data/processed/` — train-ready (ignored); `data/samples/` — только небольшие независимо написанные синтетические fixtures без исходного корпуса. Для воспроизведения исходного split используются exact IDs/hash/seed, не случайное повторное деление.

Код первичной подготовки добавлять в `scripts/data/`, normalizers — в `src/normalization/` при появлении реализации. Сейчас этих реализаций нет; не создавать альтернативные Fact схемы. Интерфейс `Normalizer` находится в `src/pipeline/adapters.py`; shared модели в `src/contracts/models.py`.

Ожидаемый handoff: train-ready JSONL, split ID lists, transform manifest, QA/error reports и normalizer с fixtures. ML обучение принадлежит ML Engineer. Ссылки: [интеграция](../docs/INTEGRATION_V1.md), [задачи](../docs/06_BACKLOG.md), [workflow](../docs/TEAM_WORKFLOW.md).
