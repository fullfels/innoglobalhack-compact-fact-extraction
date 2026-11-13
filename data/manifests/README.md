# Dataset manifests

Источник текущих SHA256 — `docs/audit/source_audit.json`. Новый manifest содержит dataset_version, raw hashes, transform git SHA/config, train/dev group split seed и ID hash, counts, schema_version. Исходные данные не хранить в manifest. Изменение transform создаёт новую версию. Официальный val неизменен.
