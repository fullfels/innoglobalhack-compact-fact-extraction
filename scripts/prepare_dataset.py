import json
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit

def prepare_train_ready_v1(input_file: str, output_dir: str, test_size: float = 0.2, random_seed: int = 42):
    input_path = Path(input_file)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Загрузка сырых данных (предполагается JSONL, где каждая строка — отдельный звонок)
    records = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    print(f"Всего загружено звонков/примеров: {len(records)}")

    # Извлекаем группы (call_id) для разделения
    groups = [r["call_id"] for r in records]

    # 2. Grouped Train/Dev Split
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_seed)
    train_idx, dev_idx = next(gss.split(records, groups=groups))

    train_data = [records[i] for i in train_idx]
    dev_data = [records[i] for i in dev_idx]

    # 3. Проверка на Data Leakage (пересечение call_id между train и dev должно быть пустым)
    train_calls = set(r["call_id"] for r in train_data)
    dev_calls = set(r["call_id"] for r in dev_data)
    overlap = train_calls.intersection(dev_calls)

    assert len(overlap) == 0, f"КРИТИЧЕСКАЯ ОШИБКА: Обнаружена утечка данных! Пересекающиеся call_id: {overlap}"
    print("✓ Проверка на Data Leakage успешно пройдена: пересечений по call_id нет.")

    # 4. Сохранение результатов
    train_file = out_path / "train.jsonl"
    dev_file = out_path / "dev.jsonl"

    with open(train_file, "w", encoding="utf-8") as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    with open(dev_file, "w", encoding="utf-8") as f:
        for item in dev_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # 5. Генерация манифеста
    manifest = {
        "version": "v1",
        "total_samples": len(records),
        "train_samples": len(train_data),
        "dev_samples": len(dev_data),
        "train_calls_count": len(train_calls),
        "dev_calls_count": len(dev_calls),
        "leakage_checked": True,
        "split_strategy": "GroupShuffleSplit by call_id",
        "random_seed": random_seed
    }

    manifest_file = out_path / "manifest.json"
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"Готово! Артефакты сохранены в {out_path.resolve()}:")
    print(f" - {train_file.name} ({len(train_data)} строк)")
    print(f" - {dev_file.name} ({len(dev_data)} строк)")
    print(f" - {manifest_file.name}")

if __name__ == "__main__":
    prepare_train_ready_v1("data/raw_calls.jsonl", "data/train_ready_v1")