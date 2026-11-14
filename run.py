import sys
from pathlib import Path

# Принудительно добавляем корень проекта в начало путей Python
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

# Выводим отладочную информацию в консоль перед импортами
print("--- ОТЛАДКА ПУТЕЙ ---")
print("Путь к run.py:", __file__)
print("Корень проекта:", current_dir)
print("Папка src существует?", (current_dir / "src").exists())
print("---------------------")

# Пробуем импортировать
import json
from src.contracts.models import CallInput
from src.pipeline.adapters import MockExtractor
from src.pipeline.service import Pipeline
from src.pipeline.normalizers import (
    CatalogNormalizer,
    TemporalNormalizer,
    StateAndGroundingNormalizer,
)

def main():
    print("=== Пайплайн успешно импортирован! Запуск... ===")
    
    catalog_path = current_dir / "catalog.json"
    catalog_data = {}
    if catalog_path.exists():
        with open(catalog_path, "r", encoding="utf-8") as f:
            catalog_data = json.load(f)
        print("[+] Каталог успешно загружен")
    else:
        print("[!] catalog.json не найден, используется пустой каталог.")

    normalizers = (
        CatalogNormalizer(catalog_data),
        TemporalNormalizer(),
        StateAndGroundingNormalizer(),
    )

    extractor = MockExtractor()
    pipeline = Pipeline(extractor=extractor, normalizers=normalizers)

    sample_call = CallInput(
        call_id="call_test_001",
        started_at="2026-03-30T10:00:00",
        turns=[
            {"idx": 1, "speaker": "client", "text": "Интернет не работает."}
        ]
    )

    prediction, warnings = pipeline.run(sample_call)
    print("\n--- Результат успешного выполнения ---")
    print(f"Prediction: {prediction}")
    print(f"Warnings: {warnings}")

if __name__ == "__main__":
    main()