#!/usr/bin/env python3
"""Шаг 2: Извлечение точных паттернов эталона из train.jsonl."""
import json
import sys
from collections import Counter, defaultdict
from case_metrics import _tokens


def inspect_train_patterns(train_path: str, cat_path: str):
    cat = json.load(open(cat_path, encoding="utf-8"))
    cat_exact = set()
    for group in cat.values():
        for item in group:
            if isinstance(item, str):
                cat_exact.add(item.lower().strip())
            elif isinstance(item, dict):
                if "name" in item:
                    cat_exact.add(item["name"].lower().strip())
                for alias in item.get("aliases", []):
                    cat_exact.add(alias.lower().strip())

    client_names = Counter()
    obj_names_by_attr = defaultdict(Counter)
    free_values_by_attr = defaultdict(Counter)
    canon_values_not_in_cat = defaultdict(Counter)
    date_diff_by_attr = defaultdict(Counter)

    with open(train_path, encoding="utf-8") as f_in:
        for line in f_in:
            if not line.strip():
                continue
            c = json.loads(line)
            started_day = c["started_at"][:10]
            for f in c.get("facts", []):
                attr = f["attribute"]
                val = f["value"]
                obj = f["object"]
                oname = obj["name"]
                otype = obj.get("type", "")

                if otype == "client":
                    client_names[oname] += 1
                obj_names_by_attr[attr][f"{otype}:{oname}"] += 1

                src = f.get("source") or {}
                has_quote = src.get("quote") is not None
                is_free = (f.get("valid_from") == c.get("started_at") and not has_quote)

                clean_val = val.replace(" — не помогло", "").strip().lower()
                if is_free:
                    free_values_by_attr[attr][val] += 1
                elif clean_val not in cat_exact:
                    canon_values_not_in_cat[attr][val] += 1

                vfrom = (f.get("valid_from") or "")[:10]
                vto = (f.get("valid_to") or "")[:10]
                date_diff_by_attr[attr][f"from==start:{vfrom == started_day}, has_to:{bool(vto)}"] += 1

    print("=== 1. ВСЕ ИМЕНА КЛИЕНТОВ В ЭТАЛОНЕ (train) ===")
    for name, cnt in client_names.most_common(20):
        print(f"  '{name}': {cnt}")

    print("\n=== 2. ТОП ОБЪЕКТОВ ПО АТРИБУТАМ ===")
    for attr, cnts in obj_names_by_attr.items():
        print(f"  [{attr}] топ-5: {cnts.most_common(5)}")

    print("\n=== 3. КАНОНИЧЕСКИЕ ЗНАЧЕНИЯ ВНЕ КАТАЛОГА (если есть) ===")
    for attr, cnts in canon_values_not_in_cat.items():
        print(f"  [{attr}] всего уникальных={len(cnts)}, топ-5: {cnts.most_common(5)}")

    print("\n=== 4. ПРИМЕРЫ СВОБОДНЫХ ФАКТОВ (LLM-judge layer) ===")
    for attr, cnts in free_values_by_attr.items():
        print(f"  [{attr}] всего={sum(cnts.values())}, уникальных={len(cnts)}, топ-8:")
        for v, cnt in cnts.most_common(8):
            print(f"      ({cnt}) {v}")

    print("\n=== 5. СТРУКТУРА ДАТ ПО АТРИБУТАМ ===")
    for attr, cnts in date_diff_by_attr.items():
        print(f"  [{attr}]: {dict(cnts)}")


if __name__ == "__main__":
    inspect_train_patterns(sys.argv[1], sys.argv[4])