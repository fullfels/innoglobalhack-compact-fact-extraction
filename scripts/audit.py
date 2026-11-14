#!/usr/bin/env python3
"""Быстрый аудит train/val (за ~1 сек) и разбор ошибок baseline."""
import json
import sys
from collections import Counter
from case_metrics import _value_sim, _tokens, match_facts, VALUE_SIM


def first_char(name: str) -> str:
    # Точная копия логики из case_metrics.py
    toks = _tokens(" ".join(name))
    return toks[0] if toks else ""


def audit_split(path: str, catalog_exact: set[str], catalog_token_sets: list[set[str]]):
    star_calls = 0
    multi_session_calls = 0
    broken_obj_calls = 0
    n_calls = 0

    attr_counts = Counter()
    obj_type_counts = Counter()
    canonical_vs_free = Counter()
    in_catalog_counts = Counter()
    polarity_counts = Counter()
    span_present = Counter()

    with open(path, encoding="utf-8") as f_in:
        for line in f_in:
            if not line.strip():
                continue
            n_calls += 1
            c = json.loads(line)
            turns = c.get("turns", [])

            if any(t["text"].startswith("**") for t in turns):
                star_calls += 1

            # Быстрая проверка на склеенные диалоги
            if len(turns) > 18:
                for t in turns[:-3]:
                    txt_low = t["text"].lower()
                    if "всего доброго" in txt_low or "до свидания" in txt_low or "хорошего дня" in txt_low:
                        multi_session_calls += 1
                        break

            started_at = c.get("started_at")
            for f in c.get("facts", []):
                attr = f["attribute"]
                val = f["value"]
                obj_name = f["object"]["name"]
                obj_type = f["object"].get("type", "unknown")

                attr_counts[attr] += 1
                obj_type_counts[obj_type] += 1
                polarity_counts[f.get("polarity", 1)] += 1

                if obj_name.startswith("u101_") or obj_name.startswith("u2027_"):
                    broken_obj_calls += 1

                src = f.get("source") or {}
                has_quote = src.get("quote") is not None
                has_span = src.get("span") is not None
                span_present[f"quote={has_quote}, span={has_span}"] += 1

                if f.get("valid_from") == started_at and not has_quote:
                    canonical_vs_free["free (LLM-judge layer)"] += 1
                else:
                    canonical_vs_free["canonical (generator layer)"] += 1

                # Быстрая проверка по каталогу за O(1) + пересечение множеств токенов
                clean_val = val.replace(" — не помогло", "").strip().lower()
                if clean_val in catalog_exact:
                    in_catalog_counts["exact_in_catalog"] += 1
                else:
                    v_toks = set(_tokens(clean_val))
                    if v_toks and any(len(v_toks & c_toks) / max(len(v_toks), len(c_toks)) >= 0.75 for c_toks in catalog_token_sets):
                        in_catalog_counts["token_overlap_in_catalog"] += 1
                    else:
                        in_catalog_counts["out_of_catalog (free text)"] += 1

    print(f"=== АУДИТ ФАЙЛА: {path} ===")
    print(f"Всего звонков: {n_calls} | Всего фактов: {sum(attr_counts.values())}")
    print(f"Звонков с инверсией спикеров ('**'): {star_calls}")
    print(f"Склеенных звонков (прощание в середине): {multi_session_calls}")
    print(f"Фактов с битым object.name (ID вместо имени): {broken_obj_calls}")
    print("Атрибуты:", dict(attr_counts))
    print("Типы объектов:", dict(obj_type_counts))
    print("Полярность:", dict(polarity_counts))
    print("Наличие quote/span:", dict(span_present))
    print("Слои разметки:", dict(canonical_vs_free))
    print("Покрытие каталогом:", dict(in_catalog_counts))
    print("-" * 60)


def analyze_baseline_errors(val_path: str, pred_path: str):
    gold_calls = {c["call_id"]: c for c in map(json.loads, open(val_path, encoding="utf-8"))}
    pred_calls = {c["call_id"]: c for c in map(json.loads, open(pred_path, encoding="utf-8"))}

    fn_reasons = Counter()
    missed_by_attr = Counter()
    fp_by_attr = Counter()
    first_letter_mismatches = []

    for cid, gcall in gold_calls.items():
        gfacts = gcall["facts"]
        pfacts = pred_calls.get(cid, {}).get("facts", [])
        matched = match_facts(gfacts, pfacts)
        matched_g_ids = {id(g) for g, p, _ in matched}
        matched_p_ids = {id(p) for g, p, _ in matched}

        for p in pfacts:
            if id(p) not in matched_p_ids:
                fp_by_attr[p.get("attribute")] += 1

        for g in gfacts:
            if id(g) in matched_g_ids:
                continue
            missed_by_attr[g["attribute"]] += 1

            same_attr = [p for p in pfacts if p.get("attribute") == g["attribute"] and id(p) not in matched_p_ids]
            if not same_attr:
                fn_reasons["1. Нет предсказаний с таким attribute"] += 1
                continue

            best_p = max(same_attr, key=lambda p: _value_sim(g["value"], p.get("value", "")))
            best_vsim = _value_sim(g["value"], best_p.get("value", ""))

            if best_vsim >= VALUE_SIM:
                g_char = first_char(g["object"]["name"])
                p_char = first_char(best_p.get("object", {}).get("name", ""))
                if g_char != p_char:
                    fn_reasons["2. Value совпало, но разная 1-я буква object.name!"] += 1
                    first_letter_mismatches.append((g["attribute"], g["object"]["name"], best_p.get("object", {}).get("name", ""), g["value"]))
                else:
                    fn_reasons["3. Value совпало, но разная polarity"] += 1
            else:
                fn_reasons["4. Низкий value_sim (< 0.85)"] += 1

    print("\n=== РАЗБОР ОШИБОК BASELINE НА VAL ===")
    print("Пропущенные факты (FN) по атрибутам:", dict(missed_by_attr))
    print("Лишние факты (FP) по атрибутам:", dict(fp_by_attr))
    print("\nПричины пропуска золотых фактов (FN):")
    for k, v in sorted(fn_reasons.items()):
        print(f"  {k}: {v}")

    if first_letter_mismatches:
        print("\nПримеры потерь из-за 1-й буквы object.name (Attr | Gold Obj vs Pred Obj | Value):")
        for attr, go, po, val in first_letter_mismatches[:12]:
            print(f"  [{attr}] Gold: '{go}' ({first_char(go)}) != Pred: '{po}' ({first_char(po)}) | val: '{val}'")


if __name__ == "__main__":
    train_f, val_f, pred_f, cat_f = sys.argv[1:5]
    cat_data = json.load(open(cat_f, encoding="utf-8"))
    cat_exact = set()
    for group in cat_data.values():
        for item in group:
            if isinstance(item, str):
                cat_exact.add(item.lower().strip())
            elif isinstance(item, dict):
                if "name" in item:
                    cat_exact.add(item["name"].lower().strip())
                for alias in item.get("aliases", []):
                    cat_exact.add(alias.lower().strip())
    cat_tok_sets = [set(_tokens(item)) for item in cat_exact if _tokens(item)]

    audit_split(train_f, cat_exact, cat_tok_sets)
    audit_split(val_f, cat_exact, cat_tok_sets)
    analyze_baseline_errors(val_f, pred_f)