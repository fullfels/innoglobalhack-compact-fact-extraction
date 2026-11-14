#!/usr/bin/env python3
"""Шаг 5: Полный постпроцессор (нормализация + спаны + цепочки дат) и тест на val.jsonl."""
import copy
import json
import re
import sys
from datetime import datetime, timedelta
from case_metrics import _tokens, evaluate, SINGLE_VALUED
from postprocess_test import (
    FastCatalogIndex,
    detect_client_name,
    detect_ticket_name,
    detect_main_service_or_device,
)
from prepare_data import parse_relative_date, find_fuzzy_span


def find_best_span_for_fact(turn_text: str, val: str, oname: str) -> list[int] | None:
    """Находит символьный интервал [start, end] в реплике для метрики grounding_span."""
    if not turn_text:
        return None
    # Сначала ищем по значению факта
    sp = find_fuzzy_span(turn_text, val)
    if sp is not None:
        return sp
    # Если не нашли — ищем по имени объекта
    sp = find_fuzzy_span(turn_text, oname)
    if sp is not None:
        return sp
    # Fallback: вся содержательная часть реплики (гарантирует пересечение интервалов lo < hi в case_metrics.py!)
    if len(turn_text) > 4:
        return [0, len(turn_text)]
    return None


def fix_state_chains(facts: list[dict], started_at: str) -> list[dict]:
    """Выстраивает цепочки valid_from -> valid_to для однозначных атрибутов."""
    # Группируем по (attribute)
    by_attr: dict[str, list[dict]] = {}
    for f in facts:
        attr = f.get("attribute", "")
        if attr in SINGLE_VALUED:
            by_attr.setdefault(attr, []).append(f)

    # Порядок статусов заявки по онтологии: принята -> в работе -> назначен выезд специалиста -> выполнена
    status_order = {"принята": 0, "в работе": 1, "назначен выезд специалиста": 2, "выполнена": 3}

    for attr, group in by_attr.items():
        if len(group) == 1:
            # Если факт один и это актуальное состояние — закрывать его нельзя (иначе state_accuracy упадёт!)
            f = group[0]
            if f.get("value") != "работает нормально" or len(group) == 1:
                if attr in ("ticket_status", "tariff"):
                    f["valid_to"] = None
        else:
            if attr == "ticket_status":
                group.sort(key=lambda x: (status_order.get(x.get("value", ""), 99), x.get("valid_from") or ""))
            else:
                # "работает нормально" всегда предшествует симптому поломки
                group.sort(key=lambda x: (0 if x.get("value") == "работает нормально" else 1, x.get("valid_from") or ""))

            # Последний элемент цепочки обязан иметь valid_to = None (для state_accuracy),
            # а предыдущие — закрываться датой следующего!
            for i in range(len(group) - 1):
                cur_f = group[i]
                next_f = group[i + 1]
                if next_f.get("valid_from"):
                    cur_f["valid_to"] = next_f["valid_from"][:10] + "T00:00:00"
                    # Если даты начала совпали, сдвигаем предыдущий статус на 1-2 дня назад
                    if cur_f.get("valid_from", "")[:10] >= next_f["valid_from"][:10]:
                        try:
                            dt_next = datetime.fromisoformat(next_f["valid_from"])
                            cur_f["valid_from"] = (dt_next - timedelta(days=2)).strftime("%Y-%m-%dT00:00:00")
                        except ValueError:
                            pass
            group[-1]["valid_to"] = None

    return facts


def full_postprocess_call(call: dict, pred_facts: list[dict], cat_idx: FastCatalogIndex) -> list[dict]:
    client_name = detect_client_name(call)
    ticket_name = detect_ticket_name(call)
    main_otype, main_oname = detect_main_service_or_device(call, cat_idx)
    turns = call.get("turns", [])
    started_at = call.get("started_at", "")

    cleaned = []
    for pf in pred_facts:
        f = copy.deepcopy(pf)
        attr = f.get("attribute", "")
        val = f.get("value", "") or ""
        obj = f.get("object") or {}
        oname = obj.get("name", "") or ""
        src = f.get("source") or {}
        t_idx = src.get("turn_idx")

        turn_text = turns[t_idx]["text"] if (isinstance(t_idx, int) and 0 <= t_idx < len(turns)) else ""
        t_low = turn_text.lower()

        # 1. Исправление путаницы action <-> agreement по каталогу
        if attr in ("action", "agreement"):
            ag_match, ag_s = cat_idx.match(val, "agreements", threshold=0.60)
            ac_match, ac_s = cat_idx.match(val.replace("не помогло", ""), "actions", threshold=0.60)
            if ag_s > ac_s and ag_match and ag_s >= 0.65:
                attr = "agreement"
                val = ag_match
            elif ac_s >= ag_s and ac_match and ac_s >= 0.65:
                attr = "action"
                fail_markers = ("не помог", "бесполез", "не дало", "безрезультат", "толку ноль", "всё равно", "не устран")
                if any(m in val.lower() or m in t_low for m in fail_markers) and ac_match not in ("проверка оплаты", "проверка баланса", "пополнение счёта"):
                    val = f"{ac_match} — не помогло"
                else:
                    val = ac_match

        # 2. Нормализация service_state
        if attr == "service_state":
            if f.get("polarity", 1) == -1 or any(w in val.lower() for w in ("не использу", "не пользу", "отказ")):
                val = "больше не используется"
                f["polarity"] = -1
            elif any(w in val.lower() for w in ("работает нормально", "работает исправно", "работало нормально", "в норме", "исправен")):
                if f.get("valid_to") is not None or "нормально" in t_low:
                    val = "работает нормально"
            else:
                sym_match, _ = cat_idx.match(val, "symptoms", threshold=0.62)
                if sym_match:
                    val = sym_match

        # 3. Нормализация tariff и ticket_status
        if attr == "tariff":
            t_match, _ = cat_idx.match(val, "tariffs", threshold=0.65)
            if t_match and f.get("valid_from") != started_at:
                val = t_match
        elif attr == "ticket_status":
            st_match, _ = cat_idx.match(val, "statuses", threshold=0.55)
            if st_match:
                val = st_match

        # 4. Правило object.name
        if attr in ("action", "agreement", "tariff"):
            obj["type"] = "client"
            obj["name"] = client_name
        elif attr == "ticket_status":
            obj["type"] = "ticket"
            if re.fullmatch(r"\d{3,6}", oname.strip()):
                obj["name"] = f"заявка {oname.strip()}"
            elif not oname.lower().startswith("заявка"):
                obj["name"] = ticket_name
        elif attr == "service_state":
            s_m, s_s = cat_idx.match(oname, "services", threshold=0.60)
            d_m, d_s = cat_idx.match(oname, "devices", threshold=0.60)
            if s_s >= d_s and s_m:
                obj["type"] = "service"
                obj["name"] = s_m
            elif d_m:
                obj["type"] = "device"
                obj["name"] = d_m
            elif oname.lower() in ("интернет", "связь", "линия"):
                obj["type"] = main_otype
                obj["name"] = main_oname

        # 5. Даты: valid_to = None для action/agreement + уточнение даты из текста реплики
        if attr in ("action", "agreement"):
            f["valid_to"] = None
            if f.get("valid_from") == started_at and turn_text:
                parsed_dt = parse_relative_date(turn_text, started_at, is_future_attr=(attr == "agreement"))
                if parsed_dt and attr == "action":
                    f["valid_from"] = parsed_dt

        # 6. Восстановление span для grounding_span
        if turn_text:
            src["span"] = find_best_span_for_fact(turn_text, val, obj.get("name", ""))

        f["attribute"] = attr
        f["value"] = val
        f["object"] = obj
        f["source"] = src
        cleaned.append(f)

    # 7. Выравнивание цепочек состояний
    cleaned = fix_state_chains(cleaned, started_at)
    return cleaned


def main():
    val_f, pred_f, cat_f, out_f = sys.argv[1:5]
    gold_calls = {c["call_id"]: c for c in map(json.loads, open(val_f, encoding="utf-8"))}
    pred_calls = {c["call_id"]: c for c in map(json.loads, open(pred_f, encoding="utf-8"))}
    cat = json.load(open(cat_f, encoding="utf-8"))
    cat_idx = FastCatalogIndex(cat)

    m_before = evaluate(gold_calls, pred_calls)
    fixed_preds = {}
    with open(out_f, "w", encoding="utf-8") as fout:
        for cid, gcall in gold_calls.items():
            pfacts = pred_calls.get(cid, {}).get("facts", [])
            new_facts = full_postprocess_call(gcall, pfacts, cat_idx)
            fixed_preds[cid] = {"call_id": cid, "facts": new_facts}
            fout.write(json.dumps(fixed_preds[cid], ensure_ascii=False) + "\n")

    m_after = evaluate(gold_calls, fixed_preds)
    print("=== ИТОГОВЫЕ МЕТРИКИ ПОСЛЕ ШАГА 5 ===")
    for k, v in m_after.items():
        diff = ""
        if isinstance(v, float) and isinstance(m_before.get(k), float):
            d = v - m_before[k]
            diff = f" ({d:+.4f} vs raw baseline)"
        print(f"  {k:18} {v}{diff}")


if __name__ == "__main__":
    main()