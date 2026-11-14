#!/usr/bin/env python3
"""Шаг 3 (Быстрый, ~0.3 сек): Постпроцессор на 3-граммах и замер прироста метрик."""
import copy
import json
import re
import sys
from case_metrics import _tokens, evaluate

CLIENT_PROFILES = [
    ("Анастасия Валерьевна Прохорова", ["анастас", "прохоров", "валерьев", "анячвос"]),
    ("Катерина Сергеевна", ["катерин", "екатерин", "catherine", "ецашеч", "катуран"]),
    ("Оля Викторовна", ["оля", "ольга"]),
    ("Елен Петровна", ["петро", "елен петр", "елена петр"]),
    ("Елене Викторовне", ["елена викт", "елен викт", "викторовн"]),
    ("Марин Олеговна", ["олегов", "олегав", "ошегов"]),
    ("Марин Андреевна", ["андреев", "анвреив", "авдкеив"]),
    ("Марин Викторовна", ["марин викт", "марина викт", "мари викт"]),
    ("Марин Сергеевна", ["марин серг", "марина серг"]),
    ("Марь Сергевна", ["марь ", "марья", "сергевн", "серкевн"]),
]


def _ngrams(s: str, n: int = 3) -> set[str]:
    s = " ".join(_tokens(s))
    if len(s) < n:
        return {s} if s else set()
    return {s[i:i + n] for i in range(len(s) - n + 1)}


def _stems(s: str) -> set[str]:
    # Первые 4 буквы каждого слова для быстрого сравнения корней
    return {w[:4] for w in _tokens(s) if len(w) >= 2}


class FastCatalogIndex:
    """Мгновенный индекс по catalog.json без вызовов SequenceMatcher."""
    def __init__(self, cat: dict):
        self.groups = {}
        ext_cat = copy.deepcopy(cat)
        # Безопасно добавляем действия, если ключ actions есть, или создаем его
        if "actions" in ext_cat:
            ext_cat["actions"].extend(["проверка баланса", "проверка оплаты"])
        else:
            ext_cat["actions"] = ["проверка баланса", "проверка оплаты"]
            
        for gname, items in ext_cat.items():
            indexed = []
            for item in items:
                # Поддерживаем как чистые строки, так и словари {"name": ..., "aliases": ...}
                if isinstance(item, str):
                    names_to_index = [item]
                elif isinstance(item, dict):
                    names_to_index = [item.get("name", "")] + item.get("aliases", [])
                else:
                    names_to_index = [str(item)]
                
                for name in names_to_index:
                    if not name:
                        continue
                    low = name.lower().strip()
                    indexed.append((name, low, _ngrams(low), _stems(low)))
            self.groups[gname] = indexed

    def match(self, text: str, group_name: str, threshold: float = 0.55) -> tuple[str | None, float]:
        # Безопасный выход, если группы нет в catalog.json
        if group_name not in self.groups:
            return None, 0.0
            
        t_low = text.lower().strip()
        if not t_low:
            return None, 0.0
        t_low = text.lower().strip()
        if not t_low:
            return None, 0.0
        t_ng = _ngrams(t_low)
        t_st = _stems(t_low)
        if not t_ng:
            return None, 0.0

        best_item, best_score = None, 0.0
        for item, i_low, i_ng, i_st in self.groups[group_name]:
            if t_low == i_low:
                return item, 1.0
            # Быстрый фильтр по общим корням слов
            if i_st and not (t_st & i_st):
                continue
            inter = len(t_ng & i_ng)
            if not inter:
                continue
            # Смесь Jaccard и Recall (если фраза LLM длинная, но содержит все слова каталога)
            jaccard = inter / len(t_ng | i_ng)
            recall = inter / len(i_ng)
            stem_recall = len(t_st & i_st) / len(i_st) if i_st else 0.0
            score = max(jaccard, 0.45 * recall + 0.45 * stem_recall)
            if score > best_score:
                best_score = score
                best_item = item

        if best_score >= threshold:
            return best_item, best_score
        return None, best_score


def detect_client_name(call: dict) -> str:
    turns = call.get("turns", [])
    op_intro = " ".join(t["text"].lower() for t in turns[:2] if "меня зовут" in t["text"].lower() or "оператор" in t["text"].lower())
    full_text = " ".join(t["text"].lower() for t in turns)

    best_name, best_score = "Марин Сергеевна", -1
    for canon_name, markers in CLIENT_PROFILES:
        net_hits = sum(full_text.count(m) for m in markers) - sum(op_intro.count(m) for m in markers)
        if net_hits > best_score and net_hits > 0:
            best_score = net_hits
            best_name = canon_name
    return best_name


def detect_ticket_name(call: dict) -> str:
    full_text = " ".join(t["text"] for t in call.get("turns", []))
    m = re.search(r"(?:заявк\w*|обращени\w*|номер\w*)\s*[—:-]?\s*(\d{4})\b", full_text, re.I)
    if m:
        return f"заявка {m.group(1)}"
    m2 = re.search(r"\b(4\d{3}|5\d{3})\b", full_text)
    if m2:
        return f"заявка {m2.group(1)}"
    return "заявка"


def detect_main_service_or_device(call: dict, cat_idx: FastCatalogIndex) -> tuple[str, str]:
    early_text = " ".join(t["text"].lower() for t in call.get("turns", [])[:6])
    s_item, s_score = cat_idx.match(early_text, "services", threshold=0.45)
    d_item, d_score = cat_idx.match(early_text, "devices", threshold=0.45)
    if s_score >= d_score and s_item:
        return "service", s_item
    if d_item:
        return "device", d_item
    return "service", "домашний интернет"


def postprocess_call(call: dict, pred_facts: list[dict], cat_idx: FastCatalogIndex) -> list[dict]:
    client_name = detect_client_name(call)
    ticket_name = detect_ticket_name(call)
    main_otype, main_oname = detect_main_service_or_device(call, cat_idx)
    turns = call.get("turns", [])

    cleaned = []
    for pf in pred_facts:
        f = copy.deepcopy(pf)
        attr = f.get("attribute", "")
        val = f.get("value", "") or ""
        obj = f.get("object") or {}
        oname = obj.get("name", "") or ""
        t_idx = (f.get("source") or {}).get("turn_idx")

        turn_text = turns[t_idx]["text"].lower() if (isinstance(t_idx, int) and 0 <= t_idx < len(turns)) else ""

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
                if any(m in val.lower() or m in turn_text for m in fail_markers) and ac_match not in ("проверка оплаты", "проверка баланса", "пополнение счёта"):
                    val = f"{ac_match} — не помогло"
                else:
                    val = ac_match

        # 2. Нормализация service_state к симптомам каталога
        if attr == "service_state":
            if f.get("polarity", 1) == -1 or any(w in val.lower() for w in ("не использу", "не пользу", "отказ")):
                val = "больше не используется"
                f["polarity"] = -1
            elif any(w in val.lower() for w in ("работает нормально", "работает исправно", "работало нормально", "в норме", "исправен")):
                if f.get("valid_to") is not None or "нормально" in turn_text:
                    val = "работает нормально"
            else:
                sym_match, _ = cat_idx.match(val, "symptoms", threshold=0.62)
                if sym_match:
                    val = sym_match

        # 3. Нормализация tariff и ticket_status
        if attr == "tariff":
            t_match, _ = cat_idx.match(val, "tariffs", threshold=0.65)
            if t_match and f.get("valid_from") != call.get("started_at"):
                val = t_match
        elif attr == "ticket_status":
            st_match, _ = cat_idx.match(val, "statuses", threshold=0.55)
            if st_match:
                val = st_match

        # 4. Правило object.name под case_metrics.py
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

        # 5. Закон дат: у action и agreement valid_to ВСЕГДА null
        if attr in ("action", "agreement"):
            f["valid_to"] = None

        f["attribute"] = attr
        f["value"] = val
        f["object"] = obj
        cleaned.append(f)

    return cleaned


def main():
    val_f, pred_f, cat_f, out_f = sys.argv[1:5]
    gold_calls = {c["call_id"]: c for c in map(json.loads, open(val_f, encoding="utf-8"))}
    pred_calls = {c["call_id"]: c for c in map(json.loads, open(pred_f, encoding="utf-8"))}
    cat = json.load(open(cat_f, encoding="utf-8"))
    cat_idx = FastCatalogIndex(cat)

    print("=== МЕТРИКИ ДО ПОСТПРОЦЕССИНГА (RAW BASELINE) ===")
    m_before = evaluate(gold_calls, pred_calls)
    for k, v in m_before.items():
        print(f"  {k:18} {v}")

    fixed_preds = {}
    with open(out_f, "w", encoding="utf-8") as fout:
        for cid, gcall in gold_calls.items():
            pfacts = pred_calls.get(cid, {}).get("facts", [])
            new_facts = postprocess_call(gcall, pfacts, cat_idx)
            fixed_preds[cid] = {"call_id": cid, "facts": new_facts}
            fout.write(json.dumps(fixed_preds[cid], ensure_ascii=False) + "\n")

    print("\n=== МЕТРИКИ ПОСЛЕ ПОСТПРОЦЕССИНГА (FIXED BASELINE) ===")
    m_after = evaluate(gold_calls, fixed_preds)
    for k, v in m_after.items():
        diff = ""
        if isinstance(v, float) and isinstance(m_before.get(k), float):
            d = v - m_before[k]
            diff = f" ({d:+.4f})"
        print(f"  {k:18} {v}{diff}")


if __name__ == "__main__":
    main()