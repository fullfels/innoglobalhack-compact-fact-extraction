#!/usr/bin/env python3
"""Шаг 4: Очистка train.jsonl, восстановление span в ASR-тексте и детерминированный парсер дат."""
import copy
import json
import re
import sys
from collections import Counter
from datetime import datetime, timedelta
from case_metrics import _tokens
from postprocess_test import (
    FastCatalogIndex,
    detect_client_name,
    detect_ticket_name,
    detect_main_service_or_device,
)

# Словарь русских числительных (включая частые ASR-искажения из корпуса)
NUM_WORDS = {
    "один": 1, "одна": 1, "одну": 1, "перв": 1,
    "два": 2, "две": 2, "двух": 2, " тва": 2, "твух": 2,
    "три": 3, "трех": 3, "трёх": 3, "тритий": 3,
    "четыре": 4, "четырех": 4, "четырёх": 4, "читыре": 4, "етыре": 4,
    "пять": 5, "пяти": 5,
    "шесть": 6, "шести": 6, "шисть": 6,
    "семь": 7, "семи": 7,
    "восемь": 8, "восьми": 8,
    "девять": 9, "десять": 10,
}

# Дни недели (0 = понедельник ... 6 = воскресенье) + ASR-варианты из корпуса
WEEKDAYS = {
    0: ["понедельник", "панедельник", "путегельник"],
    1: ["вторник", "втарник", "фторник", "втерчюк", "хярник", "сярчок"],
    2: ["сред", "сриду", "сриды", "среты"],
    3: ["четверг", "читверг", "четверк", "четфёрт", "чеввярг", "чютвырг", "гефвярг", "метвирг", "чыхвюрг", "летверг", "щервефг"],
    4: ["пятниц", "футниц", "хижниц"],
    5: ["суббот", "суббат"],
    6: ["воскресен", "хоскресен"],
}


def parse_relative_date(text: str, started_at_iso: str, is_future_attr: bool = False) -> str | None:
    """Вычисляет ISO-дату (YYYY-MM-DDTHH:MM:SS) из текста реплики относительно started_at."""
    dt = datetime.fromisoformat(started_at_iso)
    t_low = text.lower().replace("ё", "е")

    # Определяем время суток, если оно явно озвучено
    hour = 0
    if re.search(r"после\s+(?:шест|шист|6)", t_low):
        hour = 18
    elif re.search(r"после\s+(?:тр[её]х|3)", t_low):
        hour = 15
    elif re.search(r"после\s+(?:двух|твух|2)", t_low):
        hour = 14
    elif any(w in t_low for w in ("к обеду", "к абеду", "до обеда")):
        hour = 12
    elif any(w in t_low for w in ("с утра", "утром", "утрам")):
        hour = 9

    target_date = None

    # 1. Точные сдвиги в днях: вчера / позавчера / завтра / послезавтра
    if any(w in t_low for w in ("послезавтра", "пыслезавбр", "паслезавтр", "послизавтр")):
        target_date = dt.date() + timedelta(days=2)
    elif any(w in t_low for w in ("завтра", "зафтра", "зявтра", "зюбтда", "хонжра", "зустра")) and "позавчера" not in t_low:
        target_date = dt.date() + timedelta(days=1)
    elif "позавчера" in t_low:
        target_date = dt.date() - timedelta(days=2)
    elif any(w in t_low for w in ("вчера", "вчира", "фчера", "бчюра", "жчыра", "щчыра")):
        target_date = dt.date() - timedelta(days=1)
    elif any(w in t_low for w in ("сегодня", "сигодня", "сеготня", "сегадня", "ныгодня", "сещэцня")):
        target_date = dt.date()

    # 2. "через N дней" / "N дней назад" / "через неделю"
    if target_date is None:
        if "через недел" in t_low or "чирез нетел" in t_low:
            target_date = dt.date() + timedelta(days=7)
        elif "неделю назад" in t_low or "нерелю назад" in t_low:
            target_date = dt.date() - timedelta(days=7)
        else:
            for word, num in NUM_WORDS.items():
                if re.search(rf"(?:через|чирез|чере|чарез)\s+{word}\s+(?:дн|тн)", t_low):
                    target_date = dt.date() + timedelta(days=num)
                    break
                if re.search(rf"{word}\s+(?:дн|тн)\w*\s+(?:назад|назат|нозуд)", t_low):
                    target_date = dt.date() - timedelta(days=num)
                    break

    # 3. Дни недели ("со вторника", "в четверг", "до среды")
    if target_date is None:
        for wd_idx, stems in WEEKDAYS.items():
            if any(st in t_low for st in stems):
                if is_future_attr or any(p in t_low for p in ("в четверг", "в пятниц", "в суббот", "в понедельник", "в воскресень", "во вторник", "в среду", "назнач", "приедет", "придёт", "перезвон")):
                    # Ближайший такой день недели в будущем (1..7 дней вперёд)
                    days_ahead = (wd_idx - dt.weekday()) % 7
                    if days_ahead == 0:
                        days_ahead = 7
                    # Но если есть предлог "со вторника / с четверга / с пятницы / с понедельника" — это прошлое!
                    if re.search(r"\b(?:со|с|до)\s+\w*(?:" + "|".join(stems) + ")", t_low):
                        days_back = (dt.weekday() - wd_idx) % 7
                        if days_back == 0:
                            days_back = 7
                        target_date = dt.date() - timedelta(days=days_back)
                    else:
                        target_date = dt.date() + timedelta(days=days_ahead)
                else:
                    # Прошедший день недели
                    days_back = (dt.weekday() - wd_idx) % 7
                    if days_back == 0:
                        days_back = 7
                    target_date = dt.date() - timedelta(days=days_back)
                break

    if target_date is None:
        return None
    return f"{target_date.isoformat()}T{hour:02d}:00:00"


def find_fuzzy_span(turn_text: str, query_text: str) -> list[int] | None:
    """Быстрый поиск символьного спана [start, end] внутри зашумлённой реплики по совпадению корней слов."""
    if not turn_text or not query_text:
        return None
    t_low = turn_text.lower().replace("ё", "е")
    q_low = query_text.lower().replace("ё", "е").replace(" — не помогло", "")

    # 1. Точное вхождение
    idx = t_low.find(q_low)
    if idx != -1:
        return [idx, idx + len(q_low)]

    # 2. Поиск по токенам с их символьными оффсетами
    word_matches = list(re.finditer(r"[а-яёa-z0-9]+", t_low))
    if not word_matches:
        return None
    q_words = [w for w in re.findall(r"[а-яёa-z0-9]+", q_low) if len(w) >= 3]
    if not q_words:
        return None

    q_prefixes = {w[:4] for w in q_words}
    hit_indices = []
    for i, m in enumerate(word_matches):
        w = m.group(0)
        if len(w) >= 3 and w[:4] in q_prefixes:
            hit_indices.append(i)

    if not hit_indices:
        return None

    # Берём минимальное окно, покрывающее найденные опорные слова
    first_m = word_matches[hit_indices[0]]
    last_m = word_matches[hit_indices[-1]]
    span_len = last_m.end() - first_m.start()
    if span_len <= max(len(q_low) * 2.5, 60):
        return [first_m.start(), last_m.end()]
    return [first_m.start(), min(len(turn_text), first_m.start() + len(q_low))]


def clean_and_enrich_call(call: dict, cat_idx: FastCatalogIndex) -> tuple[dict, dict]:
    c = copy.deepcopy(call)
    turns = c.get("turns", [])
    stats = Counter()

    # 1. Очистка от '**' и детекция инверсии спикеров
    has_stars = any(t["text"].startswith("**") for t in turns)
    if has_stars:
        stats["star_calls_cleaned"] += 1
        for t in turns:
            t["text"] = re.sub(r"^\*\*\s*", "", t["text"]).strip()

    client_name = detect_client_name(c)
    ticket_name = detect_ticket_name(c)
    main_otype, main_oname = detect_main_service_or_device(c, cat_idx)

    for f in c.get("facts", []):
        obj = f["object"]
        oname = obj["name"]
        attr = f["attribute"]

        # 2. Лечение битых ID в object.name ('u101_00020:ticket0' и т.д.)
        if oname.startswith("u101_") or oname.startswith("u2027_"):
            stats["broken_obj_fixed"] += 1
            if ":client" in oname or attr in ("action", "agreement", "tariff"):
                obj["type"] = "client"
                obj["name"] = client_name
            elif ":ticket" in oname or attr == "ticket_status":
                obj["type"] = "ticket"
                obj["name"] = ticket_name
            else:
                obj["type"] = main_otype
                obj["name"] = main_oname

        # 3. Разметка слоя (canonical vs free) и восстановление пропущенных span
        src = f.get("source") or {}
        has_quote = src.get("quote") is not None
        if f.get("valid_from") == c.get("started_at") and not has_quote:
            f["layer"] = "free"
        else:
            f["layer"] = "canonical"

        t_idx = src.get("turn_idx")
        if src.get("span") is None and isinstance(t_idx, int) and 0 <= t_idx < len(turns):
            query = src.get("quote") or f.get("value", "")
            span = find_fuzzy_span(turns[t_idx]["text"], query)
            if span is not None:
                src["span"] = span
                if src.get("quote") is None:
                    src["quote"] = turns[t_idx]["text"][span[0]:span[1]]
                stats["spans_recovered"] += 1

    return c, stats


def test_date_parser_on_split(path: str):
    """Проверяет, насколько точно наш парсер дат угадывает valid_from по тексту реплики."""
    total_rel = 0
    hits = 0
    with open(path, encoding="utf-8") as fin:
        for line in fin:
            if not line.strip():
                continue
            c = json.loads(line)
            started_at = c["started_at"]
            started_day = started_at[:10]
            turns = c.get("turns", [])
            for f in c.get("facts", []):
                t_idx = (f.get("source") or {}).get("turn_idx")
                if not isinstance(t_idx, int) or not (0 <= t_idx < len(turns)):
                    continue
                gold_day = (f.get("valid_from") or "")[:10]
                if not gold_day or gold_day == started_day:
                    continue
                # Проверяем на фактах, где дата отличается от дня звонка
                pred_iso = parse_relative_date(
                    turns[t_idx]["text"],
                    started_at,
                    is_future_attr=(f["attribute"] == "agreement"),
                )
                if pred_iso is not None:
                    total_rel += 1
                    if pred_iso[:10] == gold_day:
                        hits += 1
    acc = hits / total_rel if total_rel else 0.0
    print(f"Точность парсера относительных дат (на репликах с явной датой): {hits}/{total_rel} ({acc:.1%})")


def main():
    train_f, val_f, cat_f, out_train_f = sys.argv[1:5]
    cat = json.load(open(cat_f, encoding="utf-8"))
    cat_idx = FastCatalogIndex(cat)

    total_stats = Counter()
    with open(train_f, encoding="utf-8") as fin, open(out_train_f, "w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            c = json.loads(line)
            clean_c, st = clean_and_enrich_call(c, cat_idx)
            total_stats.update(st)
            fout.write(json.dumps(clean_c, ensure_ascii=False) + "\n")

    print("=== РЕЗУЛЬТАТ ОЧИСТКИ TRAIN.JSONL -> TRAIN_CLEAN.JSONL ===")
    for k, v in total_stats.items():
        print(f"  {k}: {v}")

    print("\n=== ТЕСТ ДЕТЕРМИНИРОВАННОГО ПАРСЕРА ДАТ НА VAL ===")
    test_date_parser_on_split(val_f)


if __name__ == "__main__":
    main()