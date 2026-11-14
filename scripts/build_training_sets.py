#!/usr/bin/env python3
"""Шаг 6: Сборка компактного SFT-датасета для малых моделей (0.2B-1.5B) и генерация Adversarial ASR-шума."""
import copy
import json
import random
import re
import sys
from datetime import datetime, timedelta
from postprocess_test import FastCatalogIndex, detect_client_name, detect_ticket_name
from full_postprocess import find_best_span_for_fact

# Таблица типичных фонетических и клавиатурных замен ASR из корпуса
ASR_CHAR_MAP = {
    "о": ["а", "у", "э"],
    "а": ["о", "ы", "я"],
    "е": ["и", "э", "ю"],
    "и": ["ы", "е", "э"],
    "в": ["ф", "б"],
    "б": ["п", "в"],
    "д": ["т", "ж"],
    "т": ["д", "ц", "щ"],
    "з": ["с", "ж", "ц"],
    "с": ["з", "ш", "щ"],
    "к": ["г", "х"],
    "р": ["л", "ж"],
    "н": ["м", "л"],
}

FILLERS = ["эээ", "ну", "это самое", "короче", "типа", "в общем", "вот"]


def inject_adversarial_asr_noise(text: str, rng: random.Random, word_corrupt_prob: float = 0.28) -> str:
    """Имитирует усиленный ASR-шум (искажение букв, пропуски, дизфлюенции, обрыв реплики)."""
    words = text.split()
    if not words:
        return text

    out = []
    for i, w in enumerate(words):
        # Иногда вставляем слово-паразит
        if i > 1 and rng.random() < 0.04:
            out.append(rng.choice(FILLERS))

        # Искажаем слово с заданной вероятностью (не трогаем числа и короткие предлоги)
        clean_w = re.sub(r"[^а-яёА-ЯЁ]", "", w)
        if len(clean_w) >= 4 and rng.random() < word_corrupt_prob:
            chars = list(w)
            # Выбираем 1 или 2 позиции внутри слова для фонетической замены или пропуска
            n_edits = 1 if len(chars) < 7 else rng.choice([1, 2])
            for _ in range(n_edits):
                pos = rng.randint(1, len(chars) - 2)
                ch_low = chars[pos].lower()
                if ch_low in ASR_CHAR_MAP:
                    rep = rng.choice(ASR_CHAR_MAP[ch_low])
                    chars[pos] = rep.upper() if chars[pos].isupper() else rep
                elif rng.random() < 0.25:
                    # Пропуск буквы
                    chars[pos] = ""
            w = "".join(chars)
        out.append(w)

    # С вероятностью 8% имитируем обрыв длинной реплики тире
    if len(out) > 10 and rng.random() < 0.08:
        cut = rng.randint(len(out) - 3, len(out) - 1)
        out = out[:cut] + ["—"]

    return " ".join(out)


def date_to_offset_tag(iso_str: str | None, started_at: str) -> str:
    """Кодирует дату как компактный сдвиг в днях относительно started_at (напр. '0', '-5', '+2@18')."""
    if not iso_str:
        return "null"
    if iso_str == started_at:
        return "START"
    try:
        dt_start = datetime.fromisoformat(started_at)
        dt_val = datetime.fromisoformat(iso_str)
        day_diff = (dt_val.date() - dt_start.date()).days
        # Если дата больше чем на 45 дней в прошлом — это фоновое прошлое состояние ('PAST')
        if day_diff < -45:
            return "PAST"
        hour_suffix = f"@{dt_val.hour:02d}" if dt_val.hour != 0 and iso_str != started_at else ""
        return f"{day_diff:+d}{hour_suffix}" if day_diff != 0 else f"0{hour_suffix}"
    except ValueError:
        return "START"


def offset_tag_to_date(tag: str, started_at: str) -> str | None:
    """Декодирует компактный сдвиг обратно в полную ISO-дату."""
    tag = tag.strip()
    if not tag or tag == "null" or tag == "None":
        return None
    if tag == "START":
        return started_at
    dt_start = datetime.fromisoformat(started_at)
    if tag == "PAST":
        # Любая дата глубоко в прошлом для фоновых состояний (закрытых valid_to)
        return (dt_start - timedelta(days=180)).strftime(f"%Y-%m-%dT{dt_start.hour:02d}:00:00")

    m = re.match(r"^([+-]?\d+)(?:@(\d{1,2}))?$", tag)
    if m:
        days = int(m.group(1))
        hour = int(m.group(2)) if m.group(2) is not None else 0
        target = dt_start.date() + timedelta(days=days)
        return f"{target.isoformat()}T{hour:02d}:00:00"
    return started_at


def format_call_input(call: dict) -> str:
    """Формирует компактный входной текст звонка для модели."""
    lines = [f"call_id: {call['call_id']} | started_at: {call['started_at']}"]
    for t in call.get("turns", []):
        sp = "OP" if t["speaker"] == "operator" else "CL"
        lines.append(f"[{t['idx']}] {sp}: {t['text']}")
    return "\n".join(lines)


def format_call_target(call: dict) -> str:
    """Кодирует эталонные факты в сверхкомпактный построчный формат (в 4 раза меньше токенов, чем JSON)."""
    started_at = call["started_at"]
    lines = []
    for f in call.get("facts", []):
        t_idx = (f.get("source") or {}).get("turn_idx", 0)
        attr = f["attribute"]
        otype = f["object"].get("type", "client")
        oname = f["object"].get("name", "").replace("|", " ")
        val = f["value"].replace("|", " ")
        pol = f.get("polarity", 1)
        vfrom_tag = date_to_offset_tag(f.get("valid_from"), started_at)
        vto_tag = date_to_offset_tag(f.get("valid_to"), started_at)
        lines.append(f"{t_idx} | {attr} | {otype}:{oname} | {val} | {pol} | {vfrom_tag} | {vto_tag}")
    return "\n".join(lines) if lines else "NONE"


def decode_compact_output(raw_output: str, call: dict, cat_idx: FastCatalogIndex) -> list[dict]:
    """Декодирует компактный вывод модели обратно в полный список фактов по схеме кейса."""
    if not raw_output or raw_output.strip() == "NONE":
        return []
    started_at = call["started_at"]
    turns = call.get("turns", [])
    client_name = detect_client_name(call)
    ticket_name = detect_ticket_name(call)

    facts = []
    for line in raw_output.strip().splitlines():
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 7:
            continue
        try:
            t_idx = int(re.sub(r"\D", "", parts[0]) or 0)
        except ValueError:
            t_idx = 0
        attr = parts[1]
        if attr not in ("service_state", "tariff", "ticket_status", "action", "agreement"):
            continue

        obj_raw = parts[2]
        if ":" in obj_raw:
            otype, oname = obj_raw.split(":", 1)
        else:
            otype, oname = "client", obj_raw

        val = parts[3]
        try:
            pol = -1 if "-1" in parts[4] else 1
        except ValueError:
            pol = 1

        vfrom = offset_tag_to_date(parts[5], started_at)
        vto = offset_tag_to_date(parts[6], started_at)
        if attr in ("action", "agreement"):
            vto = None

        # Гарантируем валидное имя объекта
        if attr in ("action", "agreement", "tariff"):
            otype, oname = "client", client_name
        elif attr == "ticket_status" and not oname.lower().startswith("заявка"):
            otype, oname = "ticket", ticket_name

        turn_text = turns[t_idx]["text"] if 0 <= t_idx < len(turns) else ""
        span = find_best_span_for_fact(turn_text, val, oname) if turn_text else None

        facts.append({
            "object": {"type": otype, "name": oname},
            "attribute": attr,
            "value": val,
            "polarity": pol,
            "valid_from": vfrom,
            "valid_to": vto,
            "source": {"turn_idx": t_idx, "span": span},
        })
    return facts


SYSTEM_PROMPT = (
    "Извлеки все факты из транскрипта звонка в техподдержку. "
    "Формат каждой строки: turn_idx | attribute | obj_type:obj_name | value | polarity | valid_from_offset | valid_to_offset"
)


def main():
    train_clean_f, val_f, sft_train_f, sft_val_f, val_adv_f = sys.argv[1:6]
    rng = random.Random(42)

    n_train = 0
    n_aug = 0
    with open(train_clean_f, encoding="utf-8") as fin, open(sft_train_f, "w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            c = json.loads(line)
            target_str = format_call_target(c)

            # Оригинальный пример
            rec = {
                "call_id": c["call_id"],
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": format_call_input(c)},
                    {"role": "assistant", "content": target_str},
                ],
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_train += 1

            # 25% примеров дублируем с усиленным ASR-шумом (Adversarial Augmentation)
            if rng.random() < 0.25:
                c_adv = copy.deepcopy(c)
                for t in c_adv.get("turns", []):
                    t["text"] = inject_adversarial_asr_noise(t["text"], rng, word_corrupt_prob=0.30)
                rec_adv = {
                    "call_id": c["call_id"] + ":adv",
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": format_call_input(c_adv)},
                        {"role": "assistant", "content": target_str},
                    ],
                }
                fout.write(json.dumps(rec_adv, ensure_ascii=False) + "\n")
                n_aug += 1

    n_val = 0
    with open(val_f, encoding="utf-8") as fin, \
         open(sft_val_f, "w", encoding="utf-8") as f_sft_val, \
         open(val_adv_f, "w", encoding="utf-8") as f_adv:
        for line in fin:
            if not line.strip():
                continue
            c = json.loads(line)
            rec = {
                "call_id": c["call_id"],
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": format_call_input(c)},
                    {"role": "assistant", "content": format_call_target(c)},
                ],
            }
            f_sft_val.write(json.dumps(rec, ensure_ascii=False) + "\n")

            # Создаём adversarial-версию val.jsonl (noise_level = 0.45) для тестирования устойчивости
            c_adv = copy.deepcopy(c)
            c_adv["noise_level"] = 0.45
            for t in c_adv.get("turns", []):
                t["text"] = inject_adversarial_asr_noise(t["text"], rng, word_corrupt_prob=0.32)
            f_adv.write(json.dumps(c_adv, ensure_ascii=False) + "\n")
            n_val += 1

    print("=== ШАГ 6 ЗАВЕРШЁН ===")
    print(f"Собрано в {sft_train_f}: {n_train} базовых + {n_aug} adversarial = {n_train + n_aug} диалогов")
    print(f"Собрано в {sft_val_f}: {n_val} валидационных диалогов")
    print(f"Создан стресс-тест {val_adv_f}: {n_val} диалогов с noise_level=0.45")


if __name__ == "__main__":
    main()