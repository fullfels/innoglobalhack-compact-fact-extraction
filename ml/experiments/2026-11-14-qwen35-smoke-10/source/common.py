"""No ML dependencies: input isolation, JSON checks, and safe local persistence."""
from __future__ import annotations
import hashlib
import json
import os
import re
from datetime import datetime
from pathlib import Path

ATTRIBUTES = {"service_state", "tariff", "ticket_status", "action", "agreement"}
OBJECT_TYPES = {"service", "device", "ticket", "client"}

def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp, path)

def atomic_jsonl(path, rows):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temp, path)

def model_input(call):
    # Explicit allowlist: never serialise the original gold-bearing record.
    return {
        "call_id": call["call_id"],
        "started_at": call["started_at"],
        "turns": [{"idx": t["idx"], "speaker": t["speaker"], "text": t["text"]} for t in call["turns"]],
    }

def make_messages(call, instruction, catalog):
    return [
        {"role": "system", "content": instruction + "\n\nКаталог:\n" + json.dumps(catalog, ensure_ascii=False, separators=(",", ":"))},
        {"role": "user", "content": json.dumps(model_input(call), ensure_ascii=False)},
    ]

def unwrap_json(raw):
    """Remove only a complete outer Markdown fence, never rewrite fact contents."""
    text = raw.strip()
    match = re.fullmatch(r"```(?:json)?[ \t]*\r?\n(.*?)\r?\n```", text, flags=re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else text

def decode_and_validate(raw, call):
    value = json.loads(unwrap_json(raw))
    if not isinstance(value, dict) or set(value) != {"facts"} or not isinstance(value["facts"], list):
        raise ValueError('Ожидался объект с единственным полем "facts": список')
    turns = {t["idx"]: t for t in call["turns"]}
    for i, fact in enumerate(value["facts"]):
        if not isinstance(fact, dict):
            raise ValueError(f"facts[{i}]: не объект")
        required = {"object", "attribute", "value", "polarity", "speaker", "valid_from", "valid_to", "source"}
        if not required <= fact.keys():
            raise ValueError(f"facts[{i}]: отсутствуют {required - fact.keys()}")
        obj = fact["object"]
        if not isinstance(obj, dict) or not isinstance(obj.get("name"), str) or not obj["name"].strip():
            raise ValueError(f"facts[{i}]: некорректный object.name")
        if obj.get("type") not in OBJECT_TYPES:
            raise ValueError(f"facts[{i}]: некорректный object.type")
        if fact["attribute"] not in ATTRIBUTES or not isinstance(fact["value"], str) or not fact["value"].strip():
            raise ValueError(f"facts[{i}]: некорректный attribute/value")
        if type(fact["polarity"]) is not int or fact["polarity"] not in {-1, 1}:
            raise ValueError(f"facts[{i}]: polarity должен быть 1 или -1")
        for name in ("valid_from", "valid_to"):
            date = fact[name]
            if date is not None:
                if not isinstance(date, str) or len(date) != 19 or date[10] != "T":
                    raise ValueError(f"facts[{i}]: {name} не ISO datetime")
                datetime.fromisoformat(date)
        if fact["valid_from"] and fact["valid_to"] and fact["valid_from"] > fact["valid_to"]:
            raise ValueError(f"facts[{i}]: перепутан период")
        source = fact["source"]
        if not isinstance(source, dict) or type(source.get("turn_idx")) is not int or source["turn_idx"] not in turns:
            raise ValueError(f"facts[{i}]: отсутствующая реплика-источник")
        turn = turns[source["turn_idx"]]
        if fact["speaker"] != turn["speaker"]:
            raise ValueError(f"facts[{i}]: speaker не соответствует источнику")
        span, quote = source.get("span"), source.get("quote")
        if quote is not None and (not isinstance(quote, str) or quote not in turn["text"]):
            raise ValueError(f"facts[{i}]: цитата отсутствует в реплике")
        if span is not None:
            if not isinstance(span, list) or len(span) != 2 or any(type(n) is not int for n in span):
                raise ValueError(f"facts[{i}]: некорректный span")
            lo, hi = span
            if not 0 <= lo < hi <= len(turn["text"]):
                raise ValueError(f"facts[{i}]: span за пределами реплики")
            if quote is not None and turn["text"][lo:hi] != quote:
                raise ValueError(f"facts[{i}]: span и quote расходятся")
    return value["facts"]
