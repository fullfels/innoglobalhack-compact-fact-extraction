#!/usr/bin/env python3
"""T03/T04/T05: QA корпуса, leakage-free split по account_id/series и сборка train-ready v1."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from pydantic import ValidationError
from sklearn.model_selection import GroupShuffleSplit

from src.contracts.models import CallInput, Fact


BROKEN_ID_RE = re.compile(r"^u\d+_\d+:(ticket|dev|svc|client)\d*$")
VALID_OBJ_TYPES = {"service", "device", "ticket", "client"}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def extract_account_id(record: dict[str, Any]) -> str:
    series = record.get("series")
    if isinstance(series, dict) and series.get("account_id"):
        return str(series["account_id"])
    call_id = str(record["call_id"])
    return call_id.split(":", 1)[0]


def infer_object_type(obj: dict[str, Any]) -> str | None:
    raw_type = obj.get("type")
    if raw_type in VALID_OBJ_TYPES:
        return raw_type
    oid = str(obj.get("id") or "")
    if ":ticket" in oid:
        return "ticket"
    if ":dev" in oid:
        return "device"
    if ":svc" in oid:
        return "service"
    if ":client" in oid:
        return "client"
    return None


def sanitize_fact(raw_fact: dict[str, Any], turns_map: dict[int, dict[str, Any]], stats: Counter) -> dict[str, Any]:
    fact = dict(raw_fact)

    # 1. Удаляем лишнее поле confirmed, запрещённое контрактом Fact (extra='forbid')
    if "confirmed" in fact:
        fact.pop("confirmed")
        stats["stripped_confirmed_field"] += 1

    # 2. Чиним object.type ('other' -> валидный тип) и битый object.name
    obj = dict(fact["object"])
    fixed_type = infer_object_type(obj)
    if fixed_type != obj.get("type"):
        stats[f"fixed_obj_type:{obj.get('type')}->{fixed_type}"] += 1
        obj["type"] = fixed_type

    if BROKEN_ID_RE.match(str(obj.get("name", ""))):
        fallback_names = {"ticket": "заявка", "device": "устройство", "service": "услуга", "client": "клиент"}
        obj["name"] = fallback_names.get(fixed_type or "", obj["name"])
        stats["fixed_broken_object_name"] += 1
    fact["object"] = obj

    # 3. Проверка и очистка ненадёжных spans (Python code-point offsets)
    src = dict(fact["source"])
    turn = turns_map.get(src["turn_idx"])
    span = src.get("span")
    quote = src.get("quote")

    if turn is None:
        src["span"] = None
        src["quote"] = None
        stats["cleared_missing_or_empty_turn_source"] += 1
    elif span is not None:
        text = turn["text"]
        s, e = span
        if not (isinstance(s, int) and isinstance(e, int) and 0 <= s < e <= len(text)):
            src["span"] = None
            src["quote"] = None
            stats["cleared_out_of_bounds_span"] += 1
        elif quote is not None and text[s:e] != quote:
            src["span"] = None
            src["quote"] = None
            stats["cleared_mismatched_span_quote"] += 1
        elif quote is None:
            src["quote"] = text[s:e]
            stats["filled_quote_from_valid_span"] += 1
    elif quote is not None:
        text = turn["text"]
        pos = text.find(quote)
        if pos != -1:
            src["span"] = [pos, pos + len(quote)]
            stats["recovered_exact_span_from_quote"] += 1
        else:
            stats["ungrounded_asr_cleaned_quote"] += 1

    fact["source"] = src
    return fact


def detect_call_flags(turns: list[dict[str, Any]], facts: list[dict[str, Any]]) -> dict[str, Any]:
    is_speaker_inverted = any(t["text"].startswith("**") for t in turns)

    is_multi_session = False
    if len(turns) > 18:
        for t in turns[:-3]:
            low = t["text"].lower()
            if any(farewell in low for farewell in ("всего доброго", "до свидания", "хорошего дня", "пока-пока")):
                is_multi_session = True
                break

    return {
        "is_speaker_inverted": is_speaker_inverted,
        "is_multi_session": is_multi_session,
        "is_long_dialogue": len(turns) >= 40,
        "has_empty_facts": len(facts) == 0,
    }


def prepare_train_ready_v1(
    train_input: str,
    output_dir: str,
    val_input: str | None = None,
    dev_size: float = 0.15,
    random_seed: int = 42,
) -> None:
    in_path = Path(train_input)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    raw_records: list[dict[str, Any]] = []
    with in_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                raw_records.append(json.loads(line))

    stats: Counter = Counter()
    cleaned_records: list[dict[str, Any]] = []
    seen_call_ids: set[str] = set()

    for rec in raw_records:
        cid = rec["call_id"]
        if cid in seen_call_ids:
            stats["skipped_duplicate_call_id"] += 1
            continue
        seen_call_ids.add(cid)

        # Отфильтровываем пустые реплики (text == ""), сохраняя оригинальные turn.idx и text
        raw_turns = rec.get("turns", [])
        non_empty_turns = []
        for t in raw_turns:
            if not t.get("text"):
                stats["removed_empty_turns"] += 1
                continue
            non_empty_turns.append(t)

        if not non_empty_turns:
            stats["skipped_all_empty_turns_call"] += 1
            continue

        features = {
            "call_id": rec["call_id"],
            "started_at": rec["started_at"],
            "duration_sec": rec.get("duration_sec"),
            "series": rec.get("series"),
            "turns": non_empty_turns,
            "noise_level": rec.get("noise_level"),
        }
        CallInput.model_validate(features)

        turns_map = {t["idx"]: t for t in non_empty_turns}
        clean_facts = []
        for raw_f in rec.get("facts", []):
            cf = sanitize_fact(raw_f, turns_map, stats)
            try:
                Fact.model_validate(cf)
                clean_facts.append(cf)
            except ValidationError as exc:
                err_type = exc.errors()[0]["type"] if exc.errors() else "unknown"
                stats[f"dropped_invalid_fact:{err_type}"] += 1

        qa_flags = detect_call_flags(non_empty_turns, clean_facts)
        for k, v in qa_flags.items():
            if v:
                stats[f"flag:{k}"] += 1

        cleaned_records.append(
            {
                **features,
                "facts": clean_facts,
                "qa_meta": {
                    "account_id": extract_account_id(rec),
                    **qa_flags,
                },
            }
        )

    # Leakage-free split строго по account_id / series
    groups = [r["qa_meta"]["account_id"] for r in cleaned_records]
    gss = GroupShuffleSplit(n_splits=1, test_size=dev_size, random_state=random_seed)
    train_idx, dev_idx = next(gss.split(cleaned_records, groups=groups))

    train_data = [cleaned_records[i] for i in train_idx]
    dev_data = [cleaned_records[i] for i in dev_idx]

    train_accounts = {r["qa_meta"]["account_id"] for r in train_data}
    dev_accounts = {r["qa_meta"]["account_id"] for r in dev_data}
    assert len(train_accounts & dev_accounts) == 0, "LEAKAGE: пересечение account_id между train и dev!"

    train_calls = {r["call_id"] for r in train_data}
    dev_calls = {r["call_id"] for r in dev_data}
    assert len(train_calls & dev_calls) == 0, "LEAKAGE: пересечение call_id между train и dev!"

    val_sha = None
    val_account_overlap = 0
    if val_input and Path(val_input).exists():
        val_path = Path(val_input)
        val_sha = sha256_file(val_path)
        val_records = [json.loads(line) for line in val_path.open("r", encoding="utf-8") if line.strip()]
        val_accounts = {extract_account_id(r) for r in val_records}
        val_account_overlap = len((train_accounts | dev_accounts) & val_accounts)

    train_file = out_path / "train_ready.jsonl"
    dev_file = out_path / "dev_ready.jsonl"
    split_file = out_path / "split_ids.json"
    manifest_file = out_path / "manifest.json"

    with train_file.open("w", encoding="utf-8") as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    with dev_file.open("w", encoding="utf-8") as f:
        for item in dev_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    split_payload = {
        "random_seed": random_seed,
        "split_strategy": "GroupShuffleSplit by account_id (series.account_id / call_id prefix)",
        "train_call_ids": [r["call_id"] for r in train_data],
        "dev_call_ids": [r["call_id"] for r in dev_data],
        "train_account_ids": sorted(train_accounts),
        "dev_account_ids": sorted(dev_accounts),
    }
    split_file.write_text(json.dumps(split_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    manifest = {
        "dataset_version": "train-ready-v1",
        "source_train_file": str(in_path),
        "source_train_sha256": sha256_file(in_path),
        "source_val_sha256": val_sha,
        "val_account_overlap": val_account_overlap,
        "random_seed": random_seed,
        "split_strategy": "GroupShuffleSplit by account_id",
        "counts": {
            "total_calls": len(cleaned_records),
            "train_calls": len(train_data),
            "dev_calls": len(dev_data),
            "train_accounts": len(train_accounts),
            "dev_accounts": len(dev_accounts),
            "train_facts": sum(len(r["facts"]) for r in train_data),
            "dev_facts": sum(len(r["facts"]) for r in dev_data),
        },
        "leakage_checks": {
            "account_overlap_train_dev": 0,
            "call_overlap_train_dev": 0,
            "passed": True,
        },
        "transformations_and_qa": dict(sorted(stats.items())),
        "ml_contract_notes": {
            "input_fields": ["call_id", "started_at", "duration_sec", "series", "turns", "noise_level"],
            "target_field": "facts",
            "metadata_excluded_from_model": ["qa_meta"],
            "empty_turns_policy": "Empty string turns removed while preserving original turn.idx and exact text.",
            "span_policy": "Invalid/mismatched spans cleared to null; valid spans are Python code-point [start, end).",
        },
        "outputs": {
            "train_ready_sha256": sha256_file(train_file),
            "dev_ready_sha256": sha256_file(dev_file),
        },
    }
    manifest_file.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Готово: {train_file} ({len(train_data)}), {dev_file} ({len(dev_data)}), {manifest_file}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="train.jsonl")
    ap.add_argument("--val", default="val.jsonl")
    ap.add_argument("--out", default="data/train_ready_v1")
    ap.add_argument("--dev-size", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    prepare_train_ready_v1(args.train, args.out, val_input=args.val, dev_size=args.dev_size, random_seed=args.seed)