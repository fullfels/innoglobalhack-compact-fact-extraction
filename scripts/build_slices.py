#!/usr/bin/env python3
"""T29: Формирование детерминированных диагностических срезов на dev_ready.jsonl."""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

TRAP_RE = re.compile(
    r"\b(да нет|да нит|не надо|не нужно|не пользуюсь|больше не пользуюсь|не мой тариф|не подключал\w*)\b",
    flags=re.IGNORECASE,
)


def build_slices(dev_path: str, out_path: str) -> None:
    records = [
        json.loads(line)
        for line in Path(dev_path).open("r", encoding="utf-8")
        if line.strip()
    ]

    slices: dict[str, list[str]] = {
        "long_dialogues": [],
        "multi_session": [],
        "speaker_inverted": [],
        "empty_facts": [],
        "negations": [],
        "state_transitions": [],
        "multi_object": [],
        "temporal_shifted": [],
        "traps_negation_rejection": [],
    }

    for rec in records:
        cid = rec["call_id"]
        qa = rec.get("qa_meta", {})
        facts = rec.get("facts", [])
        turns = rec.get("turns", [])
        started_day = str(rec.get("started_at", ""))[:10]

        if qa.get("is_long_dialogue"):
            slices["long_dialogues"].append(cid)
        if qa.get("is_multi_session"):
            slices["multi_session"].append(cid)
        if qa.get("is_speaker_inverted"):
            slices["speaker_inverted"].append(cid)
        if qa.get("has_empty_facts") or not facts:
            slices["empty_facts"].append(cid)

        if any(f.get("polarity") == -1 for f in facts):
            slices["negations"].append(cid)

        # Переходы состояний: закрытый интервал valid_to != null или >= 2 фактов на один (object.id, attribute)
        obj_attr_counts = Counter((f["object"].get("id") or f["object"]["name"], f["attribute"]) for f in facts)
        if any(f.get("valid_to") is not None for f in facts) or any(cnt >= 2 for cnt in obj_attr_counts.values()):
            slices["state_transitions"].append(cid)

        # Multi-object: 2 и более уникальных объекта в одном звонке
        distinct_objects = {f["object"].get("id") or f["object"]["name"] for f in facts}
        if len(distinct_objects) >= 2:
            slices["multi_object"].append(cid)

        # Сдвинутые даты: valid_from отличается от дня звонка
        if any(str(f.get("valid_from", ""))[:10] != started_day for f in facts):
            slices["temporal_shifted"].append(cid)

        # Реплики-ловушки («да нет», «не пользуюсь», «не мой тариф»)
        if any(TRAP_RE.search(t.get("text", "")) for t in turns):
            slices["traps_negation_rejection"].append(cid)

    payload: dict[str, Any] = {
        "source_dev_file": str(dev_path),
        "total_dev_calls": len(records),
        "definitions": {
            "long_dialogues": "Звонки с >= 40 репликами (qa_meta.is_long_dialogue)",
            "multi_session": "Склеенные диалоги с прощанием в середине (qa_meta.is_multi_session)",
            "speaker_inverted": "Звонки с инверсией ролей спикеров '**' (qa_meta.is_speaker_inverted)",
            "empty_facts": "Звонки без gold-фактов (facts == [])",
            "negations": "Звонки, содержащие факты с polarity == -1",
            "state_transitions": "Звонки со сменой состояний (valid_to != null или >= 2 фактов по одной паре object+attribute)",
            "multi_object": "Звонки с >= 2 различными сущностями в facts",
            "temporal_shifted": "Звонки с датами фактов, отличными от дня начала звонка",
            "traps_negation_rejection": "Звонки с репликами-ловушками («да нет», «не пользуюсь», «не мой тариф»)",
        },
        "counts": {k: len(v) for k, v in slices.items()},
        "slice_ids": {k: sorted(v) for k, v in slices.items()},
    }

    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Сохранено в {out_file}:")
    for k, cnt in payload["counts"].items():
        print(f"  {k:26} {cnt}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", default="data/train_ready_v1/dev_ready.jsonl")
    ap.add_argument("--out", default="data/train_ready_v1/slice_ids.json")
    args = ap.parse_args()
    build_slices(args.dev, args.out)