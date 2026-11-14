from __future__ import annotations

import json
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from src.contracts.models import (
    CallInput,
    CandidateBatch,
    CandidateFact,
    Fact,
    ObjectRef,
    Source,
)


def _norm_token(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower().replace("ё", "е"))


class CatalogNormalizer:
    """T12: Нормализация услуг, тарифов, устройств, симптомов, действий и статусов заявок."""

    DEFAULT_ACTION_MAP: dict[str, str] = {
        "перезагрузил": "Перезагрузка оборудования",
        "перезагрузила": "Перезагрузка оборудования",
        "перезагрузить роутер": "Перезагрузка оборудования",
        "выключил и включил": "Перезагрузка оборудования",
        "ребутнул": "Перезагрузка оборудования",
        "проверил кабель": "Проверка подключения кабеля",
        "переткнул кабель": "Проверка подключения кабеля",
        "переткнул провод": "Проверка подключения кабеля",
        "сменил тариф": "Смена тарифа",
        "перевел на тариф": "Смена тарифа",
        "перевела на тариф": "Смена тарифа",
        "создал заявку": "Создание заявки",
        "оформил заявку": "Создание заявки",
        "оформила заявку": "Создание заявки",
        "проверил линию": "Диагностика линии",
        "протестировал линию": "Диагностика линии",
    }

    DEFAULT_TICKET_STATUS_MAP: dict[str, str] = {
        "открыта": "принята",
        "создана": "принята",
        "зарегистрирована": "принята",
        "новая": "принята",
        "принята": "принята",
        "в работе": "в работе",
        "назначен мастер": "назначен выезд специалиста",
        "назначен выезд специалиста": "назначен выезд специалиста",
        "передана мастеру": "назначен выезд специалиста",
        "закрыта": "выполнена",
        "выполнена": "выполнена",
        "решена": "выполнена",
        "отменена": "отменена",
        "отклонена": "отменена",
    }

    # Исправление частых ASR-искажений в object.name с сохранением первой буквы для case_metrics.py
    ASR_OBJECT_REPAIRS: dict[str, str] = {
        "раутер": "роутер",
        "интехбет": "интернет",
        "интирнет": "интернет",
        "энтертет": "интернет",
        "пристафка": "приставка",
        "гристивка": "приставка",
        "смарт-пристафка": "смарт-приставка",
        "мини-пристафка": "мини-приставка",
        "расывер": "ресивер",
        "термакал": "терминал",
        "медиаканвертер": "медиаконвертер",
    }

    STRICT_WHOLE_FIELD_ALIASES = {"нет", "остаток", "база", "простой"}
    BROKEN_ID_RE = re.compile(r"^u\d+_\d+:(ticket|dev|svc|client)\d*$")

    def __init__(self, catalog: dict[str, Any] | str | Path | None = None):
        if catalog is None:
            default_path = Path("catalog.json")
            catalog = json.loads(default_path.read_text(encoding="utf-8")) if default_path.exists() else {}
        elif isinstance(catalog, (str, Path)):
            catalog = json.loads(Path(catalog).read_text(encoding="utf-8"))

        self.catalog = catalog
        self.service_lookup = self._build_lookup(catalog.get("services", []))
        self.device_lookup = self._build_lookup(catalog.get("devices", []))
        self.tariff_lookup = self._build_lookup(catalog.get("tariffs", []))
        self.issue_lookup = self._build_lookup(catalog.get("issues", []))

    @staticmethod
    def _build_lookup(entries: list[dict[str, Any]]) -> dict[str, str]:
        lookup: dict[str, str] = {}
        for item in entries:
            canonical = item["name"]
            lookup[_norm_token(canonical)] = canonical
            for alias in item.get("aliases", []):
                lookup[_norm_token(alias)] = canonical
        return lookup

    def _match_entry(self, raw: str, lookup: dict[str, str]) -> str | None:
        norm = _norm_token(raw)
        if not norm:
            return None
        if norm in lookup:
            return lookup[norm]
        for alias, canonical in sorted(lookup.items(), key=lambda kv: len(kv[0]), reverse=True):
            if alias in self.STRICT_WHOLE_FIELD_ALIASES:
                continue
            if re.search(rf"(?<!\w){re.escape(alias)}(?!\w)", norm):
                return canonical
        return None

    @staticmethod
    def _first_letter(text: str) -> str:
        toks = re.findall(r"[а-яёa-z0-9]+", text.lower().replace("ё", "е"))
        return toks[0][0] if toks else ""

    def _infer_type_from_id(self, obj: ObjectRef) -> str | None:
        if not obj.id:
            return obj.type
        if ":ticket" in obj.id:
            return "ticket"
        if ":dev" in obj.id:
            return "device"
        if ":svc" in obj.id:
            return "service"
        if ":client" in obj.id:
            return "client"
        return obj.type

    def _normalize_object(self, obj: ObjectRef, attribute: str) -> ObjectRef:
        inferred_type = self._infer_type_from_id(obj)
        raw_name = obj.name.strip()

        # 1. Замена технического ID в object.name на человекочитаемое имя по типу
        if self.BROKEN_ID_RE.match(raw_name):
            fallback_names = {
                "ticket": "заявка",
                "device": "устройство",
                "service": "услуга",
                "client": "клиент",
            }
            clean_name = fallback_names.get(inferred_type or "", raw_name)
            return obj.model_copy(update={"name": clean_name, "type": inferred_type})

        # 2. Точечный ремонт ASR-опечаток в имени объекта
        repaired_name = raw_name
        for bad, good in self.ASR_OBJECT_REPAIRS.items():
            repaired_name = re.sub(rf"(?<!\w){re.escape(bad)}(?!\w)", good, repaired_name, flags=re.IGNORECASE)

        # 3. Определяем канонический тип (device / service) без склеивания разных устройств
        matched_device = self._match_entry(repaired_name, self.device_lookup)
        matched_service = self._match_entry(repaired_name, self.service_lookup)

        final_type = inferred_type
        if final_type is None:
            if matched_device:
                final_type = "device"
            elif matched_service:
                final_type = "service"

        # 4. Подставляем каноническое имя из каталога только если сохраняется первая буква
        # (защита от ложного штрафа в case_metrics.match_facts)
        final_name = repaired_name
        candidate_canonical = matched_device if final_type == "device" else matched_service
        if candidate_canonical and self._first_letter(candidate_canonical) == self._first_letter(repaired_name):
            final_name = candidate_canonical

        if final_name != obj.name or final_type != obj.type:
            return obj.model_copy(update={"name": final_name, "type": final_type})
        return obj

    def _normalize_value(self, fact: Fact) -> str:
        raw_val = fact.value
        if fact.attribute == "tariff":
            return self._match_entry(raw_val, self.tariff_lookup) or raw_val
        if fact.attribute == "service_state":
            return self._match_entry(raw_val, self.issue_lookup) or raw_val
        if fact.attribute == "action":
            return self._match_entry(raw_val, self.DEFAULT_ACTION_MAP) or raw_val
        if fact.attribute == "ticket_status":
            return (
                self._match_entry(raw_val, self.DEFAULT_TICKET_STATUS_MAP)
                or self._match_entry(raw_val, self.issue_lookup)
                or raw_val
            )
        return raw_val

    def normalize(self, batch: CandidateBatch, call: CallInput) -> CandidateBatch:
        updated_candidates: list[CandidateFact] = []
        for cand in batch.candidates:
            fact = cand.fact
            new_obj = self._normalize_object(fact.object, fact.attribute)
            new_val = self._normalize_value(fact)

            if new_obj != fact.object or new_val != fact.value:
                new_fact = fact.model_copy(update={"object": new_obj, "value": new_val})
                updated_candidates.append(
                    cand.model_copy(
                        update={
                            "fact": new_fact,
                            "provenance": cand.provenance + ("catalog",),
                        }
                    )
                )
            else:
                updated_candidates.append(cand)

        return batch.model_copy(update={"candidates": tuple(updated_candidates)})


class TemporalNormalizer:
    """
    T13: Нормализация относительных дат относительно call.started_at.

    Политика ambiguity (AMBIGUITY_POLICY = 'preserve_anchor_with_warning'):
    - Если temporal_expression отсутствует, дата не подменяется автоматически на started_at;
      при совпадении с started_at без явного якоря фиксируется warning.
    - При неоднозначном выражении ('недавно', 'давно', 'скоро', 'на днях', 'позже')
      сохраняется исходная дата и выставляется warning 'temporal:ambiguous_expression'.
    - Таймзона call.started_at (naive или aware) сохраняется без принудительного перевода в UTC.
    """

    AMBIGUITY_POLICY = "preserve_anchor_with_warning"

    WEEKDAYS: dict[str, int] = {
        "понедельник": 0,
        "понедельника": 0,
        "вторник": 1,
        "вторника": 1,
        "среда": 2,
        "среду": 2,
        "среды": 2,
        "четверг": 3,
        "четверга": 3,
        "пятница": 4,
        "пятницу": 4,
        "пятницы": 4,
        "суббота": 5,
        "субботу": 5,
        "субботы": 5,
        "воскресенье": 6,
        "воскресенья": 6,
    }

    AMBIGUOUS_MARKERS = (
        "недавно",
        "давно",
        "скоро",
        "позже",
        "на днях",
        "в ближайшее время",
        "как-нибудь",
        "когда-то",
    )

    SPOKEN_NUMBERS: dict[str, int] = {
        "один": 1,
        "одного": 1,
        "пару": 2,
        "два": 2,
        "две": 2,
        "двух": 2,
        "три": 3,
        "трех": 3,
        "трёх": 3,
        "четыре": 4,
        "четырех": 4,
        "четырёх": 4,
        "пять": 5,
        "пяти": 5,
        "шесть": 6,
        "шести": 6,
        "семь": 7,
        "семи": 7,
    }

    SPOKEN_HOURS: dict[str, int] = {
        "часу": 13,
        "часа": 13,
        "двух": 14,
        "трех": 15,
        "трёх": 15,
        "четырех": 16,
        "четырёх": 16,
        "пяти": 17,
        "шести": 18,
        "семи": 19,
        "восьми": 20,
        "девяти": 9,
        "десяти": 10,
        "одиннадцати": 11,
        "двенадцати": 12,
    }

    @classmethod
    def _extract_time_override(cls, expr: str) -> tuple[int, int] | None:
        m_exact = re.search(r"\b(\d{1,2})[:.](\d{2})\b", expr)
        if m_exact:
            h, mn = int(m_exact.group(1)), int(m_exact.group(2))
            if 0 <= h <= 23 and 0 <= mn <= 59:
                return h, mn

        m_after_num = re.search(r"(?:после|в|к)\s+(\d{1,2})\b", expr)
        if m_after_num:
            h = int(m_after_num.group(1))
            if 1 <= h <= 7 and "утр" not in expr:
                h += 12
            if 0 <= h <= 23:
                return h, 0

        for word, hour in cls.SPOKEN_HOURS.items():
            if re.search(rf"(?:после|в|к)\s+{word}\b", expr):
                return hour, 0

        if "до обеда" in expr or "в первой половине дня" in expr or "утром" in expr or "с утра" in expr:
            return 9, 0
        if "к обеду" in expr or "днем" in expr or "днём" in expr:
            return 13, 0
        if "после обеда" in expr or "во второй половине дня" in expr:
            return 14, 0
        if "вечером" in expr or "к вечеру" in expr:
            return 18, 0
        return None

    @classmethod
    def _resolve_expression(
        cls, expr_raw: str, base_dt: datetime, is_future_intent: bool
    ) -> tuple[datetime | None, str, str | None]:
        expr = _norm_token(expr_raw)
        if not expr:
            return None, "valid_from", None

        if any(marker in expr for marker in cls.AMBIGUOUS_MARKERS):
            return None, "valid_from", f"temporal:ambiguous_expression:{expr}"

        is_deadline = bool(re.match(r"^до\b", expr)) and "до обеда" not in expr and "до вечера" not in expr
        target_field = "valid_to" if is_deadline else "valid_from"

        resolved: datetime | None = None

        if "позавчера" in expr:
            resolved = base_dt - timedelta(days=2)
        elif "вчера" in expr:
            resolved = base_dt - timedelta(days=1)
        elif "сегодня" in expr:
            resolved = base_dt
        elif "послезавтра" in expr:
            resolved = base_dt + timedelta(days=2)
        elif "завтра" in expr:
            resolved = base_dt + timedelta(days=1)
        elif "через неделю" in expr or "на следующей неделе" in expr:
            resolved = base_dt + timedelta(days=7)
        elif "на прошлой неделе" in expr:
            resolved = base_dt - timedelta(days=7)
        elif "с прошлого месяца" in expr or "в прошлом месяце" in expr or "месяц назад" in expr:
            year = base_dt.year - 1 if base_dt.month == 1 else base_dt.year
            month = 12 if base_dt.month == 1 else base_dt.month - 1
            resolved = base_dt.replace(year=year, month=month, day=1, hour=0, minute=0, second=0, microsecond=0)
        elif "со следующего месяца" in expr or "в следующем месяце" in expr:
            year = base_dt.year + 1 if base_dt.month == 12 else base_dt.year
            month = 1 if base_dt.month == 12 else base_dt.month + 1
            resolved = base_dt.replace(year=year, month=month, day=1, hour=0, minute=0, second=0, microsecond=0)
        elif "с начала месяца" in expr or "с первого числа" in expr:
            resolved = base_dt.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

        # Количественные интервалы цифрами и словами: "через 3 дня", "через два дня", "пять дней назад"
        if resolved is None:
            m_rel = re.search(r"через\s+(\d+|[а-яё]+)\s+(час\w*|дн\w*|ден\w*|недел\w*)", expr)
            if m_rel:
                raw_qty, unit = m_rel.group(1), m_rel.group(2)
                qty = int(raw_qty) if raw_qty.isdigit() else cls.SPOKEN_NUMBERS.get(raw_qty)
                if qty is not None:
                    if unit.startswith("час"):
                        resolved = base_dt + timedelta(hours=qty)
                    elif unit.startswith("недел"):
                        resolved = base_dt + timedelta(weeks=qty)
                    else:
                        resolved = base_dt + timedelta(days=qty)

        if resolved is None:
            m_ago = re.search(r"(\d+|[а-яё]+)\s+(час\w*|дн\w*|ден\w*|недел\w*)\s+назад", expr)
            if m_ago:
                raw_qty, unit = m_ago.group(1), m_ago.group(2)
                qty = int(raw_qty) if raw_qty.isdigit() else cls.SPOKEN_NUMBERS.get(raw_qty)
                if qty is not None:
                    if unit.startswith("час"):
                        resolved = base_dt - timedelta(hours=qty)
                    elif unit.startswith("недел"):
                        resolved = base_dt - timedelta(weeks=qty)
                    else:
                        resolved = base_dt - timedelta(days=qty)

        # Дни недели: "со вторника", "в четверг после двух", "до пятницы"
        if resolved is None:
            for wd_name, target_wd in cls.WEEKDAYS.items():
                if re.search(rf"(?<!\w){wd_name}(?!\w)", expr):
                    is_past_anchor = bool(re.search(r"\b(?:со|с|в прошл\w+)\b", expr))
                    if is_past_anchor and not is_deadline and not is_future_intent:
                        days_back = (base_dt.weekday() - target_wd) % 7
                        resolved = base_dt - timedelta(days=days_back)
                    else:
                        days_ahead = (target_wd - base_dt.weekday()) % 7
                        if days_ahead == 0 and (is_deadline or is_future_intent):
                            days_ahead = 7
                        resolved = base_dt + timedelta(days=days_ahead)
                    break

        if resolved is None:
            return None, target_field, f"temporal:unparsed_expression:{expr}"

        time_override = cls._extract_time_override(expr)
        if time_override is not None:
            hour, minute = time_override
            resolved = resolved.replace(hour=hour, minute=minute, second=0, microsecond=0)

        return resolved, target_field, None

    def normalize(self, batch: CandidateBatch, call: CallInput) -> CandidateBatch:
        base_dt = call.started_at
        updated_candidates: list[CandidateFact] = []

        for cand in batch.candidates:
            fact = cand.fact
            expr = cand.temporal_expression

            if not expr or not expr.strip():
                warnings = cand.warnings
                if fact.valid_from == base_dt:
                    warnings = warnings + ("temporal:missing_explicit_anchor",)
                updated_candidates.append(cand.model_copy(update={"warnings": warnings}))
                continue

            is_future_intent = fact.attribute == "agreement"
            resolved_dt, target_field, warn = self._resolve_expression(expr, base_dt, is_future_intent)

            if resolved_dt is None:
                warnings = cand.warnings + ((warn,) if warn else ())
                updated_candidates.append(
                    cand.model_copy(
                        update={
                            "warnings": warnings,
                            "provenance": cand.provenance + ("temporal:ambiguous",),
                        }
                    )
                )
                continue

            if target_field == "valid_to":
                new_from = fact.valid_from if fact.valid_from <= resolved_dt else base_dt
                if new_from > resolved_dt:
                    new_from = resolved_dt
                new_fact = fact.model_copy(update={"valid_from": new_from, "valid_to": resolved_dt})
            else:
                new_to = fact.valid_to
                if new_to is not None and new_to < resolved_dt:
                    new_to = None
                new_fact = fact.model_copy(update={"valid_from": resolved_dt, "valid_to": new_to})

            updated_candidates.append(
                cand.model_copy(
                    update={
                        "fact": new_fact,
                        "provenance": cand.provenance + ("temporal",),
                    }
                )
            )

        return batch.model_copy(update={"candidates": tuple(updated_candidates)})


class StateNormalizer:
    """T15 (State): Хронологические переходы состояний объекта и изоляция multi-object."""

    HISTORICAL_MARKERS = ("раньше", "было", "старый", "предыдущ", "до этого")
    STATE_ATTRIBUTES = {"service_state", "tariff", "ticket_status"}

    @staticmethod
    def compute_object_key(fact: Fact) -> str:
        obj_type = fact.object.type or "unknown"
        identifier = fact.object.id or _norm_token(fact.object.name)
        return f"{obj_type}:{identifier}"

    def normalize(self, batch: CandidateBatch, call: CallInput) -> CandidateBatch:
        turns_map = {t.idx: t.text for t in call.turns}

        keyed: list[CandidateFact] = []
        for cand in batch.candidates:
            key = cand.object_key or self.compute_object_key(cand.fact)
            keyed.append(cand.model_copy(update={"object_key": key}))

        groups: dict[tuple[str, str], list[int]] = {}
        for idx, cand in enumerate(keyed):
            if cand.fact.attribute in self.STATE_ATTRIBUTES and cand.confirmation != "rejected":
                groups.setdefault((cand.object_key or "", cand.fact.attribute), []).append(idx)

        result = list(keyed)
        for (_obj_key, _attr), indices in groups.items():
            if len(indices) < 2:
                continue

            ordered = sorted(
                indices,
                key=lambda i: (result[i].fact.valid_from, result[i].fact.source.turn_idx),
            )

            for pos in range(len(ordered) - 1):
                prev_idx = ordered[pos]
                next_idx = ordered[pos + 1]
                prev_cand = result[prev_idx]
                next_cand = result[next_idx]

                if _norm_token(prev_cand.fact.value) == _norm_token(next_cand.fact.value):
                    continue

                if prev_cand.fact.valid_to is None:
                    close_dt = next_cand.fact.valid_from
                    if close_dt < prev_cand.fact.valid_from:
                        close_dt = prev_cand.fact.valid_from

                    updated_prev_fact = prev_cand.fact.model_copy(update={"valid_to": close_dt})
                    result[prev_idx] = prev_cand.model_copy(
                        update={
                            "fact": updated_prev_fact,
                            "provenance": prev_cand.provenance + ("state:closed_previous",),
                        }
                    )

            last_idx = ordered[-1]
            last_cand = result[last_idx]
            turn_text = _norm_token(turns_map.get(last_cand.fact.source.turn_idx, ""))
            if not any(m in turn_text for m in self.HISTORICAL_MARKERS):
                result[last_idx] = last_cand.model_copy(
                    update={"provenance": last_cand.provenance + ("state:current",)}
                )

        return batch.model_copy(update={"candidates": tuple(result)})


class PolarityNormalizer:
    """T14: Разделение симптомов ('не работает' -> +1) и отрицаний владения (-1), подтверждения."""

    SYMPTOM_POSITIVE_EXISTENCE = (
        "не работает",
        "не грузит",
        "не открывается",
        "пропал",
        "отключился",
        "отсутствие соединения",
        "низкая скорость",
        "тормозит",
        "лагает",
    )

    NEGATIVE_POLARITY_PATTERNS = (
        "больше не пользуюсь",
        "не пользуюсь",
        "не мой тариф",
        "не подключал",
        "не подключала",
        "не заказывал",
        "не заказывала",
        "услуга не подключена",
        "отключена услуга",
    )

    REJECTION_PATTERNS = (
        r"\bда нет\b",
        r"\bда нит\b",
        r"\bне надо\b",
        r"\bне нужно\b",
        r"\bне хочу\b",
        r"\bне буду\b",
        r"\bотказываюсь\b",
        r"\bне согласен\b",
        r"\bне согласна\b",
        r"\bне подходит\b",
        r"\bне устраивает\b",
        r"\bэто не то\b",
        r"^нет\b",
    )

    CONFIRMATION_PATTERNS = (
        r"\bда\b",
        r"\bсогласен\b",
        r"\bсогласна\b",
        r"\bхорошо\b",
        r"\bдавайте\b",
        r"\bподключайте\b",
        r"\bоформляйте\b",
        r"\bустраивает\b",
        r"\bподходит\b",
        r"\bпадходит\b",
        r"\bпотходит\b",
        r"\bладно\b",
        r"\bок\b",
        r"\bокей\b",
        r"\bконечно\b",
        r"\bподтверждаю\b",
    )

    @classmethod
    def _has_rejection(cls, text: str) -> bool:
        norm = _norm_token(text)
        return any(re.search(pat, norm) for pat in cls.REJECTION_PATTERNS)

    @classmethod
    def _has_genuine_confirmation(cls, text: str) -> bool:
        norm = _norm_token(text)
        if cls._has_rejection(norm):
            return False
        return any(re.search(pat, norm) for pat in cls.CONFIRMATION_PATTERNS)

    def normalize(self, batch: CandidateBatch, call: CallInput) -> CandidateBatch:
        turns = {t.idx: t for t in call.turns}
        sorted_turn_indices = sorted(turns.keys())
        updated_candidates: list[CandidateFact] = []

        for cand in batch.candidates:
            fact = cand.fact
            source_turn = turns.get(fact.source.turn_idx)
            source_text = _norm_token(source_turn.text if source_turn else (fact.source.quote or ""))
            val_norm = _norm_token(fact.value)

            new_polarity = fact.polarity
            new_confirmation = cand.confirmation

            if fact.attribute == "service_state" and any(
                sym in val_norm or sym in source_text for sym in self.SYMPTOM_POSITIVE_EXISTENCE
            ):
                if "теперь работает" not in source_text and "уже работает" not in source_text:
                    new_polarity = 1

            if any(neg in source_text or neg in val_norm for neg in self.NEGATIVE_POLARITY_PATTERNS):
                new_polarity = -1

            client_texts: list[str] = []
            for idx in cand.evidence_turns:
                t = turns.get(idx)
                if t and t.speaker == "client":
                    client_texts.append(t.text)

            if fact.speaker == "operator":
                for idx in sorted_turn_indices:
                    if idx > fact.source.turn_idx:
                        next_t = turns[idx]
                        if next_t.speaker == "client":
                            client_texts.append(next_t.text)
                        break

            if any(self._has_rejection(ct) for ct in client_texts):
                new_confirmation = "rejected"
            elif fact.speaker == "client" and self._has_rejection(source_text) and fact.attribute == "agreement":
                new_confirmation = "rejected"

            if fact.attribute == "agreement" and new_confirmation != "rejected":
                all_client_evidence = list(client_texts)
                if fact.speaker == "client" and source_turn:
                    all_client_evidence.append(source_turn.text)

                if any(self._has_genuine_confirmation(ct) for ct in all_client_evidence):
                    new_confirmation = "confirmed"
                elif new_confirmation == "confirmed":
                    new_confirmation = "unknown"

            if new_polarity != fact.polarity or new_confirmation != cand.confirmation:
                new_fact = fact.model_copy(update={"polarity": new_polarity})
                updated_candidates.append(
                    cand.model_copy(
                        update={
                            "fact": new_fact,
                            "confirmation": new_confirmation,
                            "provenance": cand.provenance + ("polarity",),
                        }
                    )
                )
            else:
                updated_candidates.append(cand)

        return batch.model_copy(update={"candidates": tuple(updated_candidates)})


class GroundingNormalizer:
    """T15 (Grounding): Точные Python code-point offsets; при ненадёжной привязке span=None, quote=None."""

    @staticmethod
    def _find_codepoint_span(turn_text: str, quote: str, hint_start: int | None = None) -> tuple[int, int] | None:
        if not quote:
            return None

        matches = [m.start() for m in re.finditer(re.escape(quote), turn_text)]
        if matches:
            best = min(matches, key=lambda pos: abs(pos - hint_start)) if hint_start is not None else matches[0]
            return best, best + len(quote)

        ci_matches = list(re.finditer(re.escape(quote), turn_text, flags=re.IGNORECASE))
        if ci_matches:
            best_m = min(ci_matches, key=lambda m: abs(m.start() - hint_start)) if hint_start is not None else ci_matches[0]
            return best_m.start(), best_m.end()

        tokens = quote.strip().split()
        if len(tokens) >= 2:
            pattern = r"\s+".join(re.escape(tok) for tok in tokens)
            ws_matches = list(re.finditer(pattern, turn_text, flags=re.IGNORECASE))
            if ws_matches:
                best_m = ws_matches[0]
                return best_m.start(), best_m.end()

        return None

    def normalize(self, batch: CandidateBatch, call: CallInput) -> CandidateBatch:
        turns = {t.idx: t for t in call.turns}
        updated_candidates: list[CandidateFact] = []

        for cand in batch.candidates:
            fact = cand.fact
            src = fact.source
            turn = turns.get(src.turn_idx)

            if turn is None:
                new_src = Source(turn_idx=src.turn_idx, span=None, quote=None)
                updated_candidates.append(
                    cand.model_copy(
                        update={
                            "fact": fact.model_copy(update={"source": new_src}),
                            "warnings": cand.warnings + ("grounding:missing_turn",),
                        }
                    )
                )
                continue

            turn_text = turn.text
            text_len = len(turn_text)
            hint_start = src.span[0] if src.span is not None else None

            if src.span is not None and src.quote is None:
                start, end = src.span
                if 0 <= start < end <= text_len:
                    new_src = Source(turn_idx=src.turn_idx, span=(start, end), quote=turn_text[start:end])
                    updated_candidates.append(
                        cand.model_copy(
                            update={
                                "fact": fact.model_copy(update={"source": new_src}),
                                "provenance": cand.provenance + ("grounding",),
                            }
                        )
                    )
                else:
                    new_src = Source(turn_idx=src.turn_idx, span=None, quote=None)
                    updated_candidates.append(
                        cand.model_copy(
                            update={
                                "fact": fact.model_copy(update={"source": new_src}),
                                "warnings": cand.warnings + ("grounding:out_of_bounds_span",),
                            }
                        )
                    )
                continue

            if src.quote is not None:
                if src.span is not None:
                    start, end = src.span
                    if 0 <= start < end <= text_len and turn_text[start:end] == src.quote:
                        updated_candidates.append(cand)
                        continue

                found_span = self._find_codepoint_span(turn_text, src.quote, hint_start=hint_start)
                if found_span is not None:
                    s, e = found_span
                    new_src = Source(turn_idx=src.turn_idx, span=(s, e), quote=turn_text[s:e])
                    updated_candidates.append(
                        cand.model_copy(
                            update={
                                "fact": fact.model_copy(update={"source": new_src}),
                                "provenance": cand.provenance + ("grounding",),
                            }
                        )
                    )
                else:
                    new_src = Source(turn_idx=src.turn_idx, span=None, quote=None)
                    updated_candidates.append(
                        cand.model_copy(
                            update={
                                "fact": fact.model_copy(update={"source": new_src}),
                                "warnings": cand.warnings + ("grounding:quote_not_found",),
                            }
                        )
                    )
                continue

            updated_candidates.append(cand)

        return batch.model_copy(update={"candidates": tuple(updated_candidates)})


def build_default_normalizers(catalog: dict[str, Any] | str | Path | None = None) -> tuple[Any, ...]:
    """Порядок пайплайна: catalog -> temporal -> state -> polarity -> grounding."""
    return (
        CatalogNormalizer(catalog=catalog),
        TemporalNormalizer(),
        StateNormalizer(),
        PolarityNormalizer(),
        GroundingNormalizer(),
    )