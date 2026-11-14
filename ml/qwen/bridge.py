"""CallInput -> generated official facts -> the shared CandidateBatch contract.

Grounding is validated, never repaired by changing speaker or moving turn_idx.
Temporal/confirmation resolution can be injected by DE; no invented date fallback.
"""
from __future__ import annotations
from copy import deepcopy
import json
import math
import re
from typing import Callable, Protocol
from pydantic import ValidationError
from src.contracts.models import CallInput, CandidateBatch, CandidateFact, Fact
from src.pipeline.adapters import PipelineError


def strict_json(text):
    def finite_float(value):
        result = float(value)
        if not math.isfinite(result):
            raise ValueError("non-finite JSON number")
        return result
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result
    fenced = re.fullmatch(r"```(?:json)?[ \t]*\r?\n(.*?)\r?\n```", text.strip(), re.S | re.I)
    if fenced:
        text = fenced.group(1)
    return json.loads(text, object_pairs_hook=pairs, parse_float=finite_float,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError("non-finite JSON")))


def model_input(call: CallInput):
    return {"call_id": call.call_id, "started_at": call.started_at.isoformat(timespec="seconds"),
            "turns": [{"idx": t.idx, "speaker": t.speaker, "text": t.text} for t in call.turns]}


def messages(call: CallInput, instruction, catalog):
    return [{"role": "system", "content": instruction + "\n\nКаталог:\n" + json.dumps(catalog, ensure_ascii=False, separators=(",", ":"))},
            {"role": "user", "content": json.dumps(model_input(call), ensure_ascii=False)}]


def parse_candidates(raw: str, call: CallInput, *, temporal_resolver=None, confirmation_resolver=None):
    try:
        value = strict_json(raw)
    except (ValueError, TypeError) as exc:
        raise PipelineError("QWEN_INVALID_JSON", 503, "qwen.parse", type(exc).__name__) from None
    if not isinstance(value, dict) or set(value) != {"facts"} or not isinstance(value["facts"], list):
        raise PipelineError("QWEN_INVALID_OUTPUT_SCHEMA", 503, "qwen.parse")
    turns = {t.idx: t for t in call.turns}
    candidates = []
    for original in value["facts"]:
        if not isinstance(original, dict):
            raise PipelineError("QWEN_INVALID_FACT", 503, "qwen.parse")
        row = deepcopy(original)
        if row.get("valid_from") is None:
            if temporal_resolver is None:
                raise PipelineError("QWEN_UNRESOLVED_TIME", 503, "qwen.temporal")
            # Resolver must return a justified absolute value or raise; never started_at fallback.
            row["valid_from"] = temporal_resolver(deepcopy(row), call)
            if row["valid_from"] is None:
                raise PipelineError("QWEN_UNRESOLVED_TIME", 503, "qwen.temporal")
        source = row.get("source")
        if not isinstance(source, dict) or type(source.get("turn_idx")) is not int or source["turn_idx"] not in turns:
            raise PipelineError("QWEN_INVALID_SOURCE", 503, "qwen.grounding")
        turn = turns[source["turn_idx"]]
        if row.get("speaker") != turn.speaker:
            raise PipelineError("QWEN_SPEAKER_SOURCE_MISMATCH", 503, "qwen.grounding")
        quote, span = source.get("quote"), source.get("span")
        warnings = []
        if span is None:
            if quote is None:
                source.update(span=None, quote=None)
            elif not isinstance(quote, str) or not quote or quote not in turn.text:
                raise PipelineError("QWEN_QUOTE_NOT_IN_SOURCE", 503, "qwen.grounding")
            else:
                start = turn.text.find(quote)
                if turn.text.find(quote, start + 1) >= 0:
                    source.update(span=None, quote=None)
                    warnings.append("Ambiguous optional quote offsets omitted; original turn retained")
                else:
                    source["span"] = [start, start + len(quote)]
        else:
            if not isinstance(span, (list, tuple)) or len(span) != 2 or any(type(x) is not int for x in span):
                raise PipelineError("QWEN_INVALID_SPAN", 503, "qwen.grounding")
            start, end = span
            if not 0 <= start < end <= len(turn.text):
                raise PipelineError("QWEN_INVALID_SPAN", 503, "qwen.grounding")
            literal = turn.text[start:end]
            if quote is not None and quote != literal:
                raise PipelineError("QWEN_QUOTE_SPAN_MISMATCH", 503, "qwen.grounding")
            source["quote"] = literal
        try:
            fact = Fact.model_validate(row)
            if not fact.object.name.strip() or not fact.value.strip():
                raise ValueError("blank object/value")
        except (ValidationError, ValueError) as exc:
            raise PipelineError("QWEN_INVALID_FACT", 503, "qwen.parse", type(exc).__name__) from None
        candidate = CandidateFact(fact=fact, object_key=fact.object.id,
                                  evidence_turns=(fact.source.turn_idx,),
                                  provenance=("qwen-generated", "original-turn-grounding-checked"),
                                  warnings=tuple(warnings))
        if fact.attribute == "agreement":
            if confirmation_resolver is None:
                candidate = CandidateFact(**{**candidate.model_dump(), "warnings": candidate.warnings +
                                             ("Agreement acceptance unresolved; DE confirmation required before export",)})
            else:
                resolved = CandidateFact.model_validate(confirmation_resolver(candidate, call))
                if resolved.fact != fact:
                    raise PipelineError("QWEN_CONFIRMATION_RESOLVER_CHANGED_FACT", 503, "qwen.confirmation")
                if any(i not in turns for i in resolved.evidence_turns):
                    raise PipelineError("QWEN_INVALID_CONFIRMATION_EVIDENCE", 503, "qwen.confirmation")
                if resolved.confirmation == "confirmed" and not any(turns[i].speaker == "client" for i in resolved.evidence_turns):
                    raise PipelineError("QWEN_INVALID_CONFIRMATION_EVIDENCE", 503, "qwen.confirmation")
                candidate = resolved
        candidates.append(candidate)
    return CandidateBatch(candidates=tuple(candidates))


class TextEngine(Protocol):
    def generate(self, call: CallInput) -> str: ...


class QwenBridge:
    def __init__(self, engine: TextEngine, *, temporal_resolver=None, confirmation_resolver=None):
        self.engine = engine
        self.temporal_resolver = temporal_resolver
        self.confirmation_resolver = confirmation_resolver

    def __call__(self, call, resources=None):
        try:
            raw = self.engine.generate(call)
        except PipelineError:
            raise
        except Exception as exc:
            raise PipelineError("QWEN_GENERATION_FAILED", 503, "qwen.generate", type(exc).__name__) from None
        return parse_candidates(raw, call, temporal_resolver=self.temporal_resolver,
                                confirmation_resolver=self.confirmation_resolver)
