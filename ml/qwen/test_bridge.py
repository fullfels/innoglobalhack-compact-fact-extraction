"""Independent synthetic fixtures; these tests do not claim model quality."""
import json
import pytest
from src.contracts.models import CallInput, CandidateFact
from src.pipeline.adapters import PipelineError
from ml.qwen.bridge import QwenBridge, model_input, parse_candidates


@pytest.fixture
def call():
    return CallInput.model_validate({"call_id": "synthetic", "started_at": "2026-11-14T10:00:00",
        "turns": [{"idx": 7, "speaker": "client", "text": "Интернет не работает 😞."},
                  {"idx": 21, "speaker": "operator", "text": "Завтра выезд специалиста. Подходит?"},
                  {"idx": 50, "speaker": "client", "text": "Нет, завтра не подходит."}]})


def fact():
    return {"object": {"name": "домашний интернет", "type": "service"}, "attribute": "service_state",
            "value": "не работает", "polarity": 1, "speaker": "client", "valid_from": "2026-11-14T10:00:00",
            "valid_to": None, "source": {"turn_idx": 7, "span": None, "quote": None}}


def payload(f):
    return json.dumps({"facts": [f]}, ensure_ascii=False)


def test_valid_multiple_facts_preserved(call):
    f, g = fact(), fact()
    g.update(value="периодические обрывы", valid_from="2026-11-13T00:00:00", valid_to="2026-11-14T10:00:00")
    assert len(parse_candidates(json.dumps({"facts": [g, f]}), call).candidates) == 2


def test_successful_empty_distinct_from_failed_generation(call):
    assert parse_candidates('{"facts":[]}', call).candidates == ()
    class BadEngine:
        def generate(self, _): raise RuntimeError("private raw text must not leak")
    with pytest.raises(PipelineError) as exc:
        QwenBridge(BadEngine())(call)
    assert exc.value.code == "QWEN_GENERATION_FAILED" and "private" not in str(exc.value)


@pytest.mark.parametrize("raw", ['{"facts":[],"facts":[]}', '{"facts":NaN}', '{"facts":1e400}', '{"facts":[', 'prefix {"facts":[]}'])
def test_malformed_duplicate_nonfinite_or_partial_json_fails(call, raw):
    with pytest.raises(PipelineError): parse_candidates(raw, call)


def test_speaker_mismatch_not_rewritten(call):
    f = fact(); f["source"]["turn_idx"] = 21
    with pytest.raises(PipelineError) as exc: parse_candidates(payload(f), call)
    assert exc.value.code == "QWEN_SPEAKER_SOURCE_MISMATCH"


def test_original_nonsequential_turn_indices_preserved(call):
    assert parse_candidates(payload(fact()), call).candidates[0].fact.source.turn_idx == 7
    assert [t["idx"] for t in model_input(call)["turns"]] == [7, 21, 50]


def test_unknown_date_never_falls_back_to_call_date(call):
    f = fact(); f["valid_from"] = None
    with pytest.raises(PipelineError) as exc: parse_candidates(payload(f), call)
    assert exc.value.code == "QWEN_UNRESOLVED_TIME"


def test_explicit_temporal_resolver_can_supply_grounded_date(call):
    f = fact(); f["valid_from"] = None
    candidate = parse_candidates(payload(f), call, temporal_resolver=lambda _f, _c: "2026-11-13T00:00:00").candidates[0]
    assert candidate.fact.valid_from.day == 13


def test_quote_is_grounded_in_same_original_turn_unicode_offsets(call):
    f = fact(); f["source"]["quote"] = "не работает 😞"
    source = parse_candidates(payload(f), call).candidates[0].fact.source
    assert call.turns[0].text[slice(*source.span)] == source.quote


def test_quote_in_other_turn_is_not_moved(call):
    f = fact(); f["source"]["quote"] = "выезд специалиста"
    with pytest.raises(PipelineError) as exc: parse_candidates(payload(f), call)
    assert exc.value.code == "QWEN_QUOTE_NOT_IN_SOURCE"


def test_client_turn_does_not_confirm_agreement(call):
    f = fact(); f.update(attribute="agreement", value="выезд специалиста", speaker="operator", valid_from="2026-11-15T00:00:00")
    f["source"]["turn_idx"] = 21
    c = parse_candidates(payload(f), call).candidates[0]
    assert c.confirmation == "unknown" and c.warnings


def test_unconfirmed_agreement_not_exported_by_shared_pipeline(call):
    from src.pipeline.adapters import InjectedAdapter, ModelResources
    from src.pipeline.service import Pipeline
    f = fact(); f.update(attribute="agreement", value="выезд специалиста", speaker="operator", valid_from="2026-11-15T00:00:00")
    f["source"]["turn_idx"] = 21
    adapter = InjectedAdapter(ModelResources("synthetic-only", None), lambda c, r: parse_candidates(payload(f), c))
    adapter.set_state("ready")
    prediction, warnings = Pipeline(adapter).run(call)
    assert prediction.facts == ()
    assert any("Filtered" in w for w in warnings)


@pytest.mark.parametrize("field,value", [("polarity", True), ("valid_from", 1), ("valid_to", "2026-11-13T00:00:00")])
def test_shared_contract_rejects_bad_types_intervals(call, field, value):
    f = fact(); f[field] = value
    with pytest.raises(PipelineError): parse_candidates(payload(f), call)


def test_input_allowlist_no_gold(call):
    assert set(model_input(call)) == {"call_id", "started_at", "turns"}
    with pytest.raises(Exception): CallInput.model_validate({**call.model_dump(), "facts": [fact()]})


def test_effective_prompt_view_is_idempotent_and_labels_unchanged():
    from ml.qwen.prompting import OLD, NEW, effective_example, effective_instruction
    row = {"prompt": [{"role": "system", "content": OLD}, {"role": "user", "content": "original conversation"}],
           "completion": [{"role": "assistant", "content": '{"facts":[]}'}]}
    view = effective_example(row)
    assert view["prompt"][0]["content"] == NEW and row["prompt"][0]["content"] == OLD
    assert view["completion"] == row["completion"] and view["prompt"][1] == row["prompt"][1]
    assert effective_instruction(NEW) == NEW
