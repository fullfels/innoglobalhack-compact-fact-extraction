import pytest
from src.contracts.models import CallInput, CandidateBatch, CandidateFact, Fact, ObjectRef, Source, Turn
from src.pipeline.normalizers import CatalogNormalizer, TemporalNormalizer, StateAndGroundingNormalizer


@pytest.fixture
def sample_catalog():
    return {
        "services": [
            {"name": "Домашний интернет", "aliases": ["интернет", "вайфай"]}
        ],
        "devices": [
            {"name": "Wi-Fi роутер", "aliases": ["роутер"]}
        ]
    }


def test_catalog_normalizer(sample_catalog):
    fact = Fact.model_construct(
        object=ObjectRef(name="вайфай"),
        attribute="service_state",
        value="не работает",
        polarity=1,
        speaker="client",
        valid_from="2026-03-30T10:00:00",
        source=Source(turn_idx=1, span=(0, 6), quote="вайфай")
    )
    batch = CandidateBatch.model_construct(
        candidates=(CandidateFact.model_construct(fact=fact, evidence_turns=(1,)),)
    )
    call = CallInput(
        call_id="c1",
        started_at="2026-03-30T10:00:00",
        turns=[Turn(idx=1, speaker="client", text="вайфай не работает")]
    )

    normalizer = CatalogNormalizer(sample_catalog)
    result_batch = normalizer.normalize(batch, call)

    assert result_batch.candidates[0].fact.object.name == "Домашний интернет"


def test_temporal_normalizer_relative_dates():
    fact = Fact.model_construct(
        object=ObjectRef(name="Интернет"),
        attribute="service_state",
        value="сломался",
        polarity=1,
        speaker="client",
        valid_from="вчера",
        valid_to="завтра",
        source=Source(turn_idx=1, span=(0, 5), quote="вчера")
    )
    batch = CandidateBatch.model_construct(
        candidates=(CandidateFact.model_construct(fact=fact, evidence_turns=(1,)),)
    )
    call = CallInput(
        call_id="c1",
        started_at="2026-03-30T10:00:00",
        turns=[Turn(idx=1, speaker="client", text="вчера сломался")]
    )

    normalizer = TemporalNormalizer()
    result_batch = normalizer.normalize(batch, call)

    updated_fact = result_batch.candidates[0].fact
    assert updated_fact.valid_from == "2026-03-29"
    assert updated_fact.valid_to == "2026-03-31"


def test_state_and_grounding_normalizer():
    turn_text = "У меня сломался интернет дома."
    fact = Fact.model_construct(
        object=ObjectRef(name="интернет"),
        attribute="service_state",
        value="сломался",
        polarity=1,
        speaker="client",
        valid_from="2026-03-30T10:00:00",
        source=Source.model_construct(turn_idx=1, span=(0, 0), quote="сломался интернет")
    )
    batch = CandidateBatch.model_construct(
        candidates=(CandidateFact.model_construct(fact=fact, evidence_turns=(1,)),)
    )
    call = CallInput(
        call_id="c1",
        started_at="2026-03-30T10:00:00",
        turns=[Turn(idx=1, speaker="client", text=turn_text)]
    )

    normalizer = StateAndGroundingNormalizer()
    result_batch = normalizer.normalize(batch, call)

    source = result_batch.candidates[0].fact.source
    expected_start = turn_text.index("сломался интернет")
    expected_end = expected_start + len("сломался интернет")

    assert source.span == (expected_start, expected_end)
