from datetime import datetime
import pytest

from src.contracts.models import CallInput, CandidateBatch, CandidateFact, Fact, ObjectRef, Source, Turn
from src.pipeline.adapters import ModelDescriptor
from src.pipeline.normalizers import (
    CatalogNormalizer,
    GroundingNormalizer,
    PolarityNormalizer,
    StateNormalizer,
    TemporalNormalizer,
)
from src.pipeline.service import Pipeline


def _make_call(turns_data: list[tuple[str, str]], started_at: str = "2026-05-19T17:00:00") -> CallInput:
    turns = tuple(Turn(idx=i, speaker=spk, text=txt) for i, (spk, txt) in enumerate(turns_data))
    return CallInput(call_id="test_call:0", started_at=datetime.fromisoformat(started_at), turns=turns)


def _make_cand(
    call: CallInput,
    *,
    obj_name: str = "Интернет",
    obj_type: str | None = "service",
    obj_id: str | None = None,
    attribute: str = "service_state",
    value: str = "не работает",
    polarity: int = 1,
    speaker: str = "client",
    valid_from: str | None = None,
    valid_to: str | None = None,
    turn_idx: int = 0,
    span: tuple[int, int] | None = None,
    quote: str | None = None,
    temporal_expression: str | None = None,
    confirmation: str = "unknown",
    evidence_turns: tuple[int, ...] = (),
) -> CandidateFact:
    vf = datetime.fromisoformat(valid_from) if valid_from else call.started_at
    vt = datetime.fromisoformat(valid_to) if valid_to else None
    fact = Fact(
        object=ObjectRef(name=obj_name, id=obj_id, type=obj_type),
        attribute=attribute,
        value=value,
        polarity=polarity,
        speaker=speaker,
        valid_from=vf,
        valid_to=vt,
        source=Source(turn_idx=turn_idx, span=span, quote=quote),
    )
    return CandidateFact(
        fact=fact,
        temporal_expression=temporal_expression,
        confirmation=confirmation,
        evidence_turns=evidence_turns,
    )


def test_catalog_normalizer_preserves_first_letter_and_separates_devices():
    call = _make_call([("client", "Мой раутер и пристафка не работают, тариф игровой.")])
    norm = CatalogNormalizer()

    c_router = _make_cand(call, obj_name="раутер", obj_type=None, value="не грузит")
    c_stb = _make_cand(call, obj_name="пристафка", obj_type=None, value="виснет")
    c_tariff = _make_cand(call, obj_name="Клиент", obj_type="client", attribute="tariff", value="игровой")
    c_broken = _make_cand(call, obj_name="u101_00020:ticket0", obj_id="u101_00020:ticket0", obj_type=None, attribute="ticket_status", value="создана")

    out = norm.normalize(CandidateBatch(candidates=(c_router, c_stb, c_tariff, c_broken)), call)

    # Первая буква сохраняется для case_metrics.py, устройства не склеиваются
    assert out.candidates[0].fact.object.name == "роутер"
    assert out.candidates[0].fact.object.type == "device"
    assert out.candidates[0].fact.value == "Отсутствие соединения"

    assert out.candidates[1].fact.object.name == "приставка"
    assert out.candidates[1].fact.object.type == "device"

    assert out.candidates[2].fact.value == "Тариф Игровой"

    assert out.candidates[3].fact.object.name == "заявка"
    assert out.candidates[3].fact.object.type == "ticket"
    assert out.candidates[3].fact.value == "принята"


@pytest.mark.parametrize(
    "started_at,expr,expected_from,expected_to",
    [
        ("2026-01-01T10:00:00", "вчера", "2025-12-31T10:00:00", None),
        ("2026-05-19T17:00:00", "позавчера", "2026-05-17T17:00:00", None),
        ("2026-05-19T17:00:00", "завтра", "2026-05-20T17:00:00", None),
        ("2026-05-19T17:00:00", "через неделю", "2026-05-26T17:00:00", None),
        ("2026-05-19T17:00:00", "со вторника", "2026-05-19T17:00:00", None),
        ("2026-05-19T17:00:00", "в четверг после двух", "2026-05-21T14:00:00", None),
        ("2026-05-19T17:00:00", "до пятницы", "2026-05-19T17:00:00", "2026-05-22T17:00:00"),
        ("2026-01-15T12:00:00", "с прошлого месяца", "2025-12-01T00:00:00", None),
    ],
)
def test_temporal_normalizer_dates_and_year_boundary(started_at, expr, expected_from, expected_to):
    call = _make_call([("client", "Проверка даты")], started_at=started_at)
    cand = _make_cand(call, temporal_expression=expr)
    out = TemporalNormalizer().normalize(CandidateBatch(candidates=(cand,)), call).candidates[0]

    assert out.fact.valid_from.isoformat() == expected_from
    assert (out.fact.valid_to.isoformat() if out.fact.valid_to else None) == expected_to
    assert out.fact.valid_from.tzinfo is None  # naive timezone сохранена


def test_temporal_normalizer_ambiguity_policy():
    call = _make_call([("client", "Это началось недавно")])
    cand = _make_cand(call, temporal_expression="недавно")
    out = TemporalNormalizer().normalize(CandidateBatch(candidates=(cand,)), call).candidates[0]

    assert out.fact.valid_from == call.started_at
    assert any("ambiguous" in w for w in out.warnings)


def test_polarity_and_confirmation_rules():
    call = _make_call(
        [
            ("client", "Интернет не работает со вчерашнего дня."),
            ("client", "Этой подпиской я больше не пользуюсь."),
            ("operator", "Подключить пакет гигабайтов?"),
            ("client", "Да нет, не надо."),
            ("operator", "Мастер приедет в четверг, подходит?"),
            ("client", "Да, подходит, оформляйте."),
        ]
    )

    c_symptom = _make_cand(call, value="не работает", polarity=-1, turn_idx=0)
    c_neg = _make_cand(call, obj_name="Подписка", value="больше не пользуюсь", polarity=1, turn_idx=1)
    c_rejected = _make_cand(call, attribute="agreement", value="пакет гигабайтов", speaker="operator", turn_idx=2, evidence_turns=(3,))
    c_confirmed = _make_cand(call, attribute="agreement", value="визит мастера в четверг", speaker="operator", turn_idx=4, evidence_turns=(5,))

    out = PolarityNormalizer().normalize(CandidateBatch(candidates=(c_symptom, c_neg, c_rejected, c_confirmed)), call)

    # Отрицательный симптом не превращается в polarity=-1
    assert out.candidates[0].fact.polarity == 1
    # Отрицание использования -> polarity=-1
    assert out.candidates[1].fact.polarity == -1
    # "Да нет" -> rejected
    assert out.candidates[2].confirmation == "rejected"
    # Подтверждённая договорённость -> confirmed
    assert out.candidates[3].confirmation == "confirmed"


def test_state_normalizer_transitions_and_multiobject_isolation():
    call = _make_call(
        [
            ("client", "Раньше был тариф Базовый, потом перешёл на Игровой."),
            ("client", "При этом домашний интернет работает нормально."),
        ]
    )
    t_old = _make_cand(
        call, obj_name="Клиент", obj_id="u1:client", obj_type="client",
        attribute="tariff", value="Базовый", valid_from="2026-01-01T00:00:00", turn_idx=0
    )
    t_new = _make_cand(
        call, obj_name="Клиент", obj_id="u1:client", obj_type="client",
        attribute="tariff", value="Игровой", valid_from="2026-05-01T00:00:00", turn_idx=0
    )
    svc_other = _make_cand(
        call, obj_name="Домашний интернет", obj_id="u1:svc0", obj_type="service",
        attribute="service_state", value="работает нормально", valid_from="2026-02-01T00:00:00", turn_idx=1
    )

    out = StateNormalizer().normalize(CandidateBatch(candidates=(t_old, t_new, svc_other)), call)

    # Старый тариф закрыт датой нового тарифа
    assert out.candidates[0].fact.valid_to == datetime.fromisoformat("2026-05-01T00:00:00")
    assert out.candidates[1].fact.valid_to is None
    # Независимый объект (Домашний интернет) не затронут сменой тарифа
    assert out.candidates[2].fact.valid_to is None


def test_grounding_unicode_codepoints_and_safe_fallback():
    text = "Привет 👋! У меня ёлка 🎄 и роутер не работает."
    target = "ёлка 🎄 и роутер"
    expected_start = text.index(target)
    expected_end = expected_start + len(target)

    call = _make_call([("client", text)])
    c_unicode = _make_cand(call, turn_idx=0, quote=target, span=None)
    c_missing = _make_cand(call, turn_idx=0, quote="несуществующая цитата в тексте", span=(0, 5))

    out = GroundingNormalizer().normalize(CandidateBatch(candidates=(c_unicode, c_missing)), call)

    assert out.candidates[0].fact.source.span == (expected_start, expected_end)
    assert out.candidates[0].fact.source.quote == target

    # Если цитата не найдена — координаты не выдумываются, оба поля сбрасываются в None
    assert out.candidates[1].fact.source.span is None
    assert out.candidates[1].fact.source.quote is None


def test_full_pipeline_e2e_filtering_and_validation():
    call = _make_call(
        [
            ("client", "Вчера перестал работать раутер."),
            ("operator", "Предлагаю аренду нового роутера."),
            ("client", "Да нет, не надо."),
            ("operator", "Оформлю выезд мастера в четверг после двух, подходит?"),
            ("client", "Да, подходит."),
        ]
    )

    class StubExtractor:
        descriptor = ModelDescriptor("stub-qwen", "real", "test")
        state = "ready"

        def extract(self, c: CallInput) -> CandidateBatch:
            return CandidateBatch(
                candidates=(
                    _make_cand(c, obj_name="раутер", obj_type=None, value="не работает", polarity=-1, turn_idx=0, quote="раутер", temporal_expression="вчера"),
                    _make_cand(c, attribute="agreement", value="аренда роутера", speaker="operator", turn_idx=1, evidence_turns=(2,)),
                    _make_cand(c, attribute="agreement", value="выезд мастера", speaker="operator", turn_idx=3, evidence_turns=(4,), temporal_expression="в четверг после двух"),
                )
            )

    prediction, warnings = Pipeline(StubExtractor()).run(call)
    assert len(prediction.facts) == 2
    assert prediction.facts[0].object.name == "роутер"
    assert prediction.facts[0].polarity == 1
    assert prediction.facts[0].valid_from == datetime.fromisoformat("2026-05-18T17:00:00")
    assert prediction.facts[1].attribute == "agreement"
    assert prediction.facts[1].valid_from == datetime.fromisoformat("2026-05-21T14:00:00")
    assert any("Filtered 1" in w for w in warnings)