from datetime import datetime, timezone, timedelta
from src.contracts.models import CallInput, CandidateBatch, CandidateFact, Fact, ObjectRef, Source, Turn
from src.pipeline.adapters import ModelDescriptor
from src.pipeline.normalizers import (
    CatalogNormalizer,
    GroundingNormalizer,
    PolarityNormalizer,
    StateNormalizer,
    TemporalNormalizer,
    build_default_normalizers,
)
from src.pipeline.service import Pipeline

STRESS_CATALOG = {
    "services": [
        {"name": "Домашний интернет", "aliases": ["интернет", "вай-фай", "wi-fi", "инет"]},
        {"name": "Цифровое ТВ", "aliases": ["телевидение", "тв"]},
    ],
    "devices": [
        {"name": "Wi-Fi роутер", "aliases": ["роутер", "маршрутизатор"]},
        {"name": "ТВ-приставка", "aliases": ["приставка", "смарт-приставка"]},
    ],
    "tariffs": [
        {"name": "Тариф Игровой", "aliases": ["игровой", "геймерский"]},
        {"name": "Тариф Базовый", "aliases": ["базовый", "простой"]},
    ],
    "issues": [
        {"name": "Отсутствие соединения", "aliases": ["не работает", "не грузит", "виснет", "пропал"]},
        {"name": "Низкая скорость", "aliases": ["тормозит", "медленно"]},
    ],
}


def _call(turns: list[tuple[str, str]], started_at: str = "2026-05-19T17:00:00") -> CallInput:
    return CallInput(
        call_id="stress:0",
        started_at=datetime.fromisoformat(started_at),
        turns=tuple(Turn(idx=i, speaker=s, text=t) for i, (s, t) in enumerate(turns)),
    )


def _cand(
    call: CallInput,
    *,
    name: str = "интернет",
    obj_type: str | None = "service",
    obj_id: str | None = None,
    attr: str = "service_state",
    val: str = "не работает",
    pol: int = 1,
    spk: str = "client",
    v_from: str | None = None,
    v_to: str | None = None,
    t_idx: int = 0,
    span: tuple[int, int] | None = None,
    quote: str | None = None,
    t_expr: str | None = None,
    conf: str = "unknown",
    ev: tuple[int, ...] = (),
) -> CandidateFact:
    return CandidateFact(
        fact=Fact(
            object=ObjectRef(name=name, id=obj_id, type=obj_type),
            attribute=attr,
            value=val,
            polarity=pol,
            speaker=spk,
            valid_from=datetime.fromisoformat(v_from) if v_from else call.started_at,
            valid_to=datetime.fromisoformat(v_to) if v_to else None,
            source=Source(turn_idx=t_idx, span=span, quote=quote),
        ),
        temporal_expression=t_expr,
        confirmation=conf,
        evidence_turns=ev,
    )


def test_stress_temporal_calendar_boundaries_and_timezones():
    # 1. Невисокосный февраль 2026: позавчера от 1 марта -> 27 февраля
    c_feb = _call([("client", "Позавчера пропал интернет")], started_at="2026-03-01T12:00:00")
    out_feb = TemporalNormalizer().normalize(
        CandidateBatch(candidates=(_cand(c_feb, t_expr="позавчера"),)), c_feb
    ).candidates[0]
    assert out_feb.fact.valid_from == datetime(2026, 2, 27, 12, 0, 0)

    # 2. Переход через Новый год: через два дня от 30 декабря 2026 -> 1 января 2027
    c_ny = _call([("operator", "Мастер приедет через два дня")], started_at="2026-12-30T15:00:00")
    out_ny = TemporalNormalizer().normalize(
        CandidateBatch(candidates=(_cand(c_ny, attr="agreement", val="выезд", t_expr="через два дня"),)), c_ny
    ).candidates[0]
    assert out_ny.fact.valid_from == datetime(2027, 1, 1, 15, 0, 0)

    # 3. Сохранение aware timezone (+03:00) без принудительного перевода в UTC
    c_tz = _call([("client", "Завтра перезвоню")], started_at="2026-05-19T17:00:00+03:00")
    out_tz = TemporalNormalizer().normalize(
        CandidateBatch(candidates=(_cand(c_tz, t_expr="завтра"),)), c_tz
    ).candidates[0]
    assert out_tz.fact.valid_from == datetime(2026, 5, 20, 17, 0, 0, tzinfo=timezone(timedelta(hours=3)))


def test_stress_asr_perturbations_polarity_and_confirmations():
    call = _call(
        [
            ("client", "У миня интирнет не грузит ваобще."),
            ("client", "А этой подпиской я больше не пользуюсь."),
            ("operator", "Могу подключить пакет гигабайтов."),
            ("client", "Да нит, не надо мне ничего."),
            ("operator", "Поставим визит в четверг после шести, подходит?"),
            ("client", "Да, потходит, жду."),
        ]
    )

    c1 = _cand(call, name="интирнет", obj_type=None, val="не грузит", pol=-1, t_idx=0)
    c2 = _cand(call, name="Подписка", val="больше не пользуюсь", pol=1, t_idx=1)
    c3 = _cand(call, attr="agreement", val="пакет гигабайтов", spk="operator", t_idx=2, ev=(3,))
    c4 = _cand(call, attr="agreement", val="визит мастера", spk="operator", t_idx=4, ev=(5,), t_expr="в четверг после шести")

    class Stub:
        descriptor = ModelDescriptor("stress-stub", "real", "test")
        state = "ready"

        def extract(self, c: CallInput) -> CandidateBatch:
            return CandidateBatch(candidates=(c1, c2, c3, c4))

    pred, warnings = Pipeline(Stub(), normalizers=build_default_normalizers(STRESS_CATALOG)).run(call)
    assert len(pred.facts) == 3
    assert pred.facts[0].object.name == "интернет"
    assert pred.facts[0].polarity == 1
    assert pred.facts[1].polarity == -1
    assert pred.facts[2].valid_from == datetime(2026, 5, 21, 18, 0, 0)
    assert any("Filtered 1" in w for w in warnings)


def test_stress_multi_step_ticket_lifecycle_and_device_isolation():
    call = _call(
        [
            ("operator", "Заявка 5000 была принята, потом в работе, сейчас выполнена."),
            ("client", "При этом раутер и пристафка работают нормально."),
        ]
    )
    t1 = _cand(call, name="заявка 5000", obj_type="ticket", obj_id="u1:ticket0", attr="ticket_status", val="создана", v_from="2026-05-10T00:00:00", t_idx=0)
    t2 = _cand(call, name="заявка 5000", obj_type="ticket", obj_id="u1:ticket0", attr="ticket_status", val="в работе", v_from="2026-05-12T00:00:00", t_idx=0)
    t3 = _cand(call, name="заявка 5000", obj_type="ticket", obj_id="u1:ticket0", attr="ticket_status", val="закрыта", v_from="2026-05-15T00:00:00", t_idx=0)
    d1 = _cand(call, name="раутер", obj_type="device", obj_id="u1:dev0", attr="service_state", val="работает нормально", v_from="2026-05-01T00:00:00", t_idx=1)
    d2 = _cand(call, name="пристафка", obj_type="device", obj_id="u1:dev1", attr="service_state", val="работает нормально", v_from="2026-05-01T00:00:00", t_idx=1)

    batch = CandidateBatch(candidates=(t1, t2, t3, d1, d2))
    batch = CatalogNormalizer(STRESS_CATALOG).normalize(batch, call)
    batch = StateNormalizer().normalize(batch, call)

    # Цепочка статусов заявки корректно закрыта по интервалам
    assert batch.candidates[0].fact.value == "принята"
    assert batch.candidates[0].fact.valid_to == datetime(2026, 5, 12, 0, 0, 0)
    assert batch.candidates[1].fact.valid_to == datetime(2026, 5, 15, 0, 0, 0)
    assert batch.candidates[2].fact.value == "выполнена"
    assert batch.candidates[2].fact.valid_to is None

    # Устройства не смешаны и не закрыты
    assert batch.candidates[3].fact.object.name == "роутер"
    assert batch.candidates[3].fact.valid_to is None
    assert batch.candidates[4].fact.object.name == "приставка"
    assert batch.candidates[4].fact.valid_to is None


def test_stress_unicode_zwj_and_combining_codepoints():
    # Многосимвольный ZWJ-эмодзи + комбинируемая диакритика и\u0306
    text = "Семья 👨‍👩‍👧‍👦 смотрит кино, но сбои\u0306 на роутере."
    quote = "сбои\u0306 на роутере"
    expected_start = text.index(quote)
    expected_end = expected_start + len(quote)

    call = _call([( "client", text)])
    cand = _cand(call, t_idx=0, quote=quote, span=None)
    out = GroundingNormalizer().normalize(CandidateBatch(candidates=(cand,)), call).candidates[0]

    assert out.fact.source.span == (expected_start, expected_end)
    assert text[out.fact.source.span[0]:out.fact.source.span[1]] == quote