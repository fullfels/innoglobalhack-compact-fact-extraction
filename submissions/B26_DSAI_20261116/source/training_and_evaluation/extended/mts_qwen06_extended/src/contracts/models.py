"""Integration contract v1; metadata never enters model targets."""
from datetime import datetime
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator, model_validator

class Contract(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True, revalidate_instances='always')

class Turn(Contract):
    idx: int = Field(ge=0, strict=True)
    speaker: Literal['client', 'operator']
    text: str  # Official ASR turns may be empty; preserve idx and text verbatim.

class Series(Contract):
    account_id: str = Field(min_length=1)
    call_index: int = Field(ge=0, strict=True)
    series_len: int = Field(ge=1, strict=True)

class CallInput(Contract):
    call_id: str = Field(min_length=1, max_length=256)
    started_at: datetime
    duration_sec: int | None = Field(default=None, ge=0, strict=True)
    series: Series | None = None
    turns: tuple[Turn, ...] = Field(min_length=1, max_length=500)
    noise_level: float | None = Field(default=None, ge=0, le=1)

    @field_validator('started_at', mode='before')
    @classmethod
    def iso_date(cls, v):
        if not isinstance(v, (str, datetime)) or (isinstance(v, str) and 'T' not in v and ' ' not in v):
            raise ValueError('ISO datetime required')
        return v

    @model_validator(mode='after')
    def unique_turns(self):
        if len({t.idx for t in self.turns}) != len(self.turns):
            raise ValueError('duplicate turn idx')
        return self

class ObjectRef(Contract):
    name: str = Field(min_length=1)
    id: str | None = None
    type: Literal['service', 'device', 'ticket', 'client'] | None = None

class Source(Contract):
    turn_idx: int = Field(ge=0, strict=True)
    span: tuple[StrictInt, StrictInt] | None = None
    quote: str | None = None

class Fact(Contract):
    object: ObjectRef
    attribute: Literal['service_state', 'tariff', 'ticket_status', 'action', 'agreement']
    value: str = Field(min_length=1)
    polarity: Literal[-1, 1]
    speaker: Literal['client', 'operator']
    valid_from: datetime
    valid_to: datetime | None = None
    source: Source

    @field_validator('valid_from', 'valid_to', mode='before')
    @classmethod
    def iso_fact_date(cls, v):
        if v is None:
            return v
        return CallInput.iso_date(v)

    @field_validator('polarity', mode='before')
    @classmethod
    def integer_polarity(cls, v):
        if type(v) is not int:
            raise ValueError('polarity must be integer -1 or 1')
        return v

    @model_validator(mode='after')
    def interval(self):
        if self.valid_to is not None:
            try:
                if self.valid_to < self.valid_from:
                    raise ValueError('reversed interval')
            except TypeError:
                raise ValueError('inconsistent timezone')
        return self

class CandidateFact(Contract):
    fact: Fact
    object_key: str | None = None
    evidence_turns: tuple[int, ...] = ()
    temporal_expression: str | None = None
    confirmation: Literal['confirmed', 'rejected', 'unknown'] = 'unknown'
    provenance: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

class CandidateBatch(Contract):
    candidates: tuple[CandidateFact, ...]
    warnings: tuple[str, ...] = ()
    status: Literal['ok'] = 'ok'

class CallPrediction(CallInput):
    facts: tuple[Fact, ...]

class Meta(Contract):
    request_id: str
    model_id: str
    manifest_id: str | None = None
    mode: Literal['mock', 'real']
    elapsed_ms: float
    warnings: tuple[str, ...]
    schema_valid: bool = True
    is_mock: bool

class ExtractResponse(Contract):
    prediction: CallPrediction
    meta: Meta
