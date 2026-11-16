"""Engine-independent handoff, not a guessed Qwen/Transformers signature."""
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Protocol, Literal
from src.contracts.models import CallInput, CandidateBatch, CandidateFact, Fact, ObjectRef, Source

class PipelineError(Exception):
    def __init__(self, code: str, status: int = 503, module: str = 'extractor', error_type: str | None = None):
        self.code, self.status, self.module = code, status, module
        self.error_type = error_type

@dataclass(frozen=True)
class ModelDescriptor:
    model_id: str
    mode: Literal['mock', 'real']
    engine: str = 'custom'

    def __post_init__(self):
        if self.mode not in {'mock', 'real'} or not self.model_id:
            raise ValueError('Invalid model descriptor')

class Extractor(Protocol):
    descriptor: ModelDescriptor
    state: Literal['unloaded', 'loading', 'ready', 'failed']

    def extract(self, call: CallInput) -> CandidateBatch: ...

class Normalizer(Protocol):
    def normalize(self, batch: CandidateBatch, call: CallInput) -> CandidateBatch: ...

@dataclass(frozen=True)
class ModelResources:
    model_path: str
    tokenizer: Any
    inference_config: Mapping[str, Any] = field(default_factory=dict)

class InjectedAdapter:
    """ML supplies a bridge converting its actual inference output to CandidateBatch."""
    def __init__(self, resources: ModelResources, inference: Callable[[CallInput, ModelResources], CandidateBatch], *, descriptor: ModelDescriptor | None = None):
        self.resources, self.inference = resources, inference
        self.descriptor = descriptor or ModelDescriptor("injected-model", "real")
        if self.descriptor.mode != "real":
            raise ValueError("InjectedAdapter requires a real descriptor")
        self.state = "unloaded"

    def set_state(self, state: str):
        if state not in {"unloaded", "loading", "ready", "failed"}:
            raise ValueError("Invalid model state")
        self.state = state

    def extract(self, call: CallInput) -> CandidateBatch:
        return CandidateBatch.model_validate(self.inference(call, self.resources))

class PendingRealAdapter:
    descriptor = ModelDescriptor("pending-ml-artifacts", "real", "pending")
    state = "unloaded"
    def extract(self, call: CallInput) -> CandidateBatch:
        raise PipelineError('MODEL_ARTIFACTS_NOT_CONNECTED')

# Synthetic, explicit fixtures: no dataset, gold, heuristic extraction or hidden fallback.
MOCK_LINES = {
    'Интернет не работает.': ('Интернет', 'service_state', 'не работает', 1, 'client'),
    'Я не использую роутер.': ('Роутер', 'service_state', 'использую', -1, 'client'),
    'Заявка зарегистрирована.': ('Заявка', 'ticket_status', 'зарегистрирована', 1, 'operator'),
}

class MockExtractor:
    descriptor = ModelDescriptor("synthetic-fixtures-v1", "mock", "fixture-lookup")
    state = "ready"
    def extract(self, call: CallInput) -> CandidateBatch:
        if any(t.text not in MOCK_LINES or t.speaker != MOCK_LINES[t.text][4] for t in call.turns):
            raise PipelineError('MOCK_EXAMPLE_NOT_SUPPORTED', 422)
        candidates = []
        for t in call.turns:
            name, attr, value, polarity, _ = MOCK_LINES[t.text]
            fact = Fact(object=ObjectRef(name=name), attribute=attr, value=value,
                        polarity=polarity, speaker=t.speaker, valid_from=call.started_at,
                        source=Source(turn_idx=t.idx, span=(0, len(t.text)), quote=t.text))
            candidates.append(CandidateFact(fact=fact, evidence_turns=(t.idx,), provenance=('synthetic-mock-v1',)))
        return CandidateBatch(candidates=tuple(candidates), warnings=('MOCK: synthetic fixture lookup; no trained model',))
