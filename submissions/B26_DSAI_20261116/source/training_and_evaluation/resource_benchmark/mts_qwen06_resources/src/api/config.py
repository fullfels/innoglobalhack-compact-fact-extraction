import os
from dataclasses import dataclass

@dataclass(frozen=True)
class Settings:
    mode: str = 'mock'
    max_body_bytes: int = 262144
    model_id: str = 'synthetic-fixtures-v1'
    inference_timeout_sec: float | None = None

    def __post_init__(self):
        import math
        if self.inference_timeout_sec is not None and (not math.isfinite(self.inference_timeout_sec) or self.inference_timeout_sec <= 0):
            raise ValueError('timeout must be finite and positive')

    @classmethod
    def from_env(cls):
        mode = os.getenv('MWS_MODE', 'mock')
        if mode not in {'mock', 'real'}:
            raise ValueError('MWS_MODE must be mock or real')
        return cls(inference_timeout_sec=float(os.environ['MWS_INFERENCE_TIMEOUT_SEC']) if os.getenv('MWS_INFERENCE_TIMEOUT_SEC') else None,mode=mode, model_id=os.getenv('MWS_MODEL_ID', 'synthetic-fixtures-v1' if mode == 'mock' else 'pending-ml-artifacts'))
