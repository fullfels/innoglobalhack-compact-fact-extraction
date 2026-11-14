import os
from dataclasses import dataclass

@dataclass(frozen=True)
class Settings:
    mode: str = 'mock'
    max_body_bytes: int = 262144
    model_id: str = 'synthetic-fixtures-v1'

    @classmethod
    def from_env(cls):
        mode = os.getenv('MWS_MODE', 'mock')
        if mode not in {'mock', 'real'}:
            raise ValueError('MWS_MODE must be mock or real')
        return cls(mode=mode, model_id=os.getenv('MWS_MODEL_ID', 'synthetic-fixtures-v1' if mode == 'mock' else 'pending-ml-artifacts'))
