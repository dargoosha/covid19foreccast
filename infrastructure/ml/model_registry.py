from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass(frozen=True)
class ModelSpec:
    model_type: str           # "LSTM" / "SEIR+LSTM" / "ConvLSTM"
    path: str


class ModelRegistry:
    def __init__(self, mapping: Dict[str, str]) -> None:
        self._mapping = mapping

    def get(self, model_type: str) -> ModelSpec:
        if model_type not in self._mapping:
            raise KeyError(f"Model type not configured: {model_type}")
        return ModelSpec(model_type=model_type, path=self._mapping[model_type])

    def exists(self, model_type: str) -> bool:
        try:
            spec = self.get(model_type)
        except Exception:
            return False
        return os.path.exists(spec.path)
