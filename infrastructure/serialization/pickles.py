from __future__ import annotations

import pickle
from typing import Any


def dumps(obj: Any) -> bytes:
    return pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)


def loads(blob: bytes) -> Any:
    return pickle.loads(blob)
