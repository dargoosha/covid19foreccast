from __future__ import annotations

from domain.repositories import ForecastRepository
from domain.entities import ForecastSession


def list_sessions_uc(repo: ForecastRepository, limit: int = 200) -> list[ForecastSession]:
    return repo.list_sessions(limit=limit)
