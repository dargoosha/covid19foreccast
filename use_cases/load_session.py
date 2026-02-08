from __future__ import annotations

from domain.repositories import ForecastRepository
from domain.entities import ForecastSession, ModelForecast


def load_session_uc(repo: ForecastRepository, session_id: int) -> tuple[ForecastSession, list[ModelForecast]]:
    return repo.load_session(session_id)
