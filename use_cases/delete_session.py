from __future__ import annotations

from domain.repositories import ForecastRepository


def delete_session_uc(repo: ForecastRepository, session_id: int) -> None:
    repo.delete_session(session_id)
