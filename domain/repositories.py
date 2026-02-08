from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

from domain.entities import ForecastSession, ModelForecast


class ForecastRepository(ABC):
    @abstractmethod
    def init_schema(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def create_session(self, session: ForecastSession) -> int:
        raise NotImplementedError

    @abstractmethod
    def add_model_forecast(self, mf: ModelForecast) -> int:
        raise NotImplementedError

    @abstractmethod
    def list_sessions(self, limit: int = 200) -> List[ForecastSession]:
        raise NotImplementedError

    @abstractmethod
    def load_session(self, session_id: int) -> Tuple[ForecastSession, List[ModelForecast]]:
        raise NotImplementedError

    @abstractmethod
    def delete_session(self, session_id: int) -> None:
        raise NotImplementedError

    @abstractmethod
    def session_exists(self, session_id: int) -> bool:
        raise NotImplementedError
