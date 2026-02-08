from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, date
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class ForecastSession:
    session_id: Optional[int]
    session_timestamp: datetime

    project_name: str
    project_description: str

    file_name: str
    rows_count: int
    date_range_start: date
    date_range_end: date
    start_date: date
    horizon: int


@dataclass(frozen=True)
class ModelForecast:
    forecast_id: Optional[int]
    session_id: int
    model_type: str
    predictions: List[float]
    prediction_dates: List[str]
    metrics: Dict[str, Any]
