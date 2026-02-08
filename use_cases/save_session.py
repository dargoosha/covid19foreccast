from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, date
from typing import Dict

from core.config import MODEL_ORDER
from domain.entities import ForecastSession, ModelForecast
from domain.repositories import ForecastRepository
from infrastructure.ml.predictors import ForecastResult


@dataclass(frozen=True)
class SaveSessionInput:
    project_name: str
    project_description: str

    file_name: str
    rows_count: int
    date_range_start: date
    date_range_end: date
    start_date: date
    horizon: int
    results: Dict[str, ForecastResult]


def save_session_uc(repo: ForecastRepository, inp: SaveSessionInput) -> int:
    session = ForecastSession(
        session_id=None,
        session_timestamp=datetime.now(),
        project_name=inp.project_name,
        project_description=inp.project_description,
        file_name=inp.file_name,
        rows_count=int(inp.rows_count),
        date_range_start=inp.date_range_start,
        date_range_end=inp.date_range_end,
        start_date=inp.start_date,
        horizon=int(inp.horizon),
    )
    session_id = repo.create_session(session)

    for model_type in MODEL_ORDER:
        if model_type not in inp.results:
            continue
        r = inp.results[model_type]
        mf = ModelForecast(
            forecast_id=None,
            session_id=session_id,
            model_type=model_type,
            predictions=[float(x) for x in r.pred],
            prediction_dates=[str(d) for d in r.dates_iso],
            metrics=dict(r.metrics),
        )
        repo.add_model_forecast(mf)

    return session_id
