from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import pandas as pd

from core.config import AppConfig
from infrastructure.ml.predictors import ThreeModelForecaster, ForecastResult


@dataclass(frozen=True)
class RunForecastInput:
    df: pd.DataFrame
    file_name: str
    start_date: pd.Timestamp
    horizon: int


@dataclass(frozen=True)
class RunForecastOutput:
    meta: Dict[str, Any]
    results: Dict[str, ForecastResult]


def run_forecast_uc(cfg: AppConfig, inp: RunForecastInput) -> RunForecastOutput:
    df = inp.df
    df = df.copy()

    n = len(df)
    start_idx = int(df.index[df[cfg.date_col] == inp.start_date][0])

    forecaster = ThreeModelForecaster(cfg)
    results = forecaster.forecast(df=df, start_idx=start_idx, horizon=int(inp.horizon))

    meta = {
        "file_name": inp.file_name,
        "rows_count": int(n),
        "date_range_start": pd.to_datetime(df[cfg.date_col].iloc[0]).date().isoformat(),
        "date_range_end": pd.to_datetime(df[cfg.date_col].iloc[-1]).date().isoformat(),
        "start_date": pd.to_datetime(inp.start_date).date().isoformat(),
        "horizon": int(inp.horizon),
        "window_size": int(cfg.window_size),
    }

    return RunForecastOutput(meta=meta, results=results)
