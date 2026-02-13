from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from core.config import MODEL_ORDER


COL_LABELS = {
    "day": "День",
    "date": "Дата",
    "pred_LSTM": "Прогноз LSTM",
    "pred_SEIR+LSTM old": "Прогноз старої SEIR + LSTM",
    "pred_SEIR+LSTM new": "Прогноз нової SEIR + LSTM",
}


def _get_dates_iso(obj: Any) -> list[str]:
    v = getattr(obj, "dates_iso", None)
    if v is not None:
        return [str(x) for x in list(v)]

    v = getattr(obj, "prediction_dates", None)
    if v is None:
        return []

    out: list[str] = []
    for d in list(v):
        if d is None:
            continue
        if hasattr(d, "isoformat"):
            out.append(d.isoformat())
        else:
            out.append(str(d))

    out = [pd.to_datetime(x).strftime("%Y-%m-%d") for x in out]
    return out


def _get_pred(obj: Any) -> list[float]:
    v = getattr(obj, "pred", None)
    if v is None:
        v = getattr(obj, "predictions", None)
    if v is None:
        return []
    return [float(x) for x in list(v)]


def render_forecast_table(
    chosen_date_iso: str,
    results: dict[str, Any],  # ForecastResult або ModelForecast
) -> None:
    base = None
    for k in MODEL_ORDER:
        if k in results:
            base = results[k]
            break

    if base is None:
        st.info("Немає прогнозу для відображення.")
        return

    dates_iso = _get_dates_iso(base)
    if len(dates_iso) == 0:
        st.info("Немає прогнозу (замало даних для обраної дати/горизонту).")
        return

    data: dict[str, list[Any]] = {
        "day": [0] + list(range(1, len(dates_iso) + 1)),
        "date": [chosen_date_iso] + list(dates_iso),
    }

    for mt in MODEL_ORDER:
        key = f"pred_{mt}"
        if mt in results:
            preds = _get_pred(results[mt])
            data[key] = [None] + preds[: len(dates_iso)]
        else:
            data[key] = [None] + [None] * len(dates_iso)

    df = pd.DataFrame(data).rename(columns=COL_LABELS)
    st.dataframe(df, width="stretch")
