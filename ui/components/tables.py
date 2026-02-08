from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from core.config import MODEL_ORDER


COL_LABELS = {
    "day": "День",
    "date": "Дата",
    "true": "Фактичні випадки",
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


def _build_true_map(true_df: pd.DataFrame) -> dict[str, float]:
    if true_df is None or len(true_df) == 0:
        return {}

    if "date" not in true_df.columns or "new_cases" not in true_df.columns:
        return {}

    tmp = true_df.copy()
    tmp["date"] = pd.to_datetime(tmp["date"]).dt.strftime("%Y-%m-%d")
    tmp = tmp.dropna(subset=["date"])
    # new_cases может быть NaN — это ок, просто отфильтруем
    tmp = tmp.dropna(subset=["new_cases"])
    return dict(zip(tmp["date"].tolist(), tmp["new_cases"].astype(float).tolist()))


def render_forecast_table(
    chosen_date_iso: str,
    results: dict[str, Any],          # ForecastResult або ModelForecast
    true_df: pd.DataFrame | None = None,  # optional
    show_true: bool = True,           # <--- NEW: выключает колонку факта
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

    true_map = _build_true_map(true_df) if show_true else {}

    data: dict[str, list[Any]] = {
        "day": [0] + list(range(1, len(dates_iso) + 1)),
        "date": [chosen_date_iso] + list(dates_iso),
    }

    if show_true:
        true_seg = [true_map.get(d) for d in dates_iso]
        data["true"] = [None] + true_seg

    for mt in MODEL_ORDER:
        key = f"pred_{mt}"
        if mt in results:
            preds = _get_pred(results[mt])
            data[key] = [None] + preds[: len(dates_iso)]
        else:
            data[key] = [None] + [None] * len(dates_iso)

    df = pd.DataFrame(data).rename(columns=COL_LABELS)
    st.dataframe(df, width='stretch')
