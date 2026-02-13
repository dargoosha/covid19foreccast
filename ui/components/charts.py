from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.config import MODEL_ORDER
from infrastructure.ml.predictors import ForecastResult


def render_chart(
    full_df: pd.DataFrame,  # columns: date, new_cases
    results: dict[str, ForecastResult],
    horizon: int,
) -> None:
    # ОРИГІНАЛЬНИЙ графік (як був): весь датасет + прогнози
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=full_df["date"],
            y=full_df["new_cases"],
            mode="lines",
            name="Реальні new_cases",
            opacity=0.6,
        )
    )

    base = None
    for k in MODEL_ORDER:
        if k in results:
            base = results[k]
            break

    if base is not None and len(base.dates_iso) > 0:
        x = [pd.to_datetime(d) for d in base.dates_iso]
        for mt in MODEL_ORDER:
            if mt not in results:
                continue
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=results[mt].pred,
                    mode="lines+markers",
                    name=f"{mt}",
                )
            )

    fig.update_layout(
        title="Порівняння прогнозів",
        xaxis_title="Дата",
        yaxis_title="Кількість нових випадків",
        hovermode="x unified",
    )
    fig.update_xaxes(
        rangeselector=dict(buttons=[dict(step="all", label="Весь період")]),
        rangeslider=dict(visible=True),
    )
    st.plotly_chart(fig, config={"responsive": True})


def _render_forecast_chart_window(
    df: pd.DataFrame,  # columns: date, new_cases
    results: dict[str, ForecastResult],
    chosen_date_iso: str,
    history_days: int = 30,
    show_true_history: bool = True,
) -> None:
    df2 = df.copy()
    df2["date"] = pd.to_datetime(df2["date"])
    df2 = df2.sort_values("date").reset_index(drop=True)

    chosen_dt = pd.to_datetime(chosen_date_iso)

    # last 30 days history up to chosen_dt (inclusive)
    left = chosen_dt - pd.Timedelta(days=int(history_days))
    hist = df2[(df2["date"] >= left) & (df2["date"] <= chosen_dt)]

    # true value at chosen_dt (anchor)
    row0 = df2[df2["date"] == chosen_dt]
    y0 = float(row0["new_cases"].iloc[0]) if len(row0) > 0 else None

    fig = go.Figure()

    if show_true_history:
        fig.add_trace(
            go.Scatter(
                x=hist["date"],
                y=hist["new_cases"],
                mode="lines",
                name=f"Дані датасету (останні {history_days} днів)",
                opacity=0.7,
            )
        )

    # choose base model for forecast x
    base = None
    for k in MODEL_ORDER:
        if k in results:
            base = results[k]
            break

    if base is not None and len(base.dates_iso) > 0:
        x_fore = [pd.to_datetime(d) for d in base.dates_iso]

        for mt in MODEL_ORDER:
            if mt not in results:
                continue

            y_fore = list(results[mt].pred)

            # prepend anchor point to connect with history
            if y0 is not None and len(x_fore) == len(y_fore):
                x_plot = [chosen_dt] + x_fore
                y_plot = [y0] + y_fore
            else:
                x_plot = x_fore
                y_plot = y_fore

            fig.add_trace(
                go.Scatter(
                    x=x_plot,
                    y=y_plot,
                    mode="lines+markers",
                    name=mt,
                )
            )

    fig.update_layout(
        title="Віконний графік",
        xaxis_title="Дата",
        yaxis_title="Кількість нових випадків",
        hovermode="x unified",
    )
    st.plotly_chart(fig, config={"responsive": True})



def render_new_forecast_charts_tabs(
    df_full: pd.DataFrame,  # columns: date, new_cases (весь датасет)
    results: dict[str, ForecastResult],
    chosen_date_iso: str,
    horizon: int,
) -> None:
    # Tab 1: як було (render_chart)
    # Tab 2: віконний режим (те, що ти зараз мав на першому табі) + додатковий таб без лінії датасету
    t1, t2 = st.tabs(["Графік", "Вікно прогнозу"])
    with t1:
        render_chart(full_df=df_full, results=results, horizon=horizon)

    with t2:
        s1, s2 = st.tabs(["З датасетом", "Лише прогнози"])
        with s1:
            _render_forecast_chart_window(
                df=df_full,
                results=results,
                chosen_date_iso=chosen_date_iso,
                history_days=30,
                show_true_history=True,
            )
        with s2:
            _render_forecast_chart_window(
                df=df_full,
                results=results,
                chosen_date_iso=chosen_date_iso,
                history_days=30,
                show_true_history=False,
            )
