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
