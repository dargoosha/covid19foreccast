from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from core.config import MODEL_ORDER


def _get_pred(obj: Any) -> list[float]:
    v = getattr(obj, "pred", None)
    if v is None:
        v = getattr(obj, "predictions", None)
    if v is None:
        return []
    return [float(x) for x in list(v)]


def _get_dates(obj: Any) -> list[str]:
    v = getattr(obj, "dates_iso", None)
    if v is not None:
        return [pd.to_datetime(x).strftime("%Y-%m-%d") for x in list(v)]
    v = getattr(obj, "prediction_dates", None)
    if v is None:
        return []
    return [pd.to_datetime(x).strftime("%Y-%m-%d") for x in list(v)]


def _get_metrics(obj: Any) -> dict[str, Any]:
    m = getattr(obj, "metrics", None)
    if isinstance(m, dict):
        return m
    return {}


def _pick_best_model(results: dict[str, Any]) -> str | None:
    best = None
    best_score = float("inf")
    for mt in MODEL_ORDER:
        if mt not in results:
            continue
        metrics = _get_metrics(results[mt])

        def _val(key: str) -> float:
            try:
                x = float(metrics.get(key, float("inf")))
                if np.isnan(x) or np.isinf(x):
                    return float("inf")
                return x
            except Exception:
                return float("inf")

        score = _val("rmse_pct")
        if not np.isfinite(score):
            score = _val("mae_pct")
        if not np.isfinite(score):
            score = _val("mape")

        if score < best_score:
            best_score = score
            best = mt
    return best


def render_analytics_block(results: dict[str, Any]) -> None:
    st.markdown(
        """
<style>
.analytics-wrap {
  border: 1px solid rgba(49, 51, 63, 0.20);
  border-radius: 20px;
  padding: 20px 20px;
  background: rgba(255,255,255,0.05);
}

.analytics-title {
  font-weight: 900;
  font-size: 20px;
  margin-bottom: 16px;
  letter-spacing: -0.2px;
}

.analytics-section {
  margin-top: 18px;
}

.hero-card {
  border: 1px solid rgba(49, 51, 63, 0.18);
  border-radius: 18px;
  padding: 20px 20px;
  background: rgba(255, 255, 255, 0.07);
  margin-bottom: 14px;
}

.hero-head {
  display: flex;
  justify-content: space-between;
  align-items: flex-end;
  gap: 16px;
  margin-bottom: 10px;
}

.hero-label {
  font-weight: 900;
  font-size: 16px;
}

.hero-value {
  font-weight: 950;
  font-size: 30px;
  letter-spacing: -0.5px;
}

.hero-sub {
  font-size: 15px;
  opacity: 0.90;
  line-height: 1.5;
}

.small-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(260px, 1fr));
  gap: 16px;
  margin-top: 10px;
}

.small-card {
  border: 1px solid rgba(49, 51, 63, 0.16);
  border-radius: 16px;
  padding: 16px 16px;
  background: rgba(255, 255, 255, 0.06);
}

.small-title {
  font-weight: 900;
  font-size: 15px;
  margin-bottom: 12px;
  letter-spacing: -0.1px;
}

.small-kpi {
  display: grid;
  grid-template-columns: 1fr;
  gap: 10px;
}

.small-row {
  display: flex;
  justify-content: space-between;
  gap: 12px;
  font-size: 14px;
  opacity: 0.95;
}

.small-row b {
  font-size: 16px;
  font-weight: 900;
}

@media (max-width: 1100px) {
  .small-grid { grid-template-columns: repeat(2, minmax(260px, 1fr)); }
}

@media (max-width: 800px) {
  .small-grid { grid-template-columns: 1fr; }
}
</style>
        """,
        unsafe_allow_html=True,
    )

    sums: dict[str, float] = {}
    horizons: dict[str, int] = {}
    for mt in MODEL_ORDER:
        if mt not in results:
            continue
        preds = _get_pred(results[mt])
        if not preds:
            continue
        sums[mt] = float(np.sum(np.array(preds, dtype=np.float64)))
        horizons[mt] = int(len(preds))

    best = _pick_best_model(results)
    peak_date = None
    peak_vals: dict[str, float] = {}
    if best and best in results:
        best_preds = _get_pred(results[best])
        best_dates = _get_dates(results[best])
        if best_preds and best_dates:
            peak_idx = int(np.argmax(np.array(best_preds, dtype=np.float64)))
            peak_date = best_dates[min(peak_idx, len(best_dates) - 1)]
            for mt in MODEL_ORDER:
                if mt not in results:
                    continue
                preds = _get_pred(results[mt])
                if peak_idx < len(preds):
                    peak_vals[mt] = float(preds[peak_idx])

    parts: list[str] = []
    parts.append('<div class="analytics-wrap">')
    parts.append('<div class="analytics-title">Аналітичний підсумок</div>')

    # ---- Кумулятивні випадки ----
    parts.append('<div class="analytics-section">')
    if sums:
        mn = float(min(sums.values()))
        mx = float(max(sums.values()))
        h_any = int(next(iter(horizons.values())))

        parts.append(
            f"""
<div class="hero-card">
  <div class="hero-head">
    <div class="hero-label">Кумулятивні випадки за горизонт</div>
    <div class="hero-value">від {mn:,.0f} до {mx:,.0f}</div>
  </div>
  <div class="hero-sub">
    Очікувана сумарна кількість нових випадків за період H={h_any}.
  </div>
</div>
            """.replace(",", " ")
        )

        parts.append('<div class="small-grid">')
        for mt in MODEL_ORDER:
            if mt not in sums:
                continue
            total = float(sums[mt])
            h = int(horizons.get(mt, 0))
            parts.append(
                f"""
<div class="small-card">
  <div class="small-title">{mt}</div>
  <div class="small-kpi">
    <div class="small-row"><span>Сумарно за період</span><b>{total:,.0f}</b></div>
    <div class="small-row"><span>Горизонт</span><b>{h} днів</b></div>
  </div>
</div>
                """.replace(",", " ")
            )
        parts.append("</div>")
    else:
        parts.append(
            """
<div class="hero-card">
  <div class="hero-head">
    <div class="hero-label">Кумулятивні випадки</div>
    <div class="hero-value">—</div>
  </div>
  <div class="hero-sub">Недостатньо даних для розрахунку.</div>
</div>
            """
        )
    parts.append("</div>")

    # ---- Пікове навантаження ----
    parts.append('<div class="analytics-section">')
    if peak_date and peak_vals:
        mn = float(min(peak_vals.values()))
        mx = float(max(peak_vals.values()))

        parts.append(
            f"""
<div class="hero-card">
  <div class="hero-head">
    <div class="hero-label">Пікове навантаження на лікарні</div>
    <div class="hero-value">від {mn:,.0f} до {mx:,.0f}</div>
  </div>
  <div class="hero-sub">
    Дата піку: {peak_date}. Значення інших моделей наведені для цього ж дня,
    що дозволяє оцінити можливий розкид навантаження.
  </div>
</div>
            """.replace(",", " ")
        )

        parts.append('<div class="small-grid">')
        for mt in MODEL_ORDER:
            if mt not in peak_vals:
                continue
            v = float(peak_vals[mt])
            parts.append(
                f"""
<div class="small-card">
  <div class="small-title">{mt}</div>
  <div class="small-kpi">
    <div class="small-row"><span>Денний приріст у пік</span><b>{v:,.0f}</b></div>
    <div class="small-row"><span>Дата піку</span><b>{peak_date}</b></div>
  </div>
</div>
                """.replace(",", " ")
            )
        parts.append("</div>")
    else:
        parts.append(
            """
<div class="hero-card">
  <div class="hero-head">
    <div class="hero-label">Пікове навантаження</div>
    <div class="hero-value">—</div>
  </div>
  <div class="hero-sub">Недостатньо даних для визначення пікового дня.</div>
</div>
            """
        )
    parts.append("</div>")

    parts.append("</div>")

    st.markdown("\n".join(parts), unsafe_allow_html=True)
