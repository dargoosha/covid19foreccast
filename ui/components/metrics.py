from __future__ import annotations

from typing import Any

import math
import streamlit as st

from core.config import MODEL_ORDER


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        x = float(v)
        if math.isnan(x) or math.isinf(x):
            return default
        return x
    except Exception:
        return default


def _overall_accuracy(metrics: dict[str, Any]) -> float:
    # "точність" як інверт середньої відсоткової помилки.
    # Використовуємо 3 стабільні метрики: MAPE, MAE%, RMSE%.
    # accuracy = clamp(100 - mean(errors), 0..100)
    mape = _safe_float(metrics.get("mape", 0.0), 0.0)
    mae_pct = _safe_float(metrics.get("mae_pct", 0.0), 0.0)
    rmse_pct = _safe_float(metrics.get("rmse_pct", 0.0), 0.0)

    avg_err = (mape + mae_pct + rmse_pct) / 3.0
    acc = 100.0 - avg_err
    if acc < 0:
        acc = 0.0
    if acc > 100:
        acc = 100.0
    return float(acc)


def render_metrics_block(results: dict[str, Any]) -> None:
    st.markdown(
        """
<style>
.metrics-wrap {
  border: 1px solid rgba(49, 51, 63, 0.16);
  border-radius: 18px;
  padding: 14px 14px;
  background: rgba(255,255,255,0.03);
}
.metrics-title {
  font-weight: 800;
  font-size: 14px;
  margin: 0 0 10px 0;
  opacity: 0.95;
}
.metrics-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(220px, 1fr));
  gap: 12px;
}
.metric-card {
  border: 1px solid rgba(49, 51, 63, 0.15);
  border-radius: 14px;
  padding: 12px 14px;
  background: rgba(255, 255, 255, 0.04);
}
.metric-head {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  gap: 10px;
  margin-bottom: 10px;
}
.metric-title {
  font-weight: 800;
  font-size: 14px;
  line-height: 1.15;
  overflow: hidden;
  white-space: nowrap;
  text-overflow: ellipsis;
  max-width: 100%;
}
.metric-badges {
  display: flex;
  gap: 6px;
  align-items: center;
  flex-wrap: wrap;
  justify-content: flex-end;
}
.metric-badge {
  font-size: 12px;
  padding: 3px 8px;
  border-radius: 999px;
  border: 1px solid rgba(49, 51, 63, 0.18);
  background: rgba(255, 255, 255, 0.06);
  white-space: nowrap;
}
.metric-body {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 10px;
}
.kpi {
  border-radius: 12px;
  padding: 10px 10px;
  border: 1px solid rgba(49, 51, 63, 0.10);
  background: rgba(255, 255, 255, 0.03);
}
.kpi-label {
  font-size: 12px;
  opacity: 0.75;
  margin-bottom: 4px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.kpi-value {
  font-weight: 900;
  font-size: 18px;
  letter-spacing: -0.2px;
}
@media (max-width: 1100px) {
  .metric-body { grid-template-columns: repeat(3, minmax(0, 1fr)); }
}
@media (max-width: 900px) {
  .metrics-grid { grid-template-columns: 1fr; }
  .metric-body { grid-template-columns: repeat(3, minmax(0, 1fr)); }
}
</style>
        """,
        unsafe_allow_html=True,
    )

    cards = ['<div class="metrics-wrap">', '<div class="metrics-title">Точність моделей</div>', '<div class="metrics-grid">']
    any_card = False

    for mt in MODEL_ORDER:
        if mt not in results:
            continue

        any_card = True
        obj = results[mt]
        metrics = getattr(obj, "metrics", None) or {}
        pred = getattr(obj, "pred", None)
        if pred is None:
            pred = getattr(obj, "predictions", None)
        horizon = len(pred) if pred is not None else None

        mape = _safe_float(metrics.get("mape", 0.0), 0.0)
        mae_pct = _safe_float(metrics.get("mae_pct", 0.0), 0.0)
        rmse_pct = _safe_float(metrics.get("rmse_pct", 0.0), 0.0)
        acc = _overall_accuracy(metrics)

        badge_h = f"H={horizon}" if horizon is not None else "з БД"
        badge_acc = f"Загальна точність: {acc:.1f}%"

        cards.append(
            f"""
<div class="metric-card">
  <div class="metric-head">
    <div class="metric-title">{mt}</div>
    <div class="metric-badges">
      <div class="metric-badge">{badge_acc}</div>
    </div>
  </div>
  <div class="metric-body">
    <div class="kpi">
      <div class="kpi-label">MAPE</div>
      <div class="kpi-value">{mape:.2f}%</div>
    </div>
    <div class="kpi">
      <div class="kpi-label">MAE%</div>
      <div class="kpi-value">{mae_pct:.2f}%</div>
    </div>
    <div class="kpi">
      <div class="kpi-label">RMSE%</div>
      <div class="kpi-value">{rmse_pct:.2f}%</div>
    </div>
  </div>
</div>
            """
        )

    cards.append("</div></div>")

    if not any_card:
        st.info("Немає метрик для відображення.")
        return

    st.markdown("\n".join(cards), unsafe_allow_html=True)
