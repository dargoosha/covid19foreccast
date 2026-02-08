from __future__ import annotations

from typing import Any

import streamlit as st

from core.config import MODEL_ORDER


def render_metrics_block(results: dict[str, Any]) -> None:
    st.markdown(
        """
<style>
.metrics-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(220px, 1fr));
  gap: 12px;
  margin-top: 6px;
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
  gap: 8px;
  margin-bottom: 8px;
}
.metric-title {
  font-weight: 700;
  font-size: 14px;
  line-height: 1.15;
  overflow: hidden;
  white-space: nowrap;
  text-overflow: ellipsis;
  max-width: 100%;
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
  font-weight: 800;
  font-size: 18px;
  letter-spacing: -0.2px;
}
@media (max-width: 900px) {
  .metrics-grid { grid-template-columns: 1fr; }
  .metric-body { grid-template-columns: 1fr 1fr 1fr; }
}
</style>
        """,
        unsafe_allow_html=True,
    )

    cards = ['<div class="metrics-grid">']
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

        mape = float(metrics.get("mape", 0.0))
        mae_pct = float(metrics.get("mae_pct", 0.0))
        rmse_pct = float(metrics.get("rmse_pct", 0.0))

        badge = f"H={horizon}" if horizon is not None else "з БД"

        cards.append(
            f"""
<div class="metric-card">
  <div class="metric-head">
    <div class="metric-title">{mt}</div>
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

    cards.append("</div>")

    if not any_card:
        st.info("Немає метрик для відображення.")
        return

    st.markdown("\n".join(cards), unsafe_allow_html=True)
