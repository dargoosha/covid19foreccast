from __future__ import annotations

from datetime import date as date_cls

import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go

from core.config import CFG, MODEL_ORDER
from core.errors import DatasetValidationError
from infrastructure.ml.preprocessing import load_and_validate_df, clean_core_cols
from use_cases.run_forecast import run_forecast_uc, RunForecastInput
from use_cases.save_session import save_session_uc, SaveSessionInput
from use_cases.load_session import load_session_uc
from ui.components.metrics import render_metrics_block
from ui.components.tables import render_forecast_table
from ui.components.charts import render_chart


def _save_dialog_and_persist() -> None:
    last = st.session_state.get("last_forecast")
    if not last:
        return

    @st.dialog("Збереження прогнозу")
    def _dlg() -> None:
        project_name = st.text_input("Назва проєкту", max_chars=15)

        if st.session_state.get("save_error"):
            st.error(st.session_state["save_error"])

        project_desc = st.text_area("Опис", max_chars=200, height=120)

        col1, col2 = st.columns(2)

        with col1:
            if st.button("Скасувати", width='stretch'):
                st.session_state["save_error"] = None
                st.session_state["show_save_dialog"] = False
                st.rerun()

        with col2:
            if st.button("Зберегти", type="primary", width='stretch'):
                name = (project_name or "").strip()
                desc = (project_desc or "").strip()

                if not name:
                    st.session_state["save_error"] = "Назва проєкту обовʼязкова."
                    st.session_state["show_save_dialog"] = True
                    st.rerun()
                    return

                # Пакування/збереження робимо ТУТ (після натискання), з лоадером у діалозі
                with st.spinner("Збереження..."):
                    meta = last["meta"]
                    session_id = save_session_uc(
                        st.session_state["repo"],
                        SaveSessionInput(
                            project_name=name,
                            project_description=desc,
                            file_name=str(meta["file_name"]),
                            rows_count=int(meta["rows_count"]),
                            date_range_start=date_cls.fromisoformat(str(meta["date_range_start"])),
                            date_range_end=date_cls.fromisoformat(str(meta["date_range_end"])),
                            start_date=date_cls.fromisoformat(str(meta["start_date"])),
                            horizon=int(meta["horizon"]),
                            results=last["results"],
                        ),
                    )

                st.session_state["selected_session_id"] = session_id
                st.session_state["active_view"] = "history"

                st.session_state["save_error"] = None
                st.session_state["show_save_dialog"] = False
                st.rerun()

    _dlg()


def _render_new_forecast() -> None:
    st.title("Прогноз COVID-19")

    st.markdown(
        """
Завантажте CSV з колонками: `date`, `new_cases`, `hosp_patients`, `mobility`.
"""
    )

    # ---------- state init ----------
    if "draft_df" not in st.session_state:
        st.session_state["draft_df"] = None
    if "draft_file_name" not in st.session_state:
        st.session_state["draft_file_name"] = None
    if "draft_chosen" not in st.session_state:
        st.session_state["draft_chosen"] = None
    if "draft_horizon" not in st.session_state:
        st.session_state["draft_horizon"] = 7
    if "run_trigger" not in st.session_state:
        st.session_state["run_trigger"] = False
    if "last_forecast" not in st.session_state:
        st.session_state["last_forecast"] = None
    if "last_forecast_key" not in st.session_state:
        st.session_state["last_forecast_key"] = None

    # ---------- upload ----------
    uploaded = st.file_uploader("CSV датасет", type=["csv"])
    if uploaded is None:
        st.info("Завантажте датасет.")
        return

    # читаємо/чистимо лише коли файл змінився
    current_name = getattr(uploaded, "name", "dataset.csv")
    if st.session_state["draft_file_name"] != current_name or st.session_state["draft_df"] is None:
        try:
            df_raw = pd.read_csv(uploaded)
            df = load_and_validate_df(df_raw, list(CFG.required_cols), CFG.date_col)
            df = clean_core_cols(df, ["new_cases", "hosp_patients", "mobility"])
            df = df.reset_index(drop=True)

        except DatasetValidationError as e:
            if e.missing_fields:
                st.error("Некоректний датасет.")
                st.markdown("**Відсутні обовʼязкові поля:**")
                st.code(", ".join(e.missing_fields))
            else:
                st.error(f"Некоректний датасет: {e}")
            st.stop()

        except Exception:
            st.error("Неможливо обробити файл. Перевірте формат CSV.")
            st.stop()

        st.session_state["draft_df"] = df
        st.session_state["draft_file_name"] = current_name

        # дефолтні параметри для вибору
        n = len(df)
        w = int(CFG.window_size)
        min_start = w
        max_start = max(w + 1, n - 31)
        valid_dates = df[CFG.date_col].iloc[min_start:max_start]

        if len(valid_dates) > 0:
            st.session_state["draft_chosen"] = pd.to_datetime(valid_dates.iloc[0])
        else:
            st.session_state["draft_chosen"] = None

        st.session_state["draft_horizon"] = 7
        st.session_state["run_trigger"] = False
        st.session_state["last_forecast"] = None
        st.session_state["last_forecast_key"] = None

    df = st.session_state["draft_df"]
    if df is None or len(df) == 0:
        st.error("Порожній датасет після очищення.")
        return

    st.success("Датасет завантажено та очищено. Налаштуйте параметри.")

    def _reset_run() -> None:
        st.session_state["run_trigger"] = False
        st.session_state["last_forecast"] = None
        st.session_state["last_forecast_key"] = None

    # ---------- controls ----------
    n = len(df)
    w = int(CFG.window_size)

    min_start = w
    max_start = max(w + 1, n - 31)
    valid_dates = df[CFG.date_col].iloc[min_start:max_start]
    if len(valid_dates) == 0:
        st.error("Замало даних для формування вікон та прогнозу.")
        return

    col1, col2, col3 = st.columns([1.2, 0.8, 1.0], vertical_alignment="bottom")

    with col1:
        min_date = pd.to_datetime(valid_dates.iloc[0]).date()
        max_date = pd.to_datetime(valid_dates.iloc[-1]).date()

        # текущее значение для календаря
        default_date = pd.to_datetime(st.session_state.get("draft_chosen") or valid_dates.iloc[0]).date()
        if default_date < min_date:
            default_date = min_date
        if default_date > max_date:
            default_date = max_date

        chosen_date = st.date_input(
            "Дата старту прогнозу:",
            value=default_date,
            min_value=min_date,
            max_value=max_date,
            key="ui_start_date_cal",
            on_change=_reset_run,
        )

        chosen = pd.Timestamp(chosen_date)

    with col2:
        horizon = st.selectbox(
            "Горизонт (днів):",
            [7, 14, 30],
            key="ui_horizon",
            on_change=_reset_run,
        )

    with col3:
        if st.button("Спрогнозувати", type="primary", width='stretch'):
            st.session_state["draft_chosen"] = pd.to_datetime(chosen)
            st.session_state["draft_horizon"] = int(horizon)
            st.session_state["run_trigger"] = True

    # ---- ключ прогноза (чтобы rerun не пересчитывал заново) ----
    if st.session_state.get("draft_chosen") is None:
        st.info("Оберіть дату старту прогнозу.")
        return

    forecast_key = (
        st.session_state["draft_file_name"] or "dataset.csv",
        pd.to_datetime(st.session_state["draft_chosen"]).date().isoformat(),
        int(st.session_state["draft_horizon"]),
    )

    last = st.session_state.get("last_forecast")
    last_key = st.session_state.get("last_forecast_key")

    need_run = bool(st.session_state.get("run_trigger", False)) and (last is None or last_key != forecast_key)

    if not need_run:
        if last is None:
            st.info("Натисніть «Спрогнозувати», щоб розпочати обчислення.")
            return
        # используем сохранённый результат
        meta = last["meta"]
        results = last["results"]
        chosen_date_iso = last["chosen_date_iso"]
        horizon_int = int(last["horizon"])
    else:
        with st.spinner("Виконується прогнозування..."):
            out = run_forecast_uc(
                CFG,
                RunForecastInput(
                    df=df,
                    file_name=st.session_state["draft_file_name"] or "dataset.csv",
                    start_date=pd.to_datetime(st.session_state["draft_chosen"]),
                    horizon=int(st.session_state["draft_horizon"]),
                ),
            )

        st.session_state["last_forecast"] = {
            "meta": out.meta,
            "results": out.results,
            "true_df": df[[CFG.date_col, CFG.target_col]].copy(),
            "chosen_date_iso": pd.to_datetime(st.session_state["draft_chosen"]).date().isoformat(),
            "horizon": int(st.session_state["draft_horizon"]),
        }
        st.session_state["last_forecast_key"] = forecast_key

        # критично: сбрасываем триггер, чтобы последующие rerun (например, из диалога) не пересчитывали прогноз
        st.session_state["run_trigger"] = False

        meta = out.meta
        results = out.results
        chosen_date_iso = pd.to_datetime(st.session_state["draft_chosen"]).date().isoformat()
        horizon_int = int(st.session_state["draft_horizon"])

    # ---------- render results ----------
    st.subheader("Помилки прогнозу")
    render_metrics_block(results)

    st.subheader("Таблиця прогнозу")
    render_forecast_table(
        chosen_date_iso=chosen_date_iso,
        true_df=df[[CFG.date_col, CFG.target_col]].rename(columns={CFG.target_col: "new_cases", CFG.date_col: "date"}),
        results=results,
    )

    st.subheader("Графік")
    render_chart(
        full_df=df[[CFG.date_col, CFG.target_col]].rename(columns={CFG.target_col: "new_cases", CFG.date_col: "date"}),
        results=results,
        horizon=horizon_int,
    )

    # ---------- save button ----------
    st.markdown("---")

    st.markdown(
        """
<style>
div[data-testid="stVerticalBlock"] div#st-key-save_to_history button {
  background: #16a34a !important;
  border-color: #16a34a !important;
  color: white !important;
}
div[data-testid="stVerticalBlock"] div#st-key-save_to_history button:hover {
  filter: brightness(0.95);
}
</style>
        """,
        unsafe_allow_html=True,
    )

    if "show_save_dialog" not in st.session_state:
        st.session_state["show_save_dialog"] = False
    if "save_error" not in st.session_state:
        st.session_state["save_error"] = None

    # Важно: НЕ делаем st.rerun() — иначе будет новый прогон страницы (и раньше он запускал forecast заново)
    if st.button("Зберегти цей прогноз в історію", key="save_to_history", type="secondary"):
        st.session_state["show_save_dialog"] = True
        st.session_state["save_error"] = None

    if st.session_state.get("show_save_dialog", False):
        _save_dialog_and_persist()


def _render_history_view() -> None:
    sid = st.session_state.get("selected_session_id")
    if sid is None:
        st.session_state["active_view"] = "new"
        st.rerun()

    repo = st.session_state["repo"]
    session, mfs = load_session_uc(repo, int(sid))

    st.title("Збережений прогноз")

    st.markdown(
        f"""
    **Сесія:** #{session.session_id}  
    **Проєкт:** {session.project_name}  
    **Опис:** {session.project_description if session.project_description else "—"}  
    **Час:** {session.session_timestamp.strftime("%Y-%m-%d %H:%M:%S")}  
    **Файл:** {session.file_name} (рядків: {session.rows_count})  
    **Діапазон дат:** {session.date_range_start.isoformat()} — {session.date_range_end.isoformat()}  
    **Старт:** {session.start_date.isoformat()}  
    **Горизонт:** {session.horizon}  
    """
    )

    # Метрики
    st.subheader("Метрики")
    by_type = {mf.model_type: mf for mf in mfs}
    render_metrics_block(by_type)

    # Таблиця
    st.subheader("Таблиця прогнозу")
    rows = []
    base_dates = None
    for mt in MODEL_ORDER:
        if mt in by_type:
            base_dates = list(by_type[mt].prediction_dates)
            break

    if base_dates:
        for i, d in enumerate(base_dates, start=1):
            row = {"day": i, "date": d}
            for mt in MODEL_ORDER:
                if mt in by_type and i - 1 < len(by_type[mt].predictions):
                    row[f"pred_{mt}"] = float(by_type[mt].predictions[i - 1])
            rows.append(row)

    if rows:
        render_forecast_table(
            chosen_date_iso=session.start_date.isoformat(),
            results=by_type,
            true_df=None,
            show_true=False,
        )
    else:
        st.info("Немає збережених прогнозних значень у цій сесії.")

    # Графік
    st.subheader("Графік прогнозів")
    if rows:
        fig = go.Figure()
        x = [r["date"] for r in rows]
        for mt in MODEL_ORDER:
            key = f"pred_{mt}"
            if key in rows[0]:
                fig.add_trace(go.Scatter(x=x, y=[r.get(key) for r in rows], mode="lines+markers", name=mt))
        fig.update_layout(
            title="Порівняння прогнозів",
            xaxis_title="Дата",
            yaxis_title="new_cases",
            hovermode="x unified",
        )
        st.plotly_chart(fig, config={"responsive": True})
    else:
        st.info("Немає даних для графіка.")


def render_forecast_page() -> None:
    view = st.session_state.get("active_view", "new")
    if view == "history":
        _render_history_view()
    else:
        _render_new_forecast()
