from __future__ import annotations

import streamlit as st

from core.config import CFG
from infrastructure.db.sqlite_db import SQLiteDB
from infrastructure.repositories.sqlite_forecast_repository import SQLiteForecastRepository


def ensure_state() -> None:
    if "repo" not in st.session_state:
        db = SQLiteDB(CFG.sqlite_path)
        repo = SQLiteForecastRepository(db)
        repo.init_schema()
        st.session_state["repo"] = repo

    if "selected_session_id" not in st.session_state:
        st.session_state["selected_session_id"] = None

    if "active_view" not in st.session_state:
        # "new" або "history"
        st.session_state["active_view"] = "new"

    if "last_forecast" not in st.session_state:
        # кеш результату останнього запуску (для кнопки "Зберегти")
        st.session_state["last_forecast"] = None
