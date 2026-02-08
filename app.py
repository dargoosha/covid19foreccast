from __future__ import annotations

import streamlit as st

from ui.state import ensure_state
from ui.sidebar import render_sidebar
from ui.pages.forecast_page import render_forecast_page

st.set_page_config(page_title="COVID Forecast", layout="wide")

def main() -> None:
    ensure_state()
    render_sidebar()
    render_forecast_page()

if __name__ == "__main__":
    main()
