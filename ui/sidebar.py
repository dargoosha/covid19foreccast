from __future__ import annotations

import streamlit as st

from use_cases.list_sessions import list_sessions_uc
from use_cases.delete_session import delete_session_uc


def _session_title(s) -> str:
    name = (s.project_name or "Без назви").strip()
    return name


def render_sidebar() -> None:
    repo = st.session_state["repo"]

    st.sidebar.markdown(
        """
    <style>
    /* ===== Tight sidebar layout (remove Streamlit block gaps) ===== */
    section[data-testid="stSidebar"] .block-container {
      padding-top: 0.75rem !important;
    }

    /* Streamlit adds margins around every element block */
    section[data-testid="stSidebar"] .element-container {
      margin: 0 !important;
      padding: 0 !important;
    }

    /* Some versions wrap with stVerticalBlock */
    section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
      gap: 0.25rem !important; /* ключове: прибирає величезний gap */
    }

    /* Columns add their own paddings */
    section[data-testid="stSidebar"] [data-testid="column"] {
      padding-top: 0 !important;
      padding-bottom: 0 !important;
    }

    /* Buttons – no extra margins */
    section[data-testid="stSidebar"] button {
      width: 100% !important;
      margin: 0 !important;
    }

    /* inner text ellipsis */
    section[data-testid="stSidebar"] button p,
    section[data-testid="stSidebar"] button span,
    section[data-testid="stSidebar"] button [data-testid="stMarkdownContainer"],
    section[data-testid="stSidebar"] button [data-testid="stMarkdownContainer"] * {
      white-space: nowrap !important;
      overflow: hidden !important;
      text-overflow: ellipsis !important;
      margin: 0 !important;
    }

    /* ===== Your row ===== */
    .sb-row {
      display: flex;
      gap: 8px;
      align-items: center;
      width: 100%;
      margin: 0 !important;
      padding: 0 !important;
    }

    /* left: main button */
    .sb-row .sb-main {
      flex: 1 1 auto;
      min-width: 0;
    }

    /* right: menu button */
    .sb-row .sb-menu {
      flex: 0 0 auto;
      width: 40px;
    }

    /* menu button style */
    section[data-testid="stSidebar"] div.sb-menu button {
      padding-left: 0 !important;
      padding-right: 0 !important;
      text-align: center !important;
      opacity: 0.85;
    }
    section[data-testid="stSidebar"] div.sb-menu button:hover {
      opacity: 1.0;
    }
    </style>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.markdown("### Історія прогнозів")

    sessions = list_sessions_uc(repo, limit=200)

    if st.sidebar.button("＋ Новий прогноз", width='stretch'):
        st.session_state["active_view"] = "new"
        st.session_state["selected_session_id"] = None
        st.rerun()

    st.sidebar.markdown("---")

    for s in sessions:
        is_selected = (st.session_state.get("selected_session_id") == s.session_id)
        label = _session_title(s)

        # --- layout row wrapper ---
        st.sidebar.markdown('<div class="sb-row">', unsafe_allow_html=True)

        col_main, col_menu = st.sidebar.columns([0.86, 0.14])

        with col_main:
            st.markdown('<div class="sb-main">', unsafe_allow_html=True)
            if st.button(
                label,
                key=f"open_{s.session_id}",
                width='stretch',
                type=("primary" if is_selected else "secondary"),
            ):
                st.session_state["active_view"] = "history"
                st.session_state["selected_session_id"] = s.session_id
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

        with col_menu:
            st.markdown('<div class="sb-menu">', unsafe_allow_html=True)
            # three-dots opens a popover with actions (delete)
            with st.popover("", width='stretch', icon=None):
                st.markdown(f"**{label}**")
                if st.button("Видалити", key=f"del_{s.session_id}", type="primary", width='stretch'):
                    delete_session_uc(repo, int(s.session_id))
                    if st.session_state.get("selected_session_id") == s.session_id:
                        st.session_state["selected_session_id"] = None
                        st.session_state["active_view"] = "new"
                    st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

        st.sidebar.markdown("</div>", unsafe_allow_html=True)
