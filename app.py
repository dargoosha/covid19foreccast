# app.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from config import SEIR_DEFAULTS, DEFAULT_LSTM_WINDOW
from db import init_db, save_time_series, load_time_series
from data_processing import load_csv_file, prepare_time_series
from experiments import run_experiment

# Ініціалізуємо БД
init_db()

st.set_page_config(
    page_title="Гібридна модель SEIR + LSTM для прогнозування COVID-19",
    layout="wide",
)

if "df_original" not in st.session_state:
    st.session_state.df_original = None
if "df_norm" not in st.session_state:
    st.session_state.df_norm = None
if "scalers" not in st.session_state:
    st.session_state.scalers = None
if "last_results" not in st.session_state:
    st.session_state.last_results = None
if "last_metrics" not in st.session_state:
    st.session_state.last_metrics = None

st.title("Модуль прогнозування перебігу епідемії COVID-19 (SEIR + LSTM)")

tab_data, tab_model, tab_vis = st.tabs(
    ["Дані та база даних", "Моделювання та прогноз", "Візуалізація експериментів"]
)

# ---------------------------------------------------------------------
# Вкладка "Дані та база даних"
# ---------------------------------------------------------------------
with tab_data:
    st.header("Завантаження даних та робота з локальною БД (SQLite)")

    st.markdown(
        """
        **Призначення цієї вкладки**

        1. Завантажити CSV-файл з історичними даними по США  
           (колонки `date`, `new_cases_smoothed`, `hosp_patients`, `mobility_avg` тощо).  
        2. Виконати попередню обробку (нормалізація, приведення до щоденного кроку).  
        3. Зберегти очищені часові ряди у локальну БД SQLite (`time_series`).  

        **База даних** використовується для:
        - зберігання очищених часових рядів (таблиця `time_series`);
        - зберігання інформації про експерименти та їх результати (`experiments`,
          `experiment_results`, `metrics`);
        - подальшої візуалізації та порівняння моделей.
        """
    )

    uploaded_file = st.file_uploader("Оберіть CSV-файл із локального комп'ютера", type=["csv"])

    if uploaded_file is not None:
        df_raw = load_csv_file(uploaded_file)
        st.subheader("Перші рядки завантажених даних")
        st.dataframe(df_raw.head())

        st.markdown(
            """
            У цій реалізації цільова змінна для прогнозування фіксована:
            **`new_cases_smoothed`** (згладжені нові випадки).  
            Вона автоматично перейменовується у внутрішній стовпець `new_cases`.
            """
        )

        if st.button("Виконати препроцесинг та зберегти в БД"):
            required_cols = ["date", "new_cases_smoothed"]
            for col in required_cols:
                if col not in df_raw.columns:
                    st.error(f"У файлі відсутня обов'язкова колонка `{col}`.")
                    st.stop()

            df_proc = pd.DataFrame()
            df_proc["date"] = df_raw["date"]
            df_proc["new_cases"] = df_raw["new_cases_smoothed"]

            # Госпіталізації
            if "hosp_patients" in df_raw.columns:
                df_proc["hospitalizations"] = df_raw["hosp_patients"]
            else:
                df_proc["hospitalizations"] = 0.0

            # Агрегований показник мобільності
            if "mobility_avg" in df_raw.columns:
                df_proc["mobility"] = df_raw["mobility_avg"]
            else:
                df_proc["mobility"] = 0.0

            df_proc = df_proc.ffill().bfill()

            save_time_series(df_proc, country="USA")

            df_norm, scalers = prepare_time_series(df_proc)

            st.session_state.df_original = df_proc
            st.session_state.df_norm = df_norm
            st.session_state.scalers = scalers

            st.success(
                "Дані успішно завантажено, оброблено та збережено в локальну БД SQLite."
            )

    st.markdown("---")
    st.subheader("Зчитування даних з БД")

    st.markdown(
        """
        Кнопка нижче читає останні збережені часові ряди США з локальної БД
        (таблиця `time_series`) та повторно виконує нормалізацію.

        Це зручно, якщо ви вже один раз завантажили CSV і хочете одразу перейти
        до моделювання без повторного завантаження файлу.
        """
    )

    if st.button("Завантажити останні дані з БД"):
        df_db = load_time_series(country="USA")
        if df_db.empty:
            st.warning("У базі поки немає даних. Спочатку завантажте CSV-файл.")
        else:
            st.session_state.df_original = df_db
            df_norm, scalers = prepare_time_series(df_db)
            st.session_state.df_norm = df_norm
            st.session_state.scalers = scalers
            st.success("Дані успішно зчитано з БД.")
            st.dataframe(df_db.head())


# ---------------------------------------------------------------------
# Вкладка "Моделювання та прогноз"
# ---------------------------------------------------------------------
with tab_model:
    st.header("Моделювання та прогноз")

    if st.session_state.df_norm is None:
        st.info("Спочатку завантажте та підготуйте дані на вкладці 'Дані та база даних'.")
    else:
        df_norm = st.session_state.df_norm
        df_original = st.session_state.df_original
        scalers = st.session_state.scalers

        col_left, col_right = st.columns(2)

        with col_left:
            st.subheader("Налаштування SEIR")
            beta = st.number_input("β (коефіцієнт передачі)", value=float(SEIR_DEFAULTS["beta"]), format="%.4f")
            sigma = st.number_input("σ (швидкість переходу E→I)", value=float(SEIR_DEFAULTS["sigma"]), format="%.4f")
            gamma = st.number_input("γ (швидкість одужання)", value=float(SEIR_DEFAULTS["gamma"]), format="%.4f")
            population = st.number_input("N (населення)", value=float(SEIR_DEFAULTS["population"]), step=1e6, format="%.0f")
            E0 = st.number_input("E₀ (початкова кількість експонованих)", value=float(SEIR_DEFAULTS["E0"]), format="%.0f")
            I0 = st.number_input("I₀ (початкова кількість інфікованих)", value=float(SEIR_DEFAULTS["I0"]), format="%.0f")

        with col_right:
            st.subheader("Налаштування LSTM")
            lstm_window = st.number_input("Довжина вікна L (днів)", min_value=5, max_value=60, value=DEFAULT_LSTM_WINDOW)
            lstm_units = st.number_input("Кількість нейронів у шарі LSTM", min_value=16, max_value=256, value=64, step=16)
            lstm_layers = st.number_input("Кількість LSTM-шарів", min_value=1, max_value=3, value=1)
            lstm_epochs = st.number_input("Кількість епох навчання", min_value=10, max_value=300, value=50, step=10)

        st.subheader("Параметри експерименту")
        model_type = st.radio(
            "Оберіть тип моделі",
            ["SEIR", "LSTM", "SEIR+LSTM"],
            index=2,
            help="SEIR – тільки механістична модель; LSTM – тільки нейронна; SEIR+LSTM – гібридна модель.",
        )
        horizon = st.selectbox("Горизонт прогнозування (днів)", options=[7, 14, 30], index=0)
        comment = st.text_input("Коментар до експерименту (необов'язково)", "")

        if st.button("Запустити прогноз"):
            seir_params = {
                "beta": beta,
                "sigma": sigma,
                "gamma": gamma,
                "population": population,
                "E0": E0,
                "I0": I0,
            }

            exp_id, results, metrics = run_experiment(
                df_norm=df_norm,
                df_original=df_original,
                scalers=scalers,
                model_type=model_type,
                seir_params=seir_params,
                horizon=int(horizon),
                lstm_window=int(lstm_window),
                lstm_units=int(lstm_units),
                lstm_layers=int(lstm_layers),
                lstm_epochs=int(lstm_epochs),
                comment=comment,
            )

            st.session_state.last_results = results
            st.session_state.last_metrics = metrics

            st.success(f"Прогноз успішно побудовано. ID експерименту: {exp_id}")

            st.subheader("Таблиця з фактичними та прогнозними значеннями")
            st.dataframe(results)

            st.subheader("Метрики точності (RMSE, MAE, MAPE)")
            for mtype, vals in metrics.items():
                st.markdown(
                    f"**{mtype}** – RMSE: `{vals['rmse']:.2f}`, "
                    f"MAE: `{vals['mae']:.2f}`, MAPE: `{vals['mape']:.2f}%`"
                )


# ---------------------------------------------------------------------
# Вкладка "Візуалізація експериментів"
# ---------------------------------------------------------------------
with tab_vis:
    st.header("Візуалізація експериментів")

    results = st.session_state.last_results
    metrics = st.session_state.last_metrics

    if results is None:
        st.info("Ще немає даних для візуалізації. Спочатку запустіть експеримент.")
    else:
        st.subheader("Графік часових рядів (фактичні vs прогнози)")
        fig, ax = plt.subplots(figsize=(10, 5))

        ax.plot(results["date"], results["actual"], label="Фактичні значення")
        if "seir_pred" in results.columns:
            ax.plot(results["date"], results["seir_pred"], label="SEIR прогноз")
        if "lstm_pred" in results.columns:
            ax.plot(results["date"], results["lstm_pred"], label="LSTM прогноз")
        if "hybrid_pred" in results.columns:
            ax.plot(results["date"], results["hybrid_pred"], label="SEIR+LSTM гібридний прогноз")

        ax.set_xlabel("Дата")
        ax.set_ylabel("Кількість випадків (умовні одиниці / масштабовані)")
        ax.legend()
        ax.grid(True)

        st.pyplot(fig)

        st.subheader("Метрики експерименту")
        if metrics:
            for mtype, vals in metrics.items():
                st.markdown(
                    f"- **{mtype}**: RMSE = `{vals['rmse']:.2f}`, "
                    f"MAE = `{vals['mae']:.2f}`, MAPE = `{vals['mape']:.2f}%`"
                )
        else:
            st.info("Метрики для цього експерименту ще не обчислені.")
