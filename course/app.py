import numpy as np
import pandas as pd

import streamlit as st

from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
from tensorflow.keras.models import load_model

import plotly.graph_objects as go


# ============================
#     ФУНКЦІЯ SEIR
# ============================
def simulate_seir(
    days: int,
    N: float,
    beta: float,
    sigma: float,
    gamma: float,
    E0: float = 10.0,
    I0: float = 10.0,
    R0: float = 0.0,
) -> np.ndarray:

    S = N - E0 - I0 - R0
    E = E0
    I = I0
    R = R0

    new_cases = []

    for _ in range(days):
        new_exposed = beta * S * I / N
        new_infectious = sigma * E
        new_recovered = gamma * I

        S -= new_exposed
        E += new_exposed - new_infectious
        I += new_infectious - new_recovered
        R += new_recovered

        new_cases.append(max(new_infectious, 0.0))

    return np.array(new_cases, dtype=np.float32)


# ============================
#  STREAMLIT UI
# ============================
st.set_page_config(page_title="SEIR+LSTM Hybrid Forecast", layout="wide")
st.title("Гібридна модель SEIR + LSTM")

st.markdown("""
Завантажте датасет (.csv) із колонками:
- **date**
- **new_cases**
- **hosp_patients**
- **mobility**
""")

# ============================
#   ЗАВАНТАЖЕННЯ ФАЙЛУ
# ============================
uploaded_file = st.file_uploader("Завантажте CSV датасет", type=["csv"])

if uploaded_file is None:
    st.warning("Завантажте датасет, щоб продовжити")
    st.stop()

# Читаємо CSV
df = pd.read_csv(uploaded_file)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

# ============================
# ПЕРЕВІРКА КОЛОНОК
# ============================
required_cols = {"date", "new_cases", "hosp_patients", "mobility"}
if not required_cols.issubset(df.columns):
    st.error(f"У датасеті відсутні потрібні колонки: {required_cols - set(df.columns)}")
    st.stop()

st.success("Датасет успішно завантажено!")

# Спочатку дропнем пропуски, як у тренувальному коді
df = df.dropna(subset=["new_cases", "hosp_patients", "mobility"]).reset_index(drop=True)

y_true = df["new_cases"].astype(float).to_numpy()
dates = df["date"].to_numpy()
n = len(df)

# ============================
# SEIR ПІДГОТОВКА ДАНИХ
# ============================
N_pop = 3.4e8
sigma = 1.0 / 5.0
gamma = 1.0 / 7.0
beta = 2.5 * gamma

seir_raw = simulate_seir(
    days=n,
    N=N_pop,
    beta=beta,
    sigma=sigma,
    gamma=gamma,
)

VAL_FRACTION = 0.15
TEST_FRACTION = 0.15

train_size = int(n * (1 - VAL_FRACTION - TEST_FRACTION))
val_size = int(n * VAL_FRACTION)

train_end_idx = train_size - 1
val_end_idx = train_size + val_size - 1

eps = 1e-8
scale_factor = (
    y_true[: train_end_idx + 1].mean() + eps
) / (seir_raw[: train_end_idx + 1].mean() + eps)

seir_scaled = seir_raw * scale_factor
df["seir_baseline"] = seir_scaled

feature_cols = ["new_cases", "hosp_patients", "mobility", "seir_baseline"]

X_all = df[feature_cols].to_numpy(dtype=np.float32)
residual_all = (y_true - seir_scaled).reshape(-1, 1)

scaler_X = MinMaxScaler()
scaler_res = MinMaxScaler()

scaler_X.fit(X_all[: train_end_idx + 1])
scaler_res.fit(residual_all[: train_end_idx + 1])

# ============================
# ЗАВАНТАЖЕННЯ МОДЕЛІ
# ============================
try:
    model = load_model("best_hybrid_model.keras")
    st.success("Модель best_hybrid_model.keras успішно завантажена!")
except Exception as e:
    st.error(f"Не можу завантажити модель best_hybrid_model.keras: {e}")
    st.stop()


# ============================
# ПРОГНОЗ НА ГОРИЗОНТ
# ============================
WINDOW_SIZE = 7

def forecast_horizon(start_idx, horizon):
    n_local = len(df)
    pred_dates = []
    pred_values = []

    for step in range(1, horizon + 1):
        t = start_idx + step
        if t >= n_local:
            break

        if t - WINDOW_SIZE < 0:
            continue

        X_window = X_all[t - WINDOW_SIZE : t, :]
        X_scaled = scaler_X.transform(X_window)
        X_scaled = X_scaled[np.newaxis, :, :]

        res_pred_scaled = model.predict(X_scaled, verbose=0)
        res_pred = scaler_res.inverse_transform(res_pred_scaled)[0, 0]
        y_pred = max(seir_scaled[t] + res_pred, 0.0)

        pred_dates.append(df["date"].iloc[t])
        pred_values.append(y_pred)

    return np.array(pred_dates), np.array(pred_values)


# ============================
# STREAMLIT ВИБІР ПАРАМЕТРІВ
# ============================
min_start = WINDOW_SIZE
max_start = n - 31  # виключаємо останні 30 днів

if max_start <= min_start:
    st.error("Замало даних, щоб виключити останні 30 днів і побудувати вікна.")
    st.stop()

valid_dates = df["date"].iloc[min_start:max_start]

col1, col2 = st.columns(2)

with col1:
    chosen_date = st.selectbox(
        "Оберіть дату для початку прогнозу:",
        options=valid_dates,
        format_func=lambda d: d.strftime("%Y-%m-%d"),
    )

with col2:
    horizon = st.selectbox("Горизонт прогнозу (днів):", [7, 14, 30])

start_idx = df.index[df["date"] == chosen_date][0]

pred_dates, pred_values = forecast_horizon(start_idx, horizon)

true_segment = (
    df.set_index("date")["new_cases"]
    .reindex(pred_dates)
    .to_numpy()
    if len(pred_dates) > 0
    else np.array([])
)

# ============================
# МЕТРИКИ
# ============================
if len(pred_values) > 0:
    mse = float(np.mean((true_segment - pred_values) ** 2))
    mae = float(np.mean(np.abs(true_segment - pred_values)))
else:
    mse = mae = float("nan")

st.subheader("Помилки прогнозу")
st.write(f"**MSE**: {mse:.2f}")
st.write(f"**MAE**: {mae:.2f}")

# ============================
# ІНТЕРАКТИВНИЙ ГРАФІК (PLOTLY)
# ============================
st.subheader("Графік new_cases та прогнозу (інтерактивний)")

fig = go.Figure()

# Реальний ряд
fig.add_trace(
    go.Scatter(
        x=df["date"],
        y=df["new_cases"],
        mode="lines",
        name="Реальні new_cases",
        opacity=0.6,
    )
)

# Прогноз
if len(pred_values) > 0:
    fig.add_trace(
        go.Scatter(
            x=pred_dates,
            y=pred_values,
            mode="lines+markers",
            name=f"Прогноз (+{horizon} днів)",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=pred_dates,
            y=true_segment,
            mode="lines",
            name="Реальні значення на горизонті",
            line=dict(dash="dash"),
        )
    )

fig.update_layout(
    title="Гібридний COVID-19 прогноз (SEIR+LSTM)",
    xaxis_title="Дата",
    yaxis_title="Кількість нових випадків",
    hovermode="x unified",
)

# Добавим кнопки диапазонов и ползунок по времени
fig.update_xaxes(
    rangeselector=dict(
        buttons=list([
            dict(count=7, label="7д", step="day", stepmode="backward"),
            dict(count=30, label="30д", step="day", stepmode="backward"),
            dict(count=90, label="90д", step="day", stepmode="backward"),
            dict(step="all", label="Весь період")
        ])
    ),
    rangeslider=dict(visible=True),
    type="date"
)

st.plotly_chart(fig, use_container_width=True)

st.markdown(
    f"""
**Обрана стартова дата:** {chosen_date.strftime('%Y-%m-%d')}  
**Фактична кількість кроків прогнозу:** {len(pred_values)}
"""
)
