import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

DATA_PATH = "dataset.csv"
TARGET_COL = "new_cases"

WINDOW_SIZE = 7
VAL_FRACTION = 0.15
TEST_FRACTION = 0.15


df = pd.read_csv(DATA_PATH)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

df = df.dropna(subset=[TARGET_COL, "hosp_patients", "mobility"])

dates = df["date"].to_numpy()
y_true = df[TARGET_COL].to_numpy(dtype=np.float32)

n = len(df)
print(f"Всего наблюдений после очистки: {n}")


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

        S = S - new_exposed
        E = E + new_exposed - new_infectious
        I = I + new_infectious - new_recovered
        R = R + new_recovered

        new_cases.append(max(new_infectious, 0.0))

    return np.array(new_cases, dtype=np.float32)


N = 3.4e8
incubation_period = 5.0
infectious_period = 7.0

sigma = 1.0 / incubation_period
gamma = 1.0 / infectious_period

R0 = 2.5
beta = R0 * gamma

num_days = len(df)
seir_raw = simulate_seir(
    days=num_days,
    N=N,
    beta=beta,
    sigma=sigma,
    gamma=gamma,
    E0=10.0,
    I0=10.0,
    R0=0.0,
)


train_size = int(n * (1.0 - VAL_FRACTION - TEST_FRACTION))
val_size = int(n * VAL_FRACTION)
test_size = n - train_size - val_size

train_end_idx = train_size - 1
val_end_idx = train_size + val_size - 1

print(f"train: {train_size}, val: {val_size}, test: {test_size}")


eps = 1e-8
scale_factor = (
    y_true[: train_end_idx + 1].mean() + eps
) / (seir_raw[: train_end_idx + 1].mean() + eps)

seir_scaled = seir_raw * scale_factor
df["seir_baseline"] = seir_scaled


feature_cols = [TARGET_COL, "hosp_patients", "mobility", "seir_baseline"]

X_all = df[feature_cols].to_numpy(dtype=np.float32)
residual_all = (y_true - seir_scaled).reshape(-1, 1)

scaler_X = MinMaxScaler()
scaler_res = MinMaxScaler()

scaler_X.fit(X_all[: train_end_idx + 1])
scaler_res.fit(residual_all[: train_end_idx + 1])

X_scaled = scaler_X.transform(X_all)
res_scaled = scaler_res.transform(residual_all)


def make_sequences(X, y, window_size):
    X_seq, y_seq, idxs = [], [], []
    N = X.shape[0]

    for t in range(window_size, N):
        X_seq.append(X[t - window_size : t, :])
        y_seq.append(y[t])
        idxs.append(t)

    return (
        np.array(X_seq, dtype=np.float32),
        np.array(y_seq, dtype=np.float32),
        np.array(idxs, dtype=int),
    )

X_seq, y_seq, idxs = make_sequences(X_scaled, res_scaled, WINDOW_SIZE)

train_mask = idxs <= train_end_idx
val_mask = (idxs > train_end_idx) & (idxs <= val_end_idx)
test_mask = idxs > val_end_idx

X_train, y_train = X_seq[train_mask], y_seq[train_mask]
X_val, y_val = X_seq[val_mask], y_seq[val_mask]
X_test, y_test = X_seq[test_mask], y_seq[test_mask]
idxs_test = idxs[test_mask]

print(
    f"Train окон: {X_train.shape[0]}, val: {X_val.shape[0]}, test: {X_test.shape[0]}"
)

num_features = X_train.shape[2]


model = Sequential(
    [
        LSTM(64, return_sequences=True, input_shape=(WINDOW_SIZE, num_features)),
        Dropout(0.5),
        LSTM(32),
        Dense(16, activation="relu"),
        Dense(1),
    ]
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="mse",
    metrics=["mae"],
)

print("\nАрхитектура модели LSTM (остатки):")
model.summary()

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True,
)

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1,
)


res_pred_all_scaled = model.predict(X_seq)
res_pred_all = scaler_res.inverse_transform(res_pred_all_scaled).flatten()

seir_all = seir_scaled[idxs]
y_true_all = y_true[idxs]

y_pred_all = seir_all + res_pred_all
y_pred_all = np.clip(y_pred_all, a_min=0.0, a_max=None)

dates_all = dates[idxs]

res_pred_test_scaled = res_pred_all_scaled[test_mask]
res_pred_test = scaler_res.inverse_transform(res_pred_test_scaled).flatten()

seir_test = seir_scaled[idxs_test]
y_true_test = y_true[idxs_test]

y_pred_test = seir_test + res_pred_test
y_pred_test = np.clip(y_pred_test, a_min=0.0, a_max=None)

dates_test = dates[idxs_test]

print(
    "\nПервые 20 точек тестового прогнозирования (дата, real, seir, hybrid):"
)
for i in range(min(20, len(dates_test))):
    print(
        f"{pd.to_datetime(dates_test[i]).strftime('%Y-%m-%d')} | "
        f"real={y_true_test[i]:.1f} | "
        f"seir={seir_test[i]:.1f} | "
        f"hybrid={y_pred_test[i]:.1f}"
    )


plt.figure(figsize=(14, 6))
plt.plot(df["date"], y_true, label="Истинные значения (new_cases)", alpha=0.6)
plt.plot(dates_all, seir_all, label="SEIR-базлайн", alpha=0.6)
plt.plot(dates_all, y_pred_all, label="Гибрид SEIR+LSTM", linewidth=2)
plt.xlabel("Дата")
plt.ylabel("Число новых случаев")
plt.title("Гибридный прогноз новых случаев COVID-19 (весь датасет)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(dates_test, y_true_test, label="Истинные значения (new_cases)")
plt.plot(dates_test, seir_test, label="SEIR-базлайн", alpha=0.6)
plt.plot(dates_test, y_pred_test, label="Гибрид SEIR+LSTM", linewidth=2)
plt.xlabel("Дата")
plt.ylabel("Число новых случаев")
plt.title("Гибридный прогноз новых случаев COVID-19: тестовый отрезок")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 4))
plt.plot(history.history["loss"], label="train_loss")
plt.plot(history.history["val_loss"], label="val_loss")
plt.xlabel("Эпоха")
plt.ylabel("MSE")
plt.title("Динамика функции потерь LSTM (остатки)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()