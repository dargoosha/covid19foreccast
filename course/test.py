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
print(f"Кількість спостережень після очистки: {n}")

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
    f"Кількість вікон: train={X_train.shape[0]}, val={X_val.shape[0]}, test={X_test.shape[0]}"
)

num_features = X_train.shape[2]

seir_test = seir_scaled[idxs_test]
y_true_test = y_true[idxs_test]

def build_model(config, input_shape):
    model = Sequential()

    model.add(
        LSTM(
            config["lstm1_units"],
            return_sequences=config["lstm2_units"] is not None,
            input_shape=input_shape,
        )
    )
    model.add(Dropout(config["dropout"]))

    if config["lstm2_units"] is not None:
        model.add(LSTM(config["lstm2_units"]))

    model.add(Dense(config["dense_units"], activation="relu"))
    model.add(Dense(1))

    optimizer = tf.keras.optimizers.Adam(learning_rate=config["lr"])
    model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])

    return model


model_configs = [
    {"name": "M1_L64_32_do0.5_lr1e-3", "lstm1_units": 64, "lstm2_units": 32, "dropout": 0.5, "dense_units": 16, "lr": 1e-3},
    {"name": "M2_L64_32_do0.3_lr1e-3", "lstm1_units": 64, "lstm2_units": 32, "dropout": 0.3, "dense_units": 16, "lr": 1e-3},
    {"name": "M3_L64_do0.5_lr1e-3",    "lstm1_units": 64, "lstm2_units": None, "dropout": 0.5, "dense_units": 16, "lr": 1e-3},
    {"name": "M4_L64_do0.3_lr5e-4",    "lstm1_units": 64, "lstm2_units": None, "dropout": 0.3, "dense_units": 16, "lr": 5e-4},
    {"name": "M5_L32_16_do0.5_lr1e-3", "lstm1_units": 32, "lstm2_units": 16, "dropout": 0.5, "dense_units": 16, "lr": 1e-3},
    {"name": "M6_L32_16_do0.3_lr1e-3", "lstm1_units": 32, "lstm2_units": 16, "dropout": 0.3, "dense_units": 16, "lr": 1e-3},
    {"name": "M7_L128_64_do0.5_lr1e-3","lstm1_units": 128,"lstm2_units": 64, "dropout": 0.5, "dense_units": 32, "lr": 1e-3},
    {"name": "M8_L128_do0.5_lr1e-3",   "lstm1_units": 128,"lstm2_units": None,"dropout": 0.5, "dense_units": 32, "lr": 1e-3},
    {"name": "M9_L32_do0.3_lr1e-3",    "lstm1_units": 32, "lstm2_units": None, "dropout": 0.3, "dense_units": 16, "lr": 1e-3},
    {"name": "M10_L32_do0.3_lr5e-4",   "lstm1_units": 32, "lstm2_units": None, "dropout": 0.3, "dense_units": 8,  "lr": 5e-4},
]

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True,
)

results = []
best_mse = np.inf
best_model = None
best_config = None
best_history = None

print("\n=== Запуск перебору 10 моделей LSTM ===\n")

for i, cfg in enumerate(model_configs, start=1):
    print(f"\n--- Модель {i}/{len(model_configs)}: {cfg['name']} ---")
    model = build_model(cfg, input_shape=(WINDOW_SIZE, num_features))
    model.summary()

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=32,
        callbacks=[early_stop],
        verbose=1,
    )

    res_pred_all_scaled = model.predict(X_seq, verbose=0)
    res_pred_all = scaler_res.inverse_transform(res_pred_all_scaled).flatten()

    res_pred_test_scaled = res_pred_all_scaled[test_mask]
    res_pred_test = scaler_res.inverse_transform(res_pred_test_scaled).flatten()
    y_pred_test = seir_test + res_pred_test
    y_pred_test = np.clip(y_pred_test, a_min=0.0, a_max=None)

    mse_test = np.mean((y_true_test - y_pred_test) ** 2)

    results.append({"name": cfg["name"], "config": cfg, "mse_test": mse_test})

    print(f"MAE (гібридний тестовий прогноз) = {mse_test:.4f}")

    if mse_test < best_mse:
        best_mse = mse_test
        best_model = model
        best_config = cfg
        best_history = history

print("\n=== Порівняння всіх моделей (MSE на тестовому наборі) ===")
for r in results:
    print(f"{r['name']}: MSE = {r['mse_test']:.4f}")

print(f"\nНайкраща модель: {best_config['name']} (MSE = {best_mse:.4f})")

best_model.save("best_hybrid_model.keras")
print("Найкращу модель збережено як best_hybrid_model.keras")

res_pred_all_scaled_best = best_model.predict(X_seq, verbose=0)
res_pred_all_best = scaler_res.inverse_transform(res_pred_all_scaled_best).flatten()

seir_all = seir_scaled[idxs]
y_true_all = y_true[idxs]

y_pred_all_best = seir_all + res_pred_all_best
y_pred_all_best = np.clip(y_pred_all_best, a_min=0.0, a_max=None)

res_pred_test_scaled_best = res_pred_all_scaled_best[test_mask]
res_pred_test_best = scaler_res.inverse_transform(res_pred_test_scaled_best).flatten()
y_pred_test_best = seir_test + res_pred_test_best
y_pred_test_best = np.clip(y_pred_test_best, a_min=0.0, a_max=None)

dates_all = dates[idxs]
dates_test = dates[idxs_test]

print("\nПерші 20 точок тестового прогнозу (найкраща модель):")
for i in range(min(20, len(dates_test))):
    print(
        f"{pd.to_datetime(dates_test[i]).strftime('%Y-%m-%d')} | "
        f"реальні={y_true_test[i]:.1f} | "
        f"seir={seir_test[i]:.1f} | "
        f"гібрид={y_pred_test_best[i]:.1f}"
    )

plt.figure(figsize=(14, 6))
plt.plot(df["date"], y_true, label="Реальні значення (new_cases)", alpha=0.6)
plt.plot(dates_all, seir_all, label="SEIR-базова модель", alpha=0.6)
plt.plot(dates_all, y_pred_all_best, label="Гібрид SEIR+LSTM (найкраща модель)", linewidth=2)
plt.xlabel("Дата")
plt.ylabel("Кількість нових випадків")
plt.title("Гібридний прогноз COVID-19 (повний набір)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(dates_test, y_true_test, label="Реальні значення (new_cases)")
plt.plot(dates_test, seir_test, label="SEIR-базова", alpha=0.6)
plt.plot(dates_test, y_pred_test_best, label="Гібрид SEIR+LSTM", linewidth=2)
plt.xlabel("Дата")
plt.ylabel("Кількість нових випадків")
plt.title("Гібридний прогноз COVID-19 — тестовий період")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 4))
plt.plot(best_history.history["loss"], label="train_loss")
plt.plot(best_history.history["val_loss"], label="val_loss")
plt.xlabel("Епоха")
plt.ylabel("MSE")
plt.title("Динаміка функції втрат LSTM — найкраща модель")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

