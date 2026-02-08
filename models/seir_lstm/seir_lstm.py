# train_hybrid_model.py
from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


CFG = {
    # ---- IO ----
    "data_path": "D:\Projects\Python\Covid\dataset.csv",
    "date_col": "date",
    "target": "new_cases",
    "model_path": "seir_lstm.keras",

    # ---- Features ----
    # базові ознаки + SEIR baseline
    "feature_cols": ["new_cases", "hosp_patients", "mobility", "seir_baseline"],

    # ---- Split / window ----
    "window": 7,
    "val_frac": 0.15,
    "test_frac": 0.15,

    # ---- SEIR params (baseline) ----
    "N": 3.4e8,
    "incubation_period": 5.0,
    "infectious_period": 7.0,
    "R0": 2.5,
    "E0": 10.0,
    "I0": 10.0,
    "R0_init": 0.0,

    # ---- Model ----
    "lstm1_units": 64,
    "lstm2_units": 32,
    "dense_units": 16,
    "dropout": 0.5,
    "lr": 1e-3,

    # ---- Train ----
    "epochs": 100,
    "batch": 32,
    "early_stop_patience": 10,

    "seed": 42,
}

np.random.seed(CFG["seed"])
tf.random.set_seed(CFG["seed"])


def simulate_seir(
    days: int,
    N: float,
    beta: float,
    sigma: float,
    gamma: float,
    E0: float,
    I0: float,
    R0: float,
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


def make_sequences(X: np.ndarray, y: np.ndarray, w: int):
    Xs, ys = [], []
    for t in range(w, len(X)):
        Xs.append(X[t - w:t, :])
        ys.append(y[t])
    return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.float32)


def build_model(input_shape, cfg):
    model = Sequential([
        LSTM(cfg["lstm1_units"], return_sequences=True, input_shape=input_shape),
        Dropout(cfg["dropout"]),
        LSTM(cfg["lstm2_units"]),
        Dense(cfg["dense_units"], activation="relu"),
        Dense(1),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(cfg["lr"]),
        loss="mse",
        metrics=["mae"],
    )
    return model


def main() -> None:
    # ---- load / clean ----
    df = pd.read_csv(CFG["data_path"])
    df[CFG["date_col"]] = pd.to_datetime(df[CFG["date_col"]])
    df = df.sort_values(CFG["date_col"]).dropna(
        subset=[CFG["target"], "hosp_patients", "mobility"]
    ).reset_index(drop=True)

    y_true = df[CFG["target"]].to_numpy(np.float32)
    n = len(df)

    train_size = int(n * (1.0 - CFG["val_frac"] - CFG["test_frac"]))
    val_size = int(n * CFG["val_frac"])
    train_end_idx = train_size - 1  # inclusive

    # ---- seir baseline ----
    sigma = 1.0 / float(CFG["incubation_period"])
    gamma = 1.0 / float(CFG["infectious_period"])
    beta = float(CFG["R0"]) * gamma

    seir_raw = simulate_seir(
        days=n,
        N=float(CFG["N"]),
        beta=beta,
        sigma=sigma,
        gamma=gamma,
        E0=float(CFG["E0"]),
        I0=float(CFG["I0"]),
        R0=float(CFG["R0_init"]),
    )

    eps = 1e-8
    scale_factor = (y_true[:train_end_idx + 1].mean() + eps) / (seir_raw[:train_end_idx + 1].mean() + eps)
    seir_scaled = seir_raw * scale_factor
    df["seir_baseline"] = seir_scaled

    # ---- features + residual target ----
    X_all = df[CFG["feature_cols"]].to_numpy(np.float32)
    residual_all = (y_true - seir_scaled).reshape(-1, 1).astype(np.float32)

    # ---- scale ONLY on train (no leakage) ----
    scaler_X = MinMaxScaler()
    scaler_res = MinMaxScaler()

    scaler_X.fit(X_all[:train_end_idx + 1])
    scaler_res.fit(residual_all[:train_end_idx + 1])

    X_scaled = scaler_X.transform(X_all).astype(np.float32)
    res_scaled = scaler_res.transform(residual_all).astype(np.float32)

    # ---- windows ----
    X_seq, y_seq = make_sequences(X_scaled, res_scaled, CFG["window"])

    # ---- split windows by original index ----
    # window at position k predicts original index t = k + window
    idxs = np.arange(CFG["window"], n, dtype=int)

    val_end_idx = train_size + val_size - 1
    train_mask = idxs <= train_end_idx
    val_mask = (idxs > train_end_idx) & (idxs <= val_end_idx)

    X_train, y_train = X_seq[train_mask], y_seq[train_mask]
    X_val, y_val = X_seq[val_mask], y_seq[val_mask]

    # ---- train ----
    model = build_model((CFG["window"], X_train.shape[2]), CFG)
    es = EarlyStopping(patience=CFG["early_stop_patience"], restore_best_weights=True)

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=CFG["epochs"],
        batch_size=CFG["batch"],
        callbacks=[es],
        verbose=1,
    )

    # ---- save ----
    model.save(CFG["model_path"])


if __name__ == "__main__":
    main()
