from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D, AveragePooling1D
from tensorflow.keras.callbacks import EarlyStopping


# =========================
# ПАРАМЕТРИ
# =========================
PARAMS = {
    "data_path": r"D:\Projects\Python\Covid\dataset.csv",
    "date_col": "date",
    "target_col": "new_cases",

    "base_feature_cols": ["new_cases", "hosp_patients", "mobility"],

    "window_size": 7,
    "val_fraction": 0.15,
    "test_fraction": 0.15,

    "lstm1_units": 64,
    "lstm2_units": 32,
    "dropout": 0.5,
    "dense_units": 16,
    "lr": 1e-3,

    "epochs": 100,
    "batch_size": 32,
    "early_stop_patience": 10,

    "population_col": "population",
    "population_fallback": 1_000_000,

    "E0_factor": 3.0,
    "I0_days": 7,
    "R0_init": 0.0,

    "sigma": 1 / 5.2,
    "beta_grid": (0.05, 1.20, 60),
    "gamma_grid": (0.03, 0.50, 50),
    "fit_target": "sigmaE",

    "model_save_path": "dyn_seir_lstm_clean.keras",
    "seed": 42,
}

np.random.seed(PARAMS["seed"])
tf.random.set_seed(PARAMS["seed"])


# =========================
# UTILS
# =========================
def make_sequences(X, y, w):
    Xs, ys = [], []
    for t in range(w, len(X)):
        Xs.append(X[t - w:t])
        ys.append(y[t])
    return np.array(Xs, np.float32), np.array(ys, np.float32)


def seir_simulate(new_cases, N, beta, sigma, gamma, E0_factor, I0_days, R0_init):
    T = len(new_cases)
    I0 = np.sum(new_cases[:max(1, I0_days)])
    E0 = E0_factor * I0
    S0 = max(N - E0 - I0 - R0_init, 0.0)

    S = np.zeros(T)
    E = np.zeros(T)
    I = np.zeros(T)
    R = np.zeros(T)

    S[0], E[0], I[0], R[0] = S0, E0, I0, R0_init

    for t in range(1, T):
        inf = beta * S[t - 1] * I[t - 1] / N
        inc = sigma * E[t - 1]
        rec = gamma * I[t - 1]

        S[t] = max(S[t - 1] - inf, 0.0)
        E[t] = max(E[t - 1] + inf - inc, 0.0)
        I[t] = max(I[t - 1] + inc - rec, 0.0)
        R[t] = max(R[t - 1] + rec, 0.0)

    return {
        "S": S,
        "E": E,
        "I": I,
        "R": R,
        "sigmaE": sigma * E,
    }


def fit_seir_on_train(y_train, N, cfg):
    betas = np.linspace(*cfg["beta_grid"])
    gammas = np.linspace(*cfg["gamma_grid"])

    best = (np.inf, None, None)

    for beta in betas:
        for gamma in gammas:
            sim = seir_simulate(
                y_train, N, beta, cfg["sigma"], gamma,
                cfg["E0_factor"], cfg["I0_days"], cfg["R0_init"]
            )
            loss = np.mean((y_train - sim[cfg["fit_target"]]) ** 2)
            if loss < best[0]:
                best = (loss, beta, gamma)

    return best[1], best[2]


def build_model(input_shape, cfg):
    model = Sequential([
        # # ---- very weak local smoothing ----
        # Conv1D(
        #     filters=8,          # слабко
        #     kernel_size=3,      # лише локально
        #     padding="same",
        #     activation="relu",
        #     input_shape=input_shape,
        # ),
        # MaxPooling1D(
        #     pool_size=2,
        #     strides=1,          # без грубого прорідження
        #     padding="same",
        # ),

        # ---- LSTM stack ----
        LSTM(cfg["lstm1_units"], return_sequences=True),
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


# =========================
# MAIN (TRAIN ONLY)
# =========================
def main():
    df = pd.read_csv(PARAMS["data_path"])
    df[PARAMS["date_col"]] = pd.to_datetime(df[PARAMS["date_col"]])
    df = df.sort_values(PARAMS["date_col"]).dropna(
        subset=PARAMS["base_feature_cols"] + [PARAMS["target_col"]]
    )

    y = df[PARAMS["target_col"]].to_numpy(np.float32)
    X_base = df[PARAMS["base_feature_cols"]].to_numpy(np.float32)

    n = len(df)
    train_end = int(n * (1 - PARAMS["val_fraction"] - PARAMS["test_fraction"]))

    if PARAMS["population_col"] in df.columns:
        N = float(df[PARAMS["population_col"]].iloc[0])
    else:
        N = float(PARAMS["population_fallback"])

    beta, gamma = fit_seir_on_train(y[:train_end], N, PARAMS)

    sim = seir_simulate(
        y, N, beta, PARAMS["sigma"], gamma,
        PARAMS["E0_factor"], PARAMS["I0_days"], PARAMS["R0_init"]
    )

    X_seir = np.column_stack([
        sim["S"] / N,
        sim["E"] / N,
        sim["I"] / N,
        sim["R"] / N,
        sim["sigmaE"],
    ])

    X_all = np.column_stack([X_base, X_seir])
    y_all = y.reshape(-1, 1)

    scaler_X = MinMaxScaler().fit(X_all)
    scaler_y = MinMaxScaler().fit(y_all)

    X_scaled = scaler_X.transform(X_all)
    y_scaled = scaler_y.transform(y_all)

    X_seq, y_seq = make_sequences(X_scaled, y_scaled, PARAMS["window_size"])

    split = int(len(X_seq) * 0.85)
    X_train, X_val = X_seq[:split], X_seq[split:]
    y_train, y_val = y_seq[:split], y_seq[split:]

    model = build_model(
        (PARAMS["window_size"], X_train.shape[2]),
        PARAMS,
    )

    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=PARAMS["epochs"],
        batch_size=PARAMS["batch_size"],
        callbacks=[EarlyStopping(
            patience=PARAMS["early_stop_patience"],
            restore_best_weights=True,
        )],
        verbose=1,
    )

    model.save(PARAMS["model_save_path"])


if __name__ == "__main__":
    main()
