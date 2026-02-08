# benchmark_seir_lstm.py
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

CFG = {
    "data_path": r"D:\Projects\Python\Covid\dataset.csv",
    "date_col": "date",
    "target_col": "new_cases",
    "base_feature_cols": ["new_cases", "hosp_patients", "mobility"],

    "window_size": 7,
    "val_fraction": 0.15,
    "test_fraction": 0.15,

    "population_col": "population",
    "population_fallback": 1_000_000,

    "E0_factor": 3.0,
    "I0_days": 7,
    "R0_init": 0.0,

    "sigma": 1 / 5.2,
    "beta_grid": (0.05, 1.20, 60),
    "gamma_grid": (0.03, 0.50, 50),
    "fit_target": "sigmaE",

    "model_path": r"D:\Projects\Python\Covid\models\dyn_seir_lstm\dyn_seir_lstm_clean.keras",  # поставь свой путь
}

def make_sequences(X, w):
    return np.array([X[i - w:i] for i in range(w, len(X))], dtype=np.float32)

def seir_simulate(new_cases, N, beta, sigma, gamma, E0_factor, I0_days, R0_init):
    T = len(new_cases)
    I0 = float(np.sum(new_cases[:max(1, I0_days)]))
    E0 = float(E0_factor * I0)
    S0 = float(max(N - E0 - I0 - R0_init, 0.0))

    S = np.zeros(T, dtype=np.float32)
    E = np.zeros(T, dtype=np.float32)
    I = np.zeros(T, dtype=np.float32)
    R = np.zeros(T, dtype=np.float32)

    S[0], E[0], I[0], R[0] = S0, E0, I0, float(R0_init)

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
        "sigmaE": (sigma * E).astype(np.float32),
    }

def fit_seir_on_train(y_train, N, cfg):
    betas = np.linspace(*cfg["beta_grid"])
    gammas = np.linspace(*cfg["gamma_grid"])

    best_loss = np.inf
    best_beta = None
    best_gamma = None

    for beta in betas:
        for gamma in gammas:
            sim = seir_simulate(
                y_train, N, beta, cfg["sigma"], gamma,
                cfg["E0_factor"], cfg["I0_days"], cfg["R0_init"]
            )
            loss = np.mean((y_train - sim[cfg["fit_target"]]) ** 2)
            if loss < best_loss:
                best_loss = loss
                best_beta = float(beta)
                best_gamma = float(gamma)

    return best_beta, best_gamma

def main():
    df = pd.read_csv(CFG["data_path"])
    df[CFG["date_col"]] = pd.to_datetime(df[CFG["date_col"]])
    df = df.sort_values(CFG["date_col"]).dropna(
        subset=CFG["base_feature_cols"] + [CFG["target_col"]]
    )

    y = df[CFG["target_col"]].to_numpy(np.float32)
    X_base = df[CFG["base_feature_cols"]].to_numpy(np.float32)

    n = len(df)
    train_end = int(n * (1 - CFG["val_fraction"] - CFG["test_fraction"]))

    if CFG["population_col"] in df.columns:
        N = float(df[CFG["population_col"]].iloc[0])
    else:
        N = float(CFG["population_fallback"])

    beta, gamma = fit_seir_on_train(y[:train_end], N, CFG)
    sim = seir_simulate(
        y, N, beta, CFG["sigma"], gamma,
        CFG["E0_factor"], CFG["I0_days"], CFG["R0_init"]
    )

    X_seir = np.column_stack([
        sim["S"] / N,
        sim["E"] / N,
        sim["I"] / N,
        sim["R"] / N,
        sim["sigmaE"],
    ]).astype(np.float32)

    X_all = np.column_stack([X_base, X_seir]).astype(np.float32)
    y_all = y.reshape(-1, 1).astype(np.float32)

    # ВАЖНО: делаем так же, как в твоём train-коде (fit на всём)
    scaler_X = MinMaxScaler().fit(X_all)
    scaler_y = MinMaxScaler().fit(y_all)

    X_scaled = scaler_X.transform(X_all).astype(np.float32)

    X_seq = make_sequences(X_scaled, CFG["window_size"])

    model = tf.keras.models.load_model(CFG["model_path"])
    y_pred_scaled = model.predict(X_seq, verbose=0)
    y_pred = scaler_y.inverse_transform(y_pred_scaled).reshape(-1)

    y_true = y[CFG["window_size"]:]
    dates = df[CFG["date_col"]].to_numpy()[CFG["window_size"]:]

    mse = float(np.mean((y_true - y_pred) ** 2))
    mae = float(np.mean(np.abs(y_true - y_pred)))

    print(f"MSE = {mse:.3f}")
    print(f"MAE = {mae:.3f}")

    plt.figure(figsize=(12, 5))
    plt.plot(dates, y_true, label="True")
    plt.plot(dates, y_pred, label="Prediction")
    plt.legend()
    plt.grid()
    plt.title("SEIR+LSTM benchmark")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
