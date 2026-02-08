# benchmark.py
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

CFG = {
    "data_path": r"D:\Projects\Python\Covid\dataset.csv",
    "date_col": "date",
    "target": "new_cases",
    "window": 7,
    "val_frac": 0.15,
    "test_frac": 0.15,
    "model_path": r"D:\Projects\Python\Covid\models\seir_lstm\seir_lstm.keras",

    # SEIR params (используются только если модель реально 4-feature residual)
    "N": 3.4e8,
    "incubation_period": 5.0,
    "infectious_period": 7.0,
    "R0": 2.5,
    "E0": 10.0,
    "I0": 10.0,
    "R0_init": 0.0,
}


def simulate_seir(days, N, beta, sigma, gamma, E0, I0, R0):
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

        new_cases.append(max(float(new_infectious), 0.0))

    return np.array(new_cases, dtype=np.float32)


def make_sequences(X, w):
    return np.array([X[i - w:i] for i in range(w, len(X))], dtype=np.float32)


def main(model_path: str):
    df = pd.read_csv(CFG["data_path"])
    df[CFG["date_col"]] = pd.to_datetime(df[CFG["date_col"]])
    df = df.sort_values(CFG["date_col"]).dropna(
        subset=[CFG["target"], "hosp_patients", "mobility"]
    ).reset_index(drop=True)

    y = df[CFG["target"]].to_numpy(np.float32)
    dates = df[CFG["date_col"]].to_numpy()
    n = len(df)
    train_end = int(n * (1.0 - CFG["val_frac"] - CFG["test_frac"]))  # exclusive

    model = tf.keras.models.load_model(model_path)
    input_dim = int(model.input_shape[-1])  # features count
    print("Model input_shape:", model.input_shape)

    # ===== MODE A: обычный прогноз (model expects 3 features) =====
    if input_dim == 3:
        X = df[[CFG["target"], "hosp_patients", "mobility"]].to_numpy(np.float32)

        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()
        scaler_X.fit(X[:train_end])
        scaler_y.fit(y[:train_end].reshape(-1, 1))

        X_scaled = scaler_X.transform(X).astype(np.float32)
        X_seq = make_sequences(X_scaled, CFG["window"])

        y_pred_scaled = model.predict(X_seq, verbose=0).astype(np.float32)
        y_pred = scaler_y.inverse_transform(y_pred_scaled).flatten().astype(np.float32)

        y_true = y[CFG["window"]:]
        dts = dates[CFG["window"]:]

        mse = np.mean((y_true - y_pred) ** 2)
        mae = np.mean(np.abs(y_true - y_pred))

        print("MODE: DIRECT (3 features)")
        print(f"MSE = {mse:.3f}")
        print(f"MAE = {mae:.3f}")

        plt.figure(figsize=(12, 5))
        plt.plot(dts, y_true, label="True")
        plt.plot(dts, y_pred, label="Prediction")
        plt.legend()
        plt.grid()
        plt.title("Model benchmark (DIRECT 3-feature)")
        plt.tight_layout()
        plt.show()
        return

    # ===== MODE B: SEIR + residual (model expects 4 features) =====
    if input_dim == 4:
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
        scale_factor = (y[:train_end].mean() + eps) / (seir_raw[:train_end].mean() + eps)
        seir_scaled = (seir_raw * scale_factor).astype(np.float32)

        X = np.column_stack([
            y,
            df["hosp_patients"].to_numpy(np.float32),
            df["mobility"].to_numpy(np.float32),
            seir_scaled,
        ]).astype(np.float32)

        residual = (y - seir_scaled).reshape(-1, 1).astype(np.float32)

        scaler_X = MinMaxScaler()
        scaler_res = MinMaxScaler()
        scaler_X.fit(X[:train_end])
        scaler_res.fit(residual[:train_end])

        X_scaled = scaler_X.transform(X).astype(np.float32)
        X_seq = make_sequences(X_scaled, CFG["window"])

        pred_res_scaled = model.predict(X_seq, verbose=0).astype(np.float32)
        pred_res = scaler_res.inverse_transform(pred_res_scaled).flatten().astype(np.float32)

        y_pred = (seir_scaled[CFG["window"]:] + pred_res).astype(np.float32)

        y_true = y[CFG["window"]:]
        dts = dates[CFG["window"]:]

        mse = np.mean((y_true - y_pred) ** 2)
        mae = np.mean(np.abs(y_true - y_pred))

        print(f"MSE = {mse:.3f}")
        print(f"MAE = {mae:.3f}")

        plt.figure(figsize=(12, 5))
        plt.plot(dts, y_true, label="True")
        plt.plot(dts, y_pred, label="Prediction")
        plt.legend()
        plt.grid()
        plt.title("Model benchmark (SEIR + LSTM residual)")
        plt.tight_layout()
        plt.show()
        return

    raise ValueError(f"Unsupported model input_dim={input_dim} (expected 3 or 4)")


if __name__ == "__main__":
    main(CFG["model_path"])
