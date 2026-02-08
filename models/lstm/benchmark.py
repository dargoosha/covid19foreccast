# benchmark_model.py
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

CFG = {
    "data_path": "D:/Projects/Python/Covid/dataset.csv",
    "target": "new_cases",
    "window": 7,
    "val_frac": 0.15,
    "test_frac": 0.15,
    "model_path": "D:\Projects\Python\Covid\models\lstm\lstm.keras",
}


def make_sequences(X, w):
    return np.array([X[i - w:i] for i in range(w, len(X))])


def main(model_path: str):
    df = pd.read_csv(CFG["data_path"])
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").dropna(
        subset=[CFG["target"], "hosp_patients", "mobility"]
    )

    y = df[CFG["target"]].to_numpy(np.float32)
    X = df[[CFG["target"], "hosp_patients", "mobility"]].to_numpy(np.float32)

    n = len(df)
    train_end = int(n * (1 - CFG["val_frac"] - CFG["test_frac"]))

    # ⚠️ ПОВТОРЯЕМ preprocessing
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    scaler_X.fit(X[:train_end])
    scaler_y.fit(y[:train_end].reshape(-1, 1))

    X_scaled = scaler_X.transform(X)
    X_seq = make_sequences(X_scaled, CFG["window"])

    model = tf.keras.models.load_model(model_path)

    y_pred_scaled = model.predict(X_seq, verbose=0)
    y_pred = scaler_y.inverse_transform(y_pred_scaled).flatten()

    y_true = y[CFG["window"]:]
    dates = df["date"].to_numpy()[CFG["window"]:]

    mse = np.mean((y_true - y_pred) ** 2)
    mae = np.mean(np.abs(y_true - y_pred))

    print(f"MSE = {mse:.3f}")
    print(f"MAE = {mae:.3f}")

    plt.figure(figsize=(12, 5))
    plt.plot(dates, y_true, label="True")
    plt.plot(dates, y_pred, label="Prediction")
    plt.legend()
    plt.grid()
    plt.title("Model benchmark")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main(CFG["model_path"])
