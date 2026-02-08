import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# =========================
# ПАРАМЕТРИ ЕКСПЕРИМЕНТУ
# =========================
PARAMS = {
    # ---- IO ----
    "data_path": "D:\Projects\Python\Covid\dataset.csv",
    "target_col": "new_cases",
    "model_save_path": "lstm.keras",

    # ---- Features ----
    # Чиста LSTM: прогнозує new_cases за історією ознак.
    # Вхід X: [new_cases, hosp_patients, mobility]
    "feature_cols": ["new_cases", "hosp_patients", "mobility"],

    # ---- Split / window ----
    "window_size": 7,
    "val_fraction": 0.15,
    "test_fraction": 0.15,

    # ---- LSTM модель (ПОВЕРНУТО M1) ----
    "model_name": "LSTM",
    "lstm1_units": 64,
    "lstm2_units": 32,
    "dropout": 0.5,
    "dense_units": 16,
    "lr": 1e-3,

    # ---- Training ----
    "epochs": 100,
    "batch_size": 32,
    "early_stop_patience": 10,

    # ---- Repro ----
    "seed": 42,
}

SEED = PARAMS["seed"]
np.random.seed(SEED)
tf.random.set_seed(SEED)


def make_sequences(X: np.ndarray, y: np.ndarray, window_size: int):
    X_seq, y_seq, idxs = [], [], []
    T = X.shape[0]
    for t in range(window_size, T):
        X_seq.append(X[t - window_size : t, :])
        y_seq.append(y[t])
        idxs.append(t)
    return (
        np.array(X_seq, dtype=np.float32),
        np.array(y_seq, dtype=np.float32),
        np.array(idxs, dtype=int),
    )


def build_model(cfg: dict, input_shape):
    model = Sequential()
    model.add(
        LSTM(
            cfg["lstm1_units"],
            return_sequences=True,  # бо друга LSTM є
            input_shape=input_shape,
        )
    )
    model.add(Dropout(cfg["dropout"]))
    model.add(LSTM(cfg["lstm2_units"]))
    model.add(Dense(cfg["dense_units"], activation="relu"))
    model.add(Dense(1))

    optimizer = tf.keras.optimizers.Adam(learning_rate=cfg["lr"])
    model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])
    return model


def main():
    # =========================
    # 1) Завантаження та очистка
    # =========================
    df = pd.read_csv(PARAMS["data_path"])
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # потрібні колонки для фіч + таргет
    need_cols = set(PARAMS["feature_cols"] + ["date"])
    missing = [c for c in need_cols if c not in df.columns]
    if missing:
        raise ValueError(f"У dataset.csv відсутні колонки: {missing}")

    df = df.dropna(subset=PARAMS["feature_cols"])

    dates = df["date"].to_numpy()
    y_true = df[PARAMS["target_col"]].to_numpy(dtype=np.float32).reshape(-1, 1)

    n = len(df)
    print(f"Кількість спостережень після очистки: {n}")

    # =========================
    # 2) Split train/val/test
    # =========================
    val_fraction = PARAMS["val_fraction"]
    test_fraction = PARAMS["test_fraction"]

    train_size = int(n * (1.0 - val_fraction - test_fraction))
    val_size = int(n * val_fraction)
    test_size = n - train_size - val_size

    train_end_idx = train_size - 1
    val_end_idx = train_size + val_size - 1

    print(f"train: {train_size}, val: {val_size}, test: {test_size}")

    # =========================
    # 3) Масштабування X та y
    # =========================
    X_all = df[PARAMS["feature_cols"]].to_numpy(dtype=np.float32)

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    scaler_X.fit(X_all[: train_end_idx + 1])
    scaler_y.fit(y_true[: train_end_idx + 1])

    X_scaled = scaler_X.transform(X_all)
    y_scaled = scaler_y.transform(y_true)

    # =========================
    # 4) Вікна
    # =========================
    X_seq, y_seq, idxs = make_sequences(X_scaled, y_scaled, PARAMS["window_size"])

    train_mask = idxs <= train_end_idx
    val_mask = (idxs > train_end_idx) & (idxs <= val_end_idx)
    test_mask = idxs > val_end_idx

    X_train, y_train = X_seq[train_mask], y_seq[train_mask]
    X_val, y_val = X_seq[val_mask], y_seq[val_mask]
    X_test, y_test = X_seq[test_mask], y_seq[test_mask]
    idxs_test = idxs[test_mask]

    print(f"Кількість вікон: train={X_train.shape[0]}, val={X_val.shape[0]}, test={X_test.shape[0]}")

    dates_all = dates[idxs]
    dates_test = dates[idxs_test]

    # y_true для оцінки (в оригінальному масштабі)
    y_true_all = y_true[idxs].flatten()
    y_true_test = y_true[idxs_test].flatten()

    # =========================
    # 5) Навчання M1 (чиста LSTM)
    # =========================
    model_cfg = {
        "model_name": PARAMS["model_name"],
        "lstm1_units": PARAMS["lstm1_units"],
        "lstm2_units": PARAMS["lstm2_units"],
        "dropout": PARAMS["dropout"],
        "dense_units": PARAMS["dense_units"],
        "lr": PARAMS["lr"],
    }

    model = build_model(model_cfg, input_shape=(PARAMS["window_size"], X_train.shape[2]))

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=PARAMS["early_stop_patience"],
        restore_best_weights=True,
    )

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=PARAMS["epochs"],
        batch_size=PARAMS["batch_size"],
        callbacks=[early_stop],
        verbose=1,
    )

    # =========================
    # 6) Прогноз та оцінка
    # =========================
    y_pred_all_scaled = model.predict(X_seq, verbose=0)
    y_pred_all = scaler_y.inverse_transform(y_pred_all_scaled).flatten()

    y_pred_test_scaled = y_pred_all_scaled[test_mask]
    y_pred_test = scaler_y.inverse_transform(y_pred_test_scaled).flatten()

    y_pred_all = np.clip(y_pred_all, a_min=0.0, a_max=None)
    y_pred_test = np.clip(y_pred_test, a_min=0.0, a_max=None)

    mse_test = np.mean((y_true_test - y_pred_test) ** 2)
    mae_test = np.mean(np.abs(y_true_test - y_pred_test))

    print(f"\nMSE (тест) = {mse_test:.4f}")
    print(f"MAE (тест) = {mae_test:.4f}")

    model.save(PARAMS["model_save_path"])
    print(f"Модель збережено як {PARAMS['model_save_path']}")

    print("\nПерші 20 точок тестового прогнозу:")
    for i in range(min(20, len(dates_test))):
        print(
            f"{pd.to_datetime(dates_test[i]).strftime('%Y-%m-%d')} | "
            f"реальні={y_true_test[i]:.1f} | "
            f"прогноз={y_pred_test[i]:.1f}"
        )

    # =========================
    # 7) Графіки
    # =========================
    plt.figure(figsize=(14, 6))
    plt.plot(df["date"], df[PARAMS["target_col"]].to_numpy(dtype=np.float32), label="Реальні значення (new_cases)", alpha=0.6)
    plt.plot(dates_all, y_pred_all, label=f"LSTM прогноз ({PARAMS['model_name']})", linewidth=2)
    plt.xlabel("Дата")
    plt.ylabel("Кількість нових випадків")
    plt.title("Прогноз COVID-19 (чиста LSTM) — повний набір")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.plot(dates_test, y_true_test, label="Реальні значення (new_cases)")
    plt.plot(dates_test, y_pred_test, label="LSTM прогноз", linewidth=2)
    plt.xlabel("Дата")
    plt.ylabel("Кількість нових випадків")
    plt.title("Прогноз COVID-19 (чиста LSTM) — тестовий період")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 4))
    plt.plot(history.history["loss"], label="train_loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    plt.xlabel("Епоха")
    plt.ylabel("MSE")
    plt.title(f"Динаміка втрат — {PARAMS['model_name']}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
