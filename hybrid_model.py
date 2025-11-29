# hybrid_model.py

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List

from lstm_model import LSTMResidualModel
from seir_model import (
    run_seir_simulation,
    run_seir_simulation_time_varying,
)


def _build_param_history_from_seir(
    df_original: pd.DataFrame,
    seir_params: Dict,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Формує "історичні" time-varying параметри β(t), σ(t), γ(t),
    виходячи з порівняння базового SEIR-прогнозу та реальних new_cases.

    Ідея:
      - рахуємо базовий SEIR з фіксованими параметрами;
      - рахуємо new_cases_seir(t) = sigma * E(t);
      - обчислюємо коефіцієнт r(t) = actual(t) / new_cases_seir(t);
      - β(t) = β0 * r(t) (кліпаємо r(t) в [0.25, 4.0]);
      - σ(t) ≡ σ0, γ(t) ≡ γ0 (для спрощення).
    """
    n = len(df_original)
    beta0 = float(seir_params["beta"])
    sigma0 = float(seir_params["sigma"])
    gamma0 = float(seir_params["gamma"])
    N = float(seir_params["population"])
    E0 = float(seir_params["E0"])
    I0 = float(seir_params["I0"])

    # Базова SEIR-траєкторія на всій історії
    seir_out = run_seir_simulation(
        days=n,
        beta=beta0,
        sigma=sigma0,
        gamma=gamma0,
        N=N,
        E0=E0,
        I0=I0,
    )
    seir_new_cases = np.asarray(seir_out["new_cases"], dtype=float)

    actual_cases = np.asarray(df_original["new_cases"].values, dtype=float)

    eps = 1e-6
    ratio = (actual_cases + eps) / (seir_new_cases + eps)

    # Кліпаємо ratio, щоб уникнути вибухів β(t)
    ratio = np.clip(ratio, 0.25, 4.0)

    beta_t = beta0 * ratio
    sigma_t = np.full_like(beta_t, sigma0, dtype=float)
    gamma_t = np.full_like(beta_t, gamma0, dtype=float)

    params_hist = np.stack([beta_t, sigma_t, gamma_t], axis=1)  # (n, 3)

    return params_hist, seir_new_cases


def _make_param_dataset(
    features: np.ndarray,
    param_series_scaled: np.ndarray,
    window_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Створює датасет для LSTM:
      X.shape = (samples, window_size, num_features)
      y.shape = (samples, 3)  # [beta, sigma, gamma] у масштабі [0,1]
    """
    X_list, y_list = [], []
    n = len(features)
    for i in range(n - window_size):
        X_list.append(features[i : i + window_size])
        y_list.append(param_series_scaled[i + window_size])
    X = np.array(X_list, dtype=float)
    y = np.array(y_list, dtype=float)
    return X, y


def train_hybrid_model(
    df_norm: pd.DataFrame,
    df_original: pd.DataFrame,
    feature_cols: List[str],
    window_size: int,
    seir_params: Dict,
    train_ratio: float = 0.8,
    lstm_units: int = 64,
    lstm_layers: int = 1,
    dropout: float = 0.2,
    epochs: int = 50,
) -> Tuple[LSTMResidualModel, Dict[str, np.ndarray], np.ndarray]:
    """
    Навчання гібридної моделі SEIR + LSTM з time-varying параметрами.

    Кроки:
      1) Отримуємо "історичні" β(t), σ(t), γ(t) із порівняння SEIR і реальних даних.
      2) Нормалізуємо параметри до [0, 1] по кожному столбцю.
      3) Формуємо ознаки X із df_norm[feature_cols] ковзним вікном.
      4) Навчаємо LSTM, яка по вікну ознак прогнозує [β, σ, γ].
    """
    # 1. Історичні time-varying параметри
    params_hist, seir_base_new_cases = _build_param_history_from_seir(
        df_original, seir_params
    )  # shape (n, 3)

    # 2. Масштабування параметрів до [0, 1]
    #    (проста міні-макс нормалізація по кожному параметру)
    param_min = params_hist.min(axis=0)
    param_max = params_hist.max(axis=0)
    diff = np.where(param_max - param_min < 1e-8, 1.0, param_max - param_min)

    params_hist_scaled = (params_hist - param_min) / diff

    # 3. Формуємо ознаки X з df_norm
    features = df_norm[feature_cols].values.astype(float)

    X, y = _make_param_dataset(
        features=features,
        param_series_scaled=params_hist_scaled,
        window_size=window_size,
    )

    if len(X) < 10:
        raise ValueError(
            "Недостатньо даних для навчання гібридної моделі (X.shape[0] < 10)."
        )

    train_size = int(train_ratio * len(X))
    train_size = max(1, min(train_size, len(X) - 1))

    X_train, y_train = X[:train_size], y[:train_size]
    # X_test, y_test = X[train_size:], y[train_size:]  # за потреби

    # 4. LSTM-модель параметрів
    model = LSTMResidualModel(
        window_size=window_size,
        num_features=len(feature_cols),
        lstm_units=lstm_units,
        num_layers=lstm_layers,
        dropout=dropout,
        output_dim=3,  # [beta, sigma, gamma]
    )
    model.fit(X_train, y_train, epochs=epochs, verbose=0)

    param_scalers = {
        "param_min": param_min,
        "param_max": param_max,
        "seir_base_new_cases": seir_base_new_cases,
    }

    return model, param_scalers, params_hist


def forecast_hybrid(
    df_norm: pd.DataFrame,
    df_original: pd.DataFrame,
    feature_cols: List[str],
    window_size: int,
    seir_params: Dict,
    scaler_target,      # параметр можна залишити, але всередині його більше не використовуємо
    horizon: int,
    lstm_model: LSTMResidualModel,
    param_scalers: Dict[str, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Повертає:
      - baseline_future_real: базовий SEIR з фіксованими параметрами
      - params_future: [beta, sigma, gamma], які спрогнозувала LSTM
      - hybrid_future_real: new_cases з time-varying SEIR (це гібрид)
    """

    param_min = param_scalers["param_min"]
    param_max = param_scalers["param_max"]

    features = df_norm[feature_cols].values.astype(float)
    if len(features) < window_size:
        raise ValueError(
            "Недостатньо даних для формування останнього вікна ознак для гібридного прогнозу."
        )

    last_window = features[-window_size:]
    last_window = last_window.reshape(1, window_size, len(feature_cols))

    # 1. LSTM дає параметри в [0,1] → денормуємо до реальних [beta, sigma, gamma]
    pred_scaled = lstm_model.predict(last_window)[0]  # shape (3,)

    diff = np.where(param_max - param_min < 1e-8, 1.0, param_max - param_min)
    params_future = param_min + pred_scaled * diff  # [beta, sigma, gamma]

    beta_future = np.full(horizon, params_future[0], dtype=float)
    sigma_future = np.full(horizon, params_future[1], dtype=float)
    gamma_future = np.full(horizon, params_future[2], dtype=float)

    N = float(seir_params["population"])
    E0 = float(seir_params["E0"])
    I0 = float(seir_params["I0"])

    # 2. Time-varying SEIR з цими параметрами → це і є гібридний прогноз (у реальному масштабі випадків)
    seir_out_future = run_seir_simulation_time_varying(
        beta_series=beta_future,
        sigma_series=sigma_future,
        gamma_series=gamma_future,
        N=N,
        E0=E0,
        I0=I0,
    )
    hybrid_future_real = seir_out_future["new_cases"]  # НІЯКОГО inverse_transform

    # 3. Для порівняння будуємо базовий SEIR з фіксованими параметрами на тому ж горизонті
    seir_out_full = run_seir_simulation(
        days=len(df_original) + horizon,
        beta=float(seir_params["beta"]),
        sigma=float(seir_params["sigma"]),
        gamma=float(seir_params["gamma"]),
        N=N,
        E0=E0,
        I0=I0,
    )
    seir_cases_raw_full = seir_out_full["new_cases"]
    baseline_future_real = seir_cases_raw_full[-horizon:]

    return baseline_future_real, params_future, hybrid_future_real
