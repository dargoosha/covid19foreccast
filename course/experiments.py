# experiments.py

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt

from db import create_experiment, save_experiment_results, save_metrics
from data_processing import make_lstm_dataset_multivariate
from lstm_model import LSTMResidualModel
from hybrid_model import train_hybrid_model, forecast_hybrid
from seir_model import run_seir_simulation


def mape(y_true, y_pred):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    mask = y_true != 0
    if not np.any(mask):
        return 0.0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def evaluate_model(y_true, y_pred) -> Dict[str, float]:
    rmse = sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape_val = mape(y_true, y_pred)
    return {"rmse": rmse, "mae": mae, "mape": mape_val}


def run_experiment(
    df_norm: pd.DataFrame,
    df_original: pd.DataFrame,
    scalers: Dict[str, object],
    model_type: str,
    seir_params: Dict,
    horizon: int,
    lstm_window: int,
    lstm_units: int,
    lstm_layers: int,
    lstm_epochs: int,
    comment: str = "",
) -> Tuple[int, pd.DataFrame, Dict[str, Dict[str, float]]]:
    """
    Запускає експеримент:
      model_type: "SEIR", "LSTM", "SEIR+LSTM"
    horizon – горизонт прогнозу в днях (вперед від останньої дати).

    Повертає:
      exp_id, results_df, метрики для кожної моделі.
    """

    exp_id = create_experiment(
        model_type=model_type, forecast_horizon=horizon, comment=comment
    )

    feature_cols = ["new_cases", "hospitalizations", "mobility"]
    scaler_target = scalers["new_cases"]

    last_date = df_original["date"].iloc[-1]
    future_dates = pd.date_range(
        last_date + pd.Timedelta(days=1), periods=horizon, freq="D"
    )

    metrics_all: Dict[str, Dict[str, float]] = {}
    results = pd.DataFrame({"date": future_dates})

    # Фактичні значення для метрик – останні horizon днів історії
    actual_tail = df_original["new_cases"].values[-horizon:]
    results["actual"] = actual_tail

    # -------------------------------------------------------------------------
    # 1. Базовий SEIR-прогноз (фіксовані параметри)
    # -------------------------------------------------------------------------
    seir_out_full = run_seir_simulation(
        days=len(df_original) + horizon,
        beta=seir_params["beta"],
        sigma=seir_params["sigma"],
        gamma=seir_params["gamma"],
        N=seir_params["population"],
        E0=seir_params["E0"],
        I0=seir_params["I0"],
    )
    seir_cases_raw_full = seir_out_full["new_cases"]

    # беремо останні horizon днів як прогноз
    seir_future_real = seir_cases_raw_full[-horizon:]
    results["seir_pred"] = seir_future_real


    if model_type in ["SEIR", "SEIR+LSTM"]:
        metrics_seir = evaluate_model(actual_tail, seir_future_real)
        metrics_all["SEIR"] = metrics_seir
        save_metrics(exp_id, "SEIR", **metrics_seir)

    # -------------------------------------------------------------------------
    # 2. Чиста LSTM (багатовимірна) для прямого прогнозу new_cases
    # -------------------------------------------------------------------------
    if model_type in ["LSTM", "SEIR+LSTM"]:
        target_norm = df_norm["new_cases"].values
        X, y = make_lstm_dataset_multivariate(
            df_norm, feature_cols, target_series=target_norm, window_size=lstm_window
        )

        if len(X) <= horizon:
            raise ValueError(
                "Недостатньо даних для виділення тестового набору для LSTM."
            )

        train_size = len(X) - horizon
        X_train, y_train = X[:train_size], y[:train_size]
        X_test, y_test = X[train_size:], y[train_size:]

        lstm_model_direct = LSTMResidualModel(
            window_size=lstm_window,
            num_features=len(feature_cols),
            lstm_units=lstm_units,
            num_layers=lstm_layers,
            dropout=0.2,
            output_dim=1,
        )
        lstm_model_direct.fit(X_train, y_train, epochs=lstm_epochs, verbose=0)

        lstm_pred_norm = lstm_model_direct.predict(X_test).flatten()
        lstm_pred_real = scaler_target.inverse_transform(
            lstm_pred_norm.reshape(-1, 1)
        ).flatten()

        results["lstm_pred"] = lstm_pred_real

        metrics_lstm = evaluate_model(actual_tail, lstm_pred_real)
        metrics_all["LSTM"] = metrics_lstm
        save_metrics(exp_id, "LSTM", **metrics_lstm)

    # -------------------------------------------------------------------------
    # 3. Гібрид SEIR+LSTM (time-varying параметри)
    # -------------------------------------------------------------------------
    if model_type == "SEIR+LSTM":
        hybrid_lstm_model, param_scalers, params_hist = train_hybrid_model(
            df_norm=df_norm,
            df_original=df_original,
            feature_cols=feature_cols,
            window_size=lstm_window,
            seir_params=seir_params,
            train_ratio=0.8,
            lstm_units=lstm_units,
            lstm_layers=lstm_layers,
            dropout=0.2,
            epochs=lstm_epochs,
        )

        seir_future_real_hybrid, params_future, hybrid_future_real = forecast_hybrid(
            df_norm=df_norm,
            df_original=df_original,
            feature_cols=feature_cols,
            window_size=lstm_window,
            seir_params=seir_params,
            scaler_target=scaler_target,
            horizon=horizon,
            lstm_model=hybrid_lstm_model,
            param_scalers=param_scalers,
        )

        # Для наочності: базовий SEIR із forecast_hybrid можна
        # або використовувати замість попереднього, або як окремий сценарій.
        # Тут ми залишаємо базовий "seir_pred" як вище,
        # а гібридний – у стовпчику hybrid_pred.
        results["hybrid_pred"] = hybrid_future_real

        metrics_hybrid = evaluate_model(actual_tail, hybrid_future_real)
        metrics_all["SEIR+LSTM"] = metrics_hybrid
        save_metrics(exp_id, "SEIR+LSTM", **metrics_hybrid)

    # Зберігаємо результати експерименту в БД
    save_experiment_results(exp_id, results)

    return exp_id, results, metrics_all
