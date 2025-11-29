# data_processing.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, Dict, List
from config import DEFAULT_DATE_COLUMN


def load_csv_file(file) -> pd.DataFrame:
    """
    Зчитування CSV, приведення дати та базове очищення.
    """
    df = pd.read_csv(file)

    if DEFAULT_DATE_COLUMN in df.columns:
        date_col = DEFAULT_DATE_COLUMN
    else:
        date_col = df.columns[0]

    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(by=date_col).reset_index(drop=True)
    df = df.rename(columns={date_col: "date"})

    df = df.ffill().bfill()
    return df


def prepare_time_series(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, MinMaxScaler]]:
    """
    Очікує df з колонками:
    date, new_cases, hospitalizations, mobility

    Нормалізує всі числові колонки (окрім date) окремими MinMaxScaler.
    Повертає:
      - нормалізований df_norm
      - словник scalers[col_name] -> MinMaxScaler
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    df = df.set_index("date").asfreq("D")
    df = df.ffill().bfill()

    scalers: Dict[str, MinMaxScaler] = {}
    for col in df.columns:
        scaler = MinMaxScaler()
        values = df[col].values.reshape(-1, 1)
        scaled = scaler.fit_transform(values)
        df[col] = scaled.flatten()
        scalers[col] = scaler

    df = df.reset_index()
    return df, scalers


def make_lstm_dataset_multivariate(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_series: np.ndarray,
    window_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Формує X, y для LSTM:
      X.shape = (samples, window_size, num_features)
      y.shape = (samples, 1)
    target_series – масив (len(df),) у тій же послідовності, що й df.
    """
    data = df[feature_cols].values
    y = target_series

    X_list, y_list = [], []
    for i in range(len(df) - window_size):
        X_list.append(data[i : i + window_size])
        y_list.append(y[i + window_size])

    X = np.array(X_list)
    y_out = np.array(y_list).reshape(-1, 1)
    return X, y_out
