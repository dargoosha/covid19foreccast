from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, List

import numpy as np
from core.errors import DatasetValidationError
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


@dataclass(frozen=True)
class DatasetMeta:
    file_name: str
    rows_count: int
    date_start: pd.Timestamp
    date_end: pd.Timestamp


def load_and_validate_df(
    df: pd.DataFrame,
    required_cols: list[str],
    date_col: str,
) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame):
        raise DatasetValidationError("Файл не є коректним CSV.")

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise DatasetValidationError(
            message="Відсутні обовʼязкові колонки",
            missing_fields=missing,
        )

    try:
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
    except Exception:
        raise DatasetValidationError(
            f"Колонка `{date_col}` має некоректний формат дат."
        )

    if df.empty:
        raise DatasetValidationError("Датасет порожній.")

    return df


def clean_core_cols(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c not in df.columns:
            raise DatasetValidationError(
                message=f"Відсутня колонка `{c}`",
                missing_fields=[c],
            )

        if not pd.api.types.is_numeric_dtype(df[c]):
            raise DatasetValidationError(
                f"Колонка `{c}` повинна бути числовою."
            )

    df = df.dropna(subset=cols)
    if df.empty:
        raise DatasetValidationError(
            "Після очищення не залишилось жодного валідного рядка."
        )

    return df


def make_sequences_X_only(X: np.ndarray, w: int) -> Tuple[np.ndarray, np.ndarray]:
    Xs, idxs = [], []
    for t in range(w, len(X)):
        Xs.append(X[t - w : t, :])
        idxs.append(t)
    return np.array(Xs, dtype=np.float32), np.array(idxs, dtype=int)


def fit_scalers_train_only(
    X_all: np.ndarray,
    y_all: np.ndarray,
    train_end_idx: int,
) -> tuple[MinMaxScaler, MinMaxScaler]:
    sx = MinMaxScaler()
    sy = MinMaxScaler()
    sx.fit(X_all[: train_end_idx + 1])
    sy.fit(y_all[: train_end_idx + 1])
    return sx, sy
