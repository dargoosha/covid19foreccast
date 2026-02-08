from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

from core.config import AppConfig, MODEL_ORDER
from infrastructure.ml.preprocessing import make_sequences_X_only
from infrastructure.ml.seir import simulate_seir
from infrastructure.ml.dyn_seir import fit_seir_on_train, seir_simulate_dyn


def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.array(y_true, dtype=np.float32)
    y_pred = np.array(y_pred, dtype=np.float32)
    return float(np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100.0)


def _calc_metrics(y_t: np.ndarray, y_p: np.ndarray) -> Dict[str, float]:
    if len(y_p) == 0:
        return {"mape": np.nan, "mae": np.nan, "rmse": np.nan, "mae_pct": np.nan, "rmse_pct": np.nan}
    mae = float(np.mean(np.abs(y_t - y_p)))
    rmse = float(np.sqrt(np.mean((y_t - y_p) ** 2)))
    mape = float(_mape(y_t, y_p))
    mean_true = float(np.mean(y_t) + 1e-8)
    return {
        "mape": mape,
        "mae": mae,
        "rmse": rmse,
        "mae_pct": (mae / mean_true) * 100.0,
        "rmse_pct": (rmse / mean_true) * 100.0,
    }


@dataclass(frozen=True)
class ForecastResult:
    model_type: str
    dates_iso: List[str]
    pred: List[float]
    metrics: Dict[str, Any]


class ThreeModelForecaster:
    def __init__(self, cfg: AppConfig) -> None:
        self._cfg = cfg
        self._cache = {}

    def _load(self, model_type: str):
        if model_type in self._cache:
            return self._cache[model_type]
        m = load_model(self._cfg.model_paths[model_type])
        self._cache[model_type] = m
        return m

    def forecast(self, df: pd.DataFrame, start_idx: int, horizon: int) -> Dict[str, ForecastResult]:
        cfg = self._cfg
        n = len(df)

        y_true = df[cfg.target_col].astype(float).to_numpy(dtype=np.float32)
        dates = pd.to_datetime(df[cfg.date_col]).dt.strftime("%Y-%m-%d").tolist()
        X_base = df[["new_cases", "hosp_patients", "mobility"]].to_numpy(dtype=np.float32)

        train_size = int(n * (1.0 - float(cfg.val_fraction) - float(cfg.test_fraction)))
        train_end = train_size - 1
        w = int(cfg.window_size)

        # общий список дат прогноза (t = start+1..)
        def _out_dates() -> List[str]:
            out = []
            for step in range(1, int(horizon) + 1):
                t = start_idx + step
                if t >= n:
                    break
                if t - w < 0:
                    continue
                out.append(dates[t])
            return out

        out_dates_iso = _out_dates()

        # true segment для метрик
        if len(out_dates_iso) > 0:
            date_to_idx = {dates[i]: i for i in range(n)}
            y_seg = np.array([y_true[date_to_idx[d]] for d in out_dates_iso], dtype=np.float32)
        else:
            y_seg = np.array([], dtype=np.float32)

        results: Dict[str, ForecastResult] = {}

        # ============================
        # 1) LSTM (pure)
        # ============================
        m_lstm = self._load("LSTM")
        sx = MinMaxScaler().fit(X_base[: train_end + 1])
        sy = MinMaxScaler().fit(y_true[: train_end + 1].reshape(-1, 1))

        X_lstm_scaled = sx.transform(X_base).astype(np.float32)
        X_lstm_seq, _ = make_sequences_X_only(X_lstm_scaled, w)

        pred_lstm = []
        for step in range(1, int(horizon) + 1):
            t = start_idx + step
            if t >= n or t - w < 0:
                break
            k = t - w
            y_scaled = m_lstm.predict(X_lstm_seq[k:k+1], verbose=0)
            y = float(sy.inverse_transform(y_scaled)[0, 0])
            pred_lstm.append(max(y, 0.0))

        results["LSTM"] = ForecastResult(
            model_type="LSTM",
            dates_iso=out_dates_iso[: len(pred_lstm)],
            pred=[float(x) for x in pred_lstm],
            metrics=_calc_metrics(y_seg[: len(pred_lstm)], np.array(pred_lstm, dtype=np.float32)),
        )

        # ============================
        # 2) SEIR+LSTM old (residual)
        # ============================
        m_old = self._load("SEIR+LSTM old")

        sigma = 1.0 / float(cfg.incubation_days)
        gamma = 1.0 / float(cfg.infectious_days)
        beta = float(cfg.R0_value) * gamma

        seir_raw = simulate_seir(
            days=n,
            N=float(cfg.N_pop),
            beta=float(beta),
            sigma=float(sigma),
            gamma=float(gamma),
            E0=float(cfg.E0),
            I0=float(cfg.I0),
            R0=float(cfg.R0_init),
        )

        eps = 1e-8
        scale = (y_true[: train_end + 1].mean() + eps) / (seir_raw[: train_end + 1].mean() + eps)
        seir_scaled = (seir_raw * scale).astype(np.float32)

        X_old = np.column_stack([X_base, seir_scaled]).astype(np.float32)
        residual = (y_true - seir_scaled).reshape(-1, 1).astype(np.float32)

        sx_old = MinMaxScaler().fit(X_old[: train_end + 1])
        sr = MinMaxScaler().fit(residual[: train_end + 1])

        X_old_scaled = sx_old.transform(X_old).astype(np.float32)
        X_old_seq, _ = make_sequences_X_only(X_old_scaled, w)

        pred_old = []
        for step in range(1, int(horizon) + 1):
            t = start_idx + step
            if t >= n or t - w < 0:
                break
            k = t - w
            r_scaled = m_old.predict(X_old_seq[k:k+1], verbose=0)
            r = float(sr.inverse_transform(r_scaled)[0, 0])
            y = float(seir_scaled[t] + r)
            pred_old.append(max(y, 0.0))

        results["SEIR+LSTM old"] = ForecastResult(
            model_type="SEIR+LSTM old",
            dates_iso=out_dates_iso[: len(pred_old)],
            pred=[float(x) for x in pred_old],
            metrics=_calc_metrics(y_seg[: len(pred_old)], np.array(pred_old, dtype=np.float32)),
        )

        # ============================
        # 3) SEIR+LSTM new (dyn_seir_lstm_clean)
        # ============================
        m_new = self._load("SEIR+LSTM new")

        if cfg.population_col in df.columns and pd.notna(df[cfg.population_col].iloc[0]):
            N_dyn = float(df[cfg.population_col].iloc[0])
        else:
            N_dyn = float(cfg.population_fallback)

        beta_dyn, gamma_dyn = fit_seir_on_train(
            y_train=y_true[: train_end + 1],
            N=N_dyn,
            beta_grid=cfg.beta_grid,
            gamma_grid=cfg.gamma_grid,
            sigma=float(cfg.sigma_dyn),
            E0_factor=float(cfg.E0_factor),
            I0_days=int(cfg.I0_days),
            R0_init=float(cfg.R0_init_dyn),
            fit_target=str(cfg.fit_target),
        )

        sim = seir_simulate_dyn(
            y=y_true,
            N=N_dyn,
            beta=float(beta_dyn),
            sigma=float(cfg.sigma_dyn),
            gamma=float(gamma_dyn),
            E0_factor=float(cfg.E0_factor),
            I0_days=int(cfg.I0_days),
            R0_init=float(cfg.R0_init_dyn),
        )

        X_dyn_seir = np.column_stack(
            [
                sim["S"] / N_dyn,
                sim["E"] / N_dyn,
                sim["I"] / N_dyn,
                sim["R"] / N_dyn,
                sim["sigmaE"],
            ]
        ).astype(np.float32)

        X_new = np.column_stack([X_base, X_dyn_seir]).astype(np.float32)
        y_all = y_true.reshape(-1, 1).astype(np.float32)

        # ВАЖНО: как в твоём dyn_seir_lstm_clean — fit на ВСЁМ (leakage by design)
        sx_new = MinMaxScaler().fit(X_new)
        sy_new = MinMaxScaler().fit(y_all)

        X_new_scaled = sx_new.transform(X_new).astype(np.float32)
        X_new_seq, _ = make_sequences_X_only(X_new_scaled, w)

        pred_new = []
        for step in range(1, int(horizon) + 1):
            t = start_idx + step
            if t >= n or t - w < 0:
                break
            k = t - w
            y_scaled = m_new.predict(X_new_seq[k:k+1], verbose=0)
            y = float(sy_new.inverse_transform(y_scaled)[0, 0])
            pred_new.append(max(y, 0.0))

        results["SEIR+LSTM new"] = ForecastResult(
            model_type="SEIR+LSTM new",
            dates_iso=out_dates_iso[: len(pred_new)],
            pred=[float(x) for x in pred_new],
            metrics=_calc_metrics(y_seg[: len(pred_new)], np.array(pred_new, dtype=np.float32)),
        )

        # гарантированный порядок ключей для UI/сохранения
        ordered: Dict[str, ForecastResult] = {}
        for k in MODEL_ORDER:
            if k in results:
                ordered[k] = results[k]
        return ordered
