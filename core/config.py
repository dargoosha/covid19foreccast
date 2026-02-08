from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class AppConfig:
    # ---- Data ----
    required_cols: tuple[str, ...] = ("date", "new_cases", "hosp_patients", "mobility")
    date_col: str = "date"
    target_col: str = "new_cases"
    population_col: str = "population"

    window_size: int = 7
    val_fraction: float = 0.15
    test_fraction: float = 0.15

    # ---- SEIR baseline params (for SEIR+LSTM old residual) ----
    N_pop: float = 3.4e8
    incubation_days: float = 5.0
    infectious_days: float = 7.0
    R0_value: float = 2.5
    E0: float = 10.0
    I0: float = 10.0
    R0_init: float = 0.0

    # ---- dyn-SEIR params (for SEIR+LSTM new = dyn_seir_lstm_clean) ----
    population_fallback: float = 1_000_000.0
    E0_factor: float = 3.0
    I0_days: int = 7
    R0_init_dyn: float = 0.0
    sigma_dyn: float = 1 / 5.2
    beta_grid: tuple[float, float, int] = (0.05, 1.20, 60)
    gamma_grid: tuple[float, float, int] = (0.03, 0.50, 50)
    fit_target: str = "sigmaE"

    # ---- Storage ----
    sqlite_path: str = "forecast_history.sqlite3"

    # ---- Models ----
    model_paths: Dict[str, str] = None  # type: ignore

    def __post_init__(self) -> None:
        if self.model_paths is None:
            object.__setattr__(
                self,
                "model_paths",
                {
                    "LSTM": r"models\lstm\lstm.keras",
                    "SEIR+LSTM old": r"models\seir_lstm\seir_lstm.keras",
                    "SEIR+LSTM new": r"models\dyn_seir_lstm\dyn_seir_lstm_clean.keras",
                },
            )


CFG = AppConfig()

MODEL_ORDER = ["LSTM", "SEIR+LSTM old", "SEIR+LSTM new"]
