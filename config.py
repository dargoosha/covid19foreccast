# config.py

DB_PATH = "covid_forecast.db"

TABLE_TIME_SERIES = "time_series"
TABLE_EXPERIMENTS = "experiments"
TABLE_EXPERIMENT_RESULTS = "experiment_results"
TABLE_METRICS = "metrics"

# Тепер цільова змінна в системі – УЖЕ ПЕРЕЙМЕНОВАНА new_cases
DEFAULT_DATE_COLUMN = "date"
DEFAULT_TARGET_COLUMN = "new_cases"  # <- важливо

DEFAULT_LSTM_WINDOW = 14

SEIR_DEFAULTS = {
    "beta": 0.3,
    "sigma": 1 / 5.2,
    "gamma": 1 / 7.0,
    "population": 3.3e8,
    "E0": 1000.0,
    "I0": 1000.0,
}
