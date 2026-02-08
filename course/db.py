# db.py

from sqlalchemy import (
    create_engine, Column, Integer, Float, String, Date, DateTime, Text, ForeignKey
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from datetime import datetime
import pandas as pd
from config import (
    DB_PATH,
    TABLE_TIME_SERIES,
    TABLE_EXPERIMENTS,
    TABLE_EXPERIMENT_RESULTS,
    TABLE_METRICS,
)

Base = declarative_base()


class TimeSeries(Base):
    __tablename__ = TABLE_TIME_SERIES

    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(Date, nullable=False)
    new_cases = Column(Float)
    hospitalizations = Column(Float)
    mobility = Column(Float)
    country = Column(String(64), default="USA")


class Experiment(Base):
    __tablename__ = TABLE_EXPERIMENTS

    id = Column(Integer, primary_key=True, autoincrement=True)
    model_type = Column(String(32))   # "SEIR", "LSTM", "SEIR+LSTM"
    created_at = Column(DateTime, default=datetime.utcnow)
    forecast_horizon = Column(Integer)  # K днів
    comment = Column(Text)

    results = relationship("ExperimentResult", back_populates="experiment")
    metrics = relationship("Metric", back_populates="experiment")


class ExperimentResult(Base):
    __tablename__ = TABLE_EXPERIMENT_RESULTS

    id = Column(Integer, primary_key=True, autoincrement=True)
    experiment_id = Column(Integer, ForeignKey("experiments.id"))
    date = Column(Date, nullable=False)
    actual = Column(Float)
    seir_pred = Column(Float)
    lstm_pred = Column(Float)
    hybrid_pred = Column(Float)

    experiment = relationship("Experiment", back_populates="results")


class Metric(Base):
    __tablename__ = TABLE_METRICS

    id = Column(Integer, primary_key=True, autoincrement=True)
    experiment_id = Column(Integer, ForeignKey("experiments.id"))
    model_type = Column(String(32))  # "SEIR", "LSTM", "SEIR+LSTM"
    rmse = Column(Float)
    mae = Column(Float)
    mape = Column(Float)

    experiment = relationship("Experiment", back_populates="metrics")


def get_engine():
    return create_engine(f"sqlite:///{DB_PATH}", echo=False, future=True)


def init_db():
    engine = get_engine()
    Base.metadata.create_all(engine)
    return engine


def get_session():
    engine = get_engine()
    Session = sessionmaker(bind=engine)
    return Session()


# --- Збереження / завантаження часових рядів ---


def save_time_series(df: pd.DataFrame, country: str = "USA"):
    """
    Очікує DataFrame з колонками: date, new_cases, hospitalizations, mobility.
    """
    session = get_session()
    try:
        for _, row in df.iterrows():
            ts = TimeSeries(
                date=row["date"].date() if isinstance(row["date"], pd.Timestamp) else row["date"],
                new_cases=float(row.get("new_cases", float("nan"))),
                hospitalizations=float(row.get("hospitalizations", float("nan"))),
                mobility=float(row.get("mobility", float("nan"))),
                country=country,
            )
            session.add(ts)
        session.commit()
    finally:
        session.close()


def load_time_series(country: str = "USA"):
    session = get_session()
    try:
        q = (
            session.query(TimeSeries)
            .filter(TimeSeries.country == country)
            .order_by(TimeSeries.date.asc())
        )
        rows = q.all()
        if not rows:
            return pd.DataFrame()
        data = [
            {
                "date": r.date,
                "new_cases": r.new_cases,
                "hospitalizations": r.hospitalizations,
                "mobility": r.mobility,
            }
            for r in rows
        ]
        return pd.DataFrame(data)
    finally:
        session.close()


# --- Експерименти та метрики ---


def create_experiment(model_type: str, forecast_horizon: int, comment: str = "") -> int:
    session = get_session()
    try:
        exp = Experiment(model_type=model_type, forecast_horizon=forecast_horizon, comment=comment)
        session.add(exp)
        session.commit()
        return exp.id
    finally:
        session.close()


def save_experiment_results(experiment_id: int, df_results: pd.DataFrame):
    """
    df_results: колонки [date, actual, seir_pred, lstm_pred, hybrid_pred]
    """
    session = get_session()
    try:
        for _, row in df_results.iterrows():
            r = ExperimentResult(
                experiment_id=experiment_id,
                date=row["date"].date() if isinstance(row["date"], pd.Timestamp) else row["date"],
                actual=float(row.get("actual", float("nan"))),
                seir_pred=float(row.get("seir_pred", float("nan"))),
                lstm_pred=float(row.get("lstm_pred", float("nan"))),
                hybrid_pred=float(row.get("hybrid_pred", float("nan"))),
            )
            session.add(r)
        session.commit()
    finally:
        session.close()


def save_metrics(experiment_id: int, model_type: str, rmse: float, mae: float, mape: float):
    session = get_session()
    try:
        m = Metric(
            experiment_id=experiment_id,
            model_type=model_type,
            rmse=rmse,
            mae=mae,
            mape=mape,
        )
        session.add(m)
        session.commit()
    finally:
        session.close()
