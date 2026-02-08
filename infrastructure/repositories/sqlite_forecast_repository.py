from __future__ import annotations

from datetime import datetime, date
from typing import List, Tuple

from domain.entities import ForecastSession, ModelForecast
from domain.repositories import ForecastRepository
from infrastructure.db.sqlite_db import SQLiteDB
from infrastructure.serialization.pickles import dumps, loads


class SQLiteForecastRepository(ForecastRepository):
    def __init__(self, db: SQLiteDB) -> None:
        self._db = db

    def init_schema(self) -> None:
        with self._db.connect() as con:
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS forecast_session (
                    session_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_timestamp TEXT NOT NULL,

                    project_name TEXT NOT NULL DEFAULT '',
                    project_description TEXT NOT NULL DEFAULT '',

                    file_name TEXT NOT NULL,
                    rows_count INTEGER NOT NULL,
                    date_range_start TEXT NOT NULL,
                    date_range_end TEXT NOT NULL,
                    start_date TEXT NOT NULL,
                    horizon INTEGER NOT NULL
                );
                """
            )

            con.execute(
                """
                CREATE TABLE IF NOT EXISTS model_forecast (
                    forecast_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id INTEGER NOT NULL,
                    model_type TEXT NOT NULL,
                    predictions BLOB NOT NULL,
                    metrics BLOB NOT NULL,
                    FOREIGN KEY(session_id) REFERENCES forecast_session(session_id) ON DELETE CASCADE
                );
                """
            )

            # ---- lightweight migration for existing DB ----
            cols = {row[1] for row in con.execute("PRAGMA table_info(forecast_session);").fetchall()}
            if "project_name" not in cols:
                con.execute("ALTER TABLE forecast_session ADD COLUMN project_name TEXT NOT NULL DEFAULT ''")
            if "project_description" not in cols:
                con.execute("ALTER TABLE forecast_session ADD COLUMN project_description TEXT NOT NULL DEFAULT ''")

    def create_session(self, session: ForecastSession) -> int:
        with self._db.connect() as con:
            cur = con.execute(
                """
                INSERT INTO forecast_session (
                    session_timestamp,
                    project_name, project_description,
                    file_name, rows_count,
                    date_range_start, date_range_end,
                    start_date, horizon
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session.session_timestamp.isoformat(timespec="seconds"),
                    session.project_name,
                    session.project_description,
                    session.file_name,
                    int(session.rows_count),
                    session.date_range_start.isoformat(),
                    session.date_range_end.isoformat(),
                    session.start_date.isoformat(),
                    int(session.horizon),
                ),
            )
            return int(cur.lastrowid)

    def add_model_forecast(self, mf: ModelForecast) -> int:
        with self._db.connect() as con:
            cur = con.execute(
                """
                INSERT INTO model_forecast (session_id, model_type, predictions, metrics)
                VALUES (?, ?, ?, ?)
                """,
                (
                    int(mf.session_id),
                    mf.model_type,
                    dumps({"dates": mf.prediction_dates, "pred": mf.predictions}),
                    dumps(mf.metrics),
                ),
            )
            return int(cur.lastrowid)

    def list_sessions(self, limit: int = 200) -> List[ForecastSession]:
        with self._db.connect() as con:
            rows = con.execute(
                """
                SELECT session_id, session_timestamp,
                       project_name, project_description,
                       file_name, rows_count,
                       date_range_start, date_range_end,
                       start_date, horizon
                FROM forecast_session
                ORDER BY session_timestamp DESC, session_id DESC
                LIMIT ?
                """,
                (int(limit),),
            ).fetchall()

        out: List[ForecastSession] = []
        for r in rows:
            out.append(
                ForecastSession(
                    session_id=int(r[0]),
                    session_timestamp=datetime.fromisoformat(r[1]),
                    project_name=str(r[2] or ""),
                    project_description=str(r[3] or ""),
                    file_name=str(r[4]),
                    rows_count=int(r[5]),
                    date_range_start=date.fromisoformat(r[6]),
                    date_range_end=date.fromisoformat(r[7]),
                    start_date=date.fromisoformat(r[8]),
                    horizon=int(r[9]),
                )
            )
        return out

    def load_session(self, session_id: int) -> Tuple[ForecastSession, List[ModelForecast]]:
        with self._db.connect() as con:
            s = con.execute(
                """
                SELECT session_id, session_timestamp,
                       project_name, project_description,
                       file_name, rows_count,
                       date_range_start, date_range_end,
                       start_date, horizon
                FROM forecast_session
                WHERE session_id = ?
                """,
                (int(session_id),),
            ).fetchone()
            if s is None:
                raise KeyError(f"session_id={session_id} not found")

            session = ForecastSession(
                session_id=int(s[0]),
                session_timestamp=datetime.fromisoformat(s[1]),
                project_name=str(s[2] or ""),
                project_description=str(s[3] or ""),
                file_name=str(s[4]),
                rows_count=int(s[5]),
                date_range_start=date.fromisoformat(s[6]),
                date_range_end=date.fromisoformat(s[7]),
                start_date=date.fromisoformat(s[8]),
                horizon=int(s[9]),
            )

            rows = con.execute(
                """
                SELECT forecast_id, session_id, model_type, predictions, metrics
                FROM model_forecast
                WHERE session_id = ?
                ORDER BY forecast_id ASC
                """,
                (int(session_id),),
            ).fetchall()

        mfs: List[ModelForecast] = []
        for r in rows:
            pack = loads(r[3])
            met = loads(r[4])
            mfs.append(
                ModelForecast(
                    forecast_id=int(r[0]),
                    session_id=int(r[1]),
                    model_type=str(r[2]),
                    predictions=list(pack["pred"]),
                    prediction_dates=list(pack["dates"]),
                    metrics=dict(met),
                )
            )

        return session, mfs

    def delete_session(self, session_id: int) -> None:
        with self._db.connect() as con:
            con.execute("DELETE FROM forecast_session WHERE session_id = ?", (int(session_id),))

    def session_exists(self, session_id: int) -> bool:
        with self._db.connect() as con:
            row = con.execute(
                "SELECT 1 FROM forecast_session WHERE session_id = ? LIMIT 1",
                (int(session_id),),
            ).fetchone()
        return row is not None
