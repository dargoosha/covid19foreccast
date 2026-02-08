from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from typing import Iterator


class SQLiteDB:
    def __init__(self, path: str) -> None:
        self._path = path

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        con = sqlite3.connect(self._path, detect_types=sqlite3.PARSE_DECLTYPES)
        try:
            con.execute("PRAGMA foreign_keys = ON;")
            yield con
            con.commit()
        finally:
            con.close()
