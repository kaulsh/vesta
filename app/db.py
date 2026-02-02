import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from flask import Flask, current_app, g


SCHEMA = """
CREATE TABLE IF NOT EXISTS cycles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    start_date TEXT NOT NULL,
    end_date TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""


def get_db() -> sqlite3.Connection:
    if "db" not in g:
        g.db = sqlite3.connect(current_app.config["DATABASE"])
        g.db.row_factory = sqlite3.Row
    return g.db


def close_db(error=None) -> None:
    db = g.pop("db", None)
    if db is not None:
        db.close()


def init_db(app: Flask) -> None:
    db_path = Path(app.config["DATABASE"])
    db_path.parent.mkdir(parents=True, exist_ok=True)

    @app.teardown_appcontext
    def teardown_db(error=None):
        close_db(error)

    with app.app_context():
        db = get_db()
        db.executescript(SCHEMA)
        db.commit()


def insert_cycle(start_date: str, end_date: str) -> None:
    db = get_db()
    db.execute(
        "INSERT INTO cycles (start_date, end_date, created_at) VALUES (?, ?, ?)",
        (start_date, end_date, datetime.utcnow().isoformat()),
    )
    db.commit()


def fetch_latest_cycles(limit: int = 6, oldest_first: bool = True) -> List[Dict[str, str]]:
    db = get_db()
    rows = db.execute(
        "SELECT start_date, end_date FROM cycles ORDER BY start_date DESC LIMIT ?",
        (limit,),
    ).fetchall()
    result = [
        {"start_date": row["start_date"], "end_date": row["end_date"]}
        for row in rows
    ]
    # Return oldest -> newest by default (for model input)
    return list(reversed(result)) if oldest_first else result


def count_cycles() -> int:
    db = get_db()
    row = db.execute("SELECT COUNT(*) AS count FROM cycles").fetchone()
    return int(row["count"])
