"""Response log for multiple-choice practice pages.

One SQLite file shared by all Streamlit processes (WAL mode handles the
8 server instances). Location: $PCHEM_PRACTICE_DB, default
~/pchem-data/practice.sqlite.
"""
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

SCHEMA = """
CREATE TABLE IF NOT EXISTS responses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    practice_set TEXT NOT NULL,
    student TEXT NOT NULL,
    student_key TEXT NOT NULL,
    session_id TEXT NOT NULL,
    question_id TEXT NOT NULL,
    attempt INTEGER NOT NULL,
    choice_index INTEGER NOT NULL,
    choice_text TEXT NOT NULL,
    correct INTEGER NOT NULL,
    seconds REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_student ON responses (practice_set, student_key);
"""


def db_path() -> Path:
    return Path(os.environ.get("PCHEM_PRACTICE_DB", Path.home() / "pchem-data" / "practice.sqlite"))


def student_key(name: str) -> str:
    return " ".join(name.lower().split())


def _connect() -> sqlite3.Connection:
    path = db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path, timeout=10)
    con.execute("PRAGMA journal_mode=WAL")
    con.executescript(SCHEMA)
    return con


def log_response(practice_set, student, session_id, question_id, choice_index,
                 choice_text, correct, seconds) -> int:
    """Insert one attempt and return its attempt number for this student and question."""
    key = student_key(student)
    con = _connect()
    try:
        with con:
            (prev,) = con.execute(
                "SELECT COUNT(*) FROM responses WHERE practice_set=? AND student_key=? AND question_id=?",
                (practice_set, key, question_id),
            ).fetchone()
            attempt = prev + 1
            con.execute(
                "INSERT INTO responses (ts, practice_set, student, student_key, session_id, question_id,"
                " attempt, choice_index, choice_text, correct, seconds) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (datetime.now(timezone.utc).isoformat(timespec="seconds"), practice_set, student.strip(),
                 key, session_id, question_id, attempt, int(choice_index), choice_text,
                 int(bool(correct)), round(float(seconds), 1)),
            )
    finally:
        con.close()
    return attempt


def responses(practice_set=None) -> pd.DataFrame:
    con = _connect()
    try:
        if practice_set is None:
            return pd.read_sql_query("SELECT * FROM responses ORDER BY id", con)
        return pd.read_sql_query(
            "SELECT * FROM responses WHERE practice_set=? ORDER BY id", con, params=(practice_set,)
        )
    finally:
        con.close()


def finished_questions(practice_set, student, max_tries) -> set:
    """Question ids this student has answered correctly, or missed max_tries times."""
    con = _connect()
    try:
        rows = con.execute(
            "SELECT question_id FROM responses WHERE practice_set=? AND student_key=?"
            " GROUP BY question_id HAVING MAX(correct)=1 OR COUNT(*)>=?",
            (practice_set, student_key(student), max_tries),
        ).fetchall()
    finally:
        con.close()
    return {r[0] for r in rows}
