import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from app.core.config import get_settings

settings = get_settings()

def _connect():
    Path(settings.audit_db_path).parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(settings.audit_db_path)

def init_db() -> None:
    con = _connect()
    try:
        con.execute(
            """CREATE TABLE IF NOT EXISTS query_audit (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                question TEXT NOT NULL,
                source_used TEXT NOT NULL,
                trace_json TEXT NOT NULL,
                citations_json TEXT NOT NULL DEFAULT '[]'
            )"""
        )
        con.commit()
    finally:
        con.close()

def write_audit(question: str, source_used: str, trace: list[str], citations: list[dict] | None = None) -> None:
    con = _connect()
    try:
        con.execute(
            "INSERT INTO query_audit(created_at, question, source_used, trace_json, citations_json) VALUES (?, ?, ?, ?, ?)",
            (
                datetime.now(timezone.utc).isoformat(),
                question,
                source_used,
                json.dumps(trace),
                json.dumps(citations or []),
            ),
        )
        con.commit()
    finally:
        con.close()