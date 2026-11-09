import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from queue import Queue, Empty
from typing import Any
from app.core.config import get_settings

settings = get_settings()

_SCHEMA = """
CREATE TABLE IF NOT EXISTS query_audit (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    question TEXT NOT NULL,
    source_used TEXT NOT NULL,
    trace_json TEXT NOT NULL,
    citations_json TEXT NOT NULL DEFAULT '[]'
);
CREATE INDEX IF NOT EXISTS idx_query_audit_created ON query_audit(created_at);
"""


class SQLiteConnectionPool:
    """Thread-safe connection pool with WAL mode and busy timeout for concurrent loads."""

    def __init__(self, db_path: str, max_connections: int = 10, timeout: float = 10.0):
        self.db_path = db_path
        self.max_connections = max_connections
        self.timeout = timeout
        self._pool: Queue[sqlite3.Connection] = Queue(maxsize=max_connections)
        self._created_connections = 0
        self._lock = threading.Lock()
        self._init_done = False

    def _create_connection(self) -> sqlite3.Connection:
        path = Path(self.db_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        con = sqlite3.connect(
            self.db_path,
            timeout=self.timeout,
            check_same_thread=False,
        )
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA journal_mode = WAL;")
        con.execute("PRAGMA synchronous = NORMAL;")
        con.execute("PRAGMA busy_timeout = 5000;")
        return con

    def init_db(self) -> None:
        with self._lock:
            con = self._create_connection()
            try:
                con.executescript(_SCHEMA)
                con.commit()
                self._init_done = True
            finally:
                con.close()

    @contextmanager
    def get_connection(self):
        if not self._init_done:
            self.init_db()

        con = None
        try:
            con = self._pool.get_nowait()
        except Empty:
            with self._lock:
                if self._created_connections < self.max_connections:
                    con = self._create_connection()
                    self._created_connections += 1
            if con is None:
                con = self._pool.get(timeout=self.timeout)

        try:
            yield con
        finally:
            try:
                self._pool.put_nowait(con)
            except Exception:
                con.close()

    def close_all(self) -> None:
        """Drain and close all connections in pool."""
        with self._lock:
            while not self._pool.empty():
                try:
                    con = self._pool.get_nowait()
                    con.close()
                except Empty:
                    break
            self._created_connections = 0


_pool = SQLiteConnectionPool(settings.audit_db_path)


def init_db() -> None:
    _pool.init_db()


def write_audit(
    question: str,
    source_used: str,
    trace: list[str],
    citations: list[dict[str, Any]] | None = None,
) -> None:
    with _pool.get_connection() as con:
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


def get_recent_audits(limit: int = 10) -> list[dict[str, Any]]:
    with _pool.get_connection() as con:
        cursor = con.execute(
            "SELECT id, created_at, question, source_used, trace_json, citations_json FROM query_audit ORDER BY id DESC LIMIT ?",
            (limit,),
        )
        rows = cursor.fetchall()
        return [
            {
                "id": r["id"],
                "created_at": r["created_at"],
                "question": r["question"],
                "source_used": r["source_used"],
                "trace": json.loads(r["trace_json"]),
                "citations": json.loads(r["citations_json"]),
            }
            for r in rows
        ]
