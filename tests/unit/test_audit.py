import concurrent.futures
import sqlite3
from pathlib import Path
from app.services.audit import SQLiteConnectionPool, write_audit, get_recent_audits


def test_sqlite_pool_wal_mode_and_init(tmp_dir):
    db_path = str(tmp_dir / "audit_test.db")
    pool = SQLiteConnectionPool(db_path, max_connections=5)
    pool.init_db()

    # Verify database file was dynamically created
    assert Path(db_path).exists()

    # Verify WAL journal mode
    with pool.get_connection() as con:
        cur = con.execute("PRAGMA journal_mode;")
        mode = cur.fetchone()[0]
        assert mode.lower() == "wal"

        # Verify query_audit table exists
        cur = con.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='query_audit';")
        assert cur.fetchone() is not None

    pool.close_all()


def test_concurrent_writes_no_locking(tmp_dir):
    db_path = str(tmp_dir / "concurrent_audit.db")
    pool = SQLiteConnectionPool(db_path, max_connections=8)
    pool.init_db()

    def worker(worker_id: int):
        with pool.get_connection() as con:
            con.execute(
                "INSERT INTO query_audit(created_at, question, source_used, trace_json, citations_json) VALUES (?, ?, ?, ?, ?)",
                (
                    "2026-01-01T00:00:00Z",
                    f"Question from worker {worker_id}",
                    "kb",
                    '["step1", "step2"]',
                    '[]',
                ),
            )
            con.commit()

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(worker, i) for i in range(40)]
        for f in concurrent.futures.as_completed(futures):
            f.result()  # Should complete without sqlite3.OperationalError: database is locked

    with pool.get_connection() as con:
        cur = con.execute("SELECT COUNT(*) FROM query_audit")
        count = cur.fetchone()[0]
        assert count == 40

    pool.close_all()


def test_write_and_read_audits(tmp_dir, monkeypatch):
    test_db = str(tmp_dir / "audit_rw.db")
    from app.services import audit
    test_pool = SQLiteConnectionPool(test_db)
    monkeypatch.setattr(audit, "_pool", test_pool)

    audit.init_db()
    audit.write_audit(
        question="What is the capital of France?",
        source_used="kb",
        trace=["cache_miss", "router_kb", "generate"],
        citations=[{"title": "Geography", "url": "https://geo.example.com"}],
    )

    records = audit.get_recent_audits(limit=5)
    assert len(records) == 1
    assert records[0]["question"] == "What is the capital of France?"
    assert records[0]["source_used"] == "kb"
    assert len(records[0]["trace"]) == 3
    assert records[0]["citations"][0]["title"] == "Geography"

    test_pool.close_all()
