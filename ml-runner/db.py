"""
SQLite-backed job tracking for ml-runner.

Stores a durable record of every generate/train job. The job_runner pairs each
row with an in-memory subprocess handle so the poller thread can update the row
when the process exits.

Schema:
    jobs(
        id            TEXT PRIMARY KEY,   -- uuid
        type          TEXT,               -- 'generate' | 'train'
        set_name      TEXT,
        status        TEXT,               -- 'queued'|'running'|'complete'|'failed'
        created_at    TEXT,               -- ISO 8601
        started_at    TEXT,
        finished_at   TEXT,
        result        TEXT,               -- generated text (generate) or sample (train)
        error         TEXT,               -- error message on failure
        log_path      TEXT,               -- path to the captured subprocess log
        params_json   TEXT                -- JSON blob of the original request params
    )
"""
import json
import os
import sqlite3
import threading
import uuid
from datetime import datetime, timezone

# Module-level lock so concurrent Flask threads don't fight over the connection.
_lock = threading.Lock()


def _now():
    return datetime.now(timezone.utc).isoformat()


def db_path(data_dir):
    return os.path.join(data_dir, "jobs.db")


def _connect(data_dir):
    conn = sqlite3.connect(db_path(data_dir), timeout=30.0)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(data_dir):
    """Create the jobs table if it doesn't exist. Safe to call on every startup."""
    with _lock:
        conn = _connect(data_dir)
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    id          TEXT PRIMARY KEY,
                    type        TEXT NOT NULL,
                    set_name    TEXT NOT NULL,
                    status      TEXT NOT NULL,
                    created_at  TEXT NOT NULL,
                    started_at  TEXT,
                    finished_at TEXT,
                    result      TEXT,
                    error       TEXT,
                    log_path    TEXT,
                    params_json TEXT
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_jobs_type ON jobs(type)"
            )
            conn.commit()
        finally:
            conn.close()


def create_job(data_dir, job_type, set_name, params, log_path):
    """Insert a fresh 'queued' job row and return its id."""
    job_id = str(uuid.uuid4())
    with _lock:
        conn = _connect(data_dir)
        try:
            conn.execute(
                """
                INSERT INTO jobs
                    (id, type, set_name, status, created_at, log_path, params_json)
                VALUES (?, ?, ?, 'queued', ?, ?, ?)
                """,
                (
                    job_id,
                    job_type,
                    set_name,
                    _now(),
                    log_path,
                    json.dumps(params, default=str),
                ),
            )
            conn.commit()
        finally:
            conn.close()
    return job_id


def mark_running(data_dir, job_id):
    with _lock:
        conn = _connect(data_dir)
        try:
            conn.execute(
                "UPDATE jobs SET status='running', started_at=? WHERE id=?",
                (_now(), job_id),
            )
            conn.commit()
        finally:
            conn.close()


def mark_complete(data_dir, job_id, result):
    with _lock:
        conn = _connect(data_dir)
        try:
            conn.execute(
                "UPDATE jobs SET status='complete', finished_at=?, result=? WHERE id=?",
                (_now(), result, job_id),
            )
            conn.commit()
        finally:
            conn.close()


def mark_failed(data_dir, job_id, error):
    with _lock:
        conn = _connect(data_dir)
        try:
            conn.execute(
                "UPDATE jobs SET status='failed', finished_at=?, error=? WHERE id=?",
                (_now(), error, job_id),
            )
            conn.commit()
        finally:
            conn.close()


def mark_interrupted_running(data_dir):
    """
    On ml-runner startup, any job left in 'running' from a previous process
    can't be reattached. Mark it failed so callers see a clean state.
    """
    with _lock:
        conn = _connect(data_dir)
        try:
            conn.execute(
                """
                UPDATE jobs
                SET status='failed',
                    finished_at=?,
                    error='interrupted: ml-runner restarted while job was running'
                WHERE status='running'
                """,
                (_now(),),
            )
            conn.commit()
        finally:
            conn.close()


def get_job(data_dir, job_id):
    with _lock:
        conn = _connect(data_dir)
        try:
            row = conn.execute(
                "SELECT * FROM jobs WHERE id=?", (job_id,)
            ).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()


def list_jobs(data_dir, job_type=None, status=None, limit=100):
    query = "SELECT * FROM jobs"
    clauses = []
    params = []
    if job_type:
        clauses.append("type=?")
        params.append(job_type)
    if status:
        clauses.append("status=?")
        params.append(status)
    if clauses:
        query += " WHERE " + " AND ".join(clauses)
    query += " ORDER BY created_at DESC LIMIT ?"
    params.append(limit)
    with _lock:
        conn = _connect(data_dir)
        try:
            rows = conn.execute(query, params).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()