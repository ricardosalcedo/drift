"""SQLite database: schema, migrations, prompt versioning, response cache."""
import hashlib, sqlite3
from datetime import datetime, timezone, timedelta
from pathlib import Path

DB_PATH = Path("drift.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY, timestamp TEXT NOT NULL, model TEXT NOT NULL,
    prompt_id TEXT NOT NULL, prompt_version TEXT DEFAULT '', prompt TEXT NOT NULL,
    response TEXT NOT NULL, score REAL, judge_reason TEXT, latency_ms INTEGER,
    tokens_in INTEGER DEFAULT 0, tokens_out INTEGER DEFAULT 0,
    cost_usd REAL, tags TEXT DEFAULT '', cached INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS prompt_versions (
    id INTEGER PRIMARY KEY, prompt_id TEXT NOT NULL, version TEXT NOT NULL,
    prompt_text TEXT NOT NULL, criteria TEXT, created_at TEXT NOT NULL,
    UNIQUE(prompt_id, version)
);
CREATE TABLE IF NOT EXISTS cache (
    id INTEGER PRIMARY KEY, prompt_hash TEXT NOT NULL, model TEXT NOT NULL,
    response TEXT NOT NULL, score REAL, judge_reason TEXT, latency_ms INTEGER,
    tokens_in INTEGER DEFAULT 0, tokens_out INTEGER DEFAULT 0,
    created_at TEXT NOT NULL, UNIQUE(prompt_hash, model)
);
CREATE INDEX IF NOT EXISTS idx_runs_model_prompt ON runs(model, prompt_id);
CREATE INDEX IF NOT EXISTS idx_runs_ts ON runs(timestamp);
CREATE INDEX IF NOT EXISTS idx_cache_hash ON cache(prompt_hash, model);
"""

_MIGRATIONS = [
    ("tokens_in", "INTEGER DEFAULT 0"), ("tokens_out", "INTEGER DEFAULT 0"),
    ("cost_usd", "REAL"), ("tags", "TEXT DEFAULT ''"),
    ("prompt_version", "TEXT DEFAULT ''"), ("cached", "INTEGER DEFAULT 0"),
]


def connect():
    """Open DB connection, create schema, run migrations."""
    db = sqlite3.connect(str(DB_PATH))
    db.row_factory = sqlite3.Row
    db.executescript(SCHEMA)
    cols = {r[1] for r in db.execute("PRAGMA table_info(runs)").fetchall()}
    for col, typedef in _MIGRATIONS:
        if col not in cols:
            db.execute(f"ALTER TABLE runs ADD COLUMN {col} {typedef}")
    return db


def get_prompt_version(db, prompt_id, prompt_text, criteria):
    """Get or create a version string for a prompt. Auto-increments on text change."""
    existing = db.execute(
        "SELECT version, prompt_text, criteria FROM prompt_versions WHERE prompt_id=? ORDER BY created_at DESC LIMIT 1",
        (prompt_id,)
    ).fetchone()
    if existing:
        if existing["prompt_text"] == prompt_text and (existing["criteria"] or "") == (criteria or ""):
            return existing["version"]
        ver_num = int(existing["version"].lstrip("v")) + 1
    else:
        ver_num = 1
    version = f"v{ver_num}"
    db.execute(
        "INSERT OR IGNORE INTO prompt_versions (prompt_id, version, prompt_text, criteria, created_at) VALUES (?,?,?,?,?)",
        (prompt_id, version, prompt_text, criteria, _now())
    )
    db.commit()
    return version


def cache_get(db, model, prompt_text, max_age_hours=24):
    """Return cached response if fresh, else None."""
    h = hashlib.sha256(f"{model}|{prompt_text}".encode()).hexdigest()
    row = db.execute("SELECT * FROM cache WHERE prompt_hash=? AND model=?", (h, model)).fetchone()
    if not row:
        return None
    if datetime.now(timezone.utc) - datetime.fromisoformat(row["created_at"]) > timedelta(hours=max_age_hours):
        return None
    return row


def cache_set(db, model, prompt_text, response, score, reason, latency, tok_in, tok_out):
    """Upsert a response into cache."""
    h = hashlib.sha256(f"{model}|{prompt_text}".encode()).hexdigest()
    db.execute(
        "INSERT OR REPLACE INTO cache (prompt_hash,model,response,score,judge_reason,latency_ms,tokens_in,tokens_out,created_at) VALUES (?,?,?,?,?,?,?,?,?)",
        (h, model, response, score, reason, latency, tok_in, tok_out, _now())
    )


def _now():
    return datetime.now(timezone.utc).isoformat()
