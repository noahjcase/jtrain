"""DuckDB connection and numbered migrations.

Timestamps are stored as naive UTC TIMESTAMPs; use `to_db` / `from_db` at the boundary.
"""

from __future__ import annotations

import datetime as dt
import re
from importlib import resources
from pathlib import Path

import duckdb

UTC = dt.UTC
_MIGRATION_RE = re.compile(r"^(\d{3})_[a-z0-9_]+\.sql$")


def migrations() -> list[tuple[int, str, str]]:
    """All migrations as (version, name, sql), ordered by version."""
    pkg = resources.files("jtrainer") / "migrations"
    found = []
    for entry in pkg.iterdir():
        m = _MIGRATION_RE.match(entry.name)
        if m:
            found.append((int(m.group(1)), entry.name, entry.read_text()))
    found.sort()
    versions = [v for v, _, _ in found]
    if versions != list(range(1, len(versions) + 1)):
        raise RuntimeError(f"Migration numbering has gaps or duplicates: {versions}")
    return found


def schema_version(conn: duckdb.DuckDBPyConnection) -> int:
    exists = conn.execute(
        "SELECT count(*) FROM information_schema.tables WHERE table_name = 'schema_version'"
    ).fetchone()[0]
    if not exists:
        return 0
    row = conn.execute("SELECT max(version) FROM schema_version").fetchone()
    return row[0] or 0


def migrate(conn: duckdb.DuckDBPyConnection, target: int | None = None) -> list[int]:
    """Apply pending migrations up to `target` (default: latest). Returns versions applied."""
    conn.execute(
        "CREATE TABLE IF NOT EXISTS schema_version ("
        " version INTEGER PRIMARY KEY, name VARCHAR NOT NULL, applied_at TIMESTAMP NOT NULL)"
    )
    current = schema_version(conn)
    applied = []
    for version, name, sql in migrations():
        if version <= current or (target is not None and version > target):
            continue
        conn.execute("BEGIN TRANSACTION")
        try:
            conn.execute(sql)
            conn.execute(
                "INSERT INTO schema_version VALUES (?, ?, ?)",
                [version, name, to_db(dt.datetime.now(UTC))],
            )
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
        applied.append(version)
    return applied


def connect(path: str | Path = ":memory:", *, read_only: bool = False) -> duckdb.DuckDBPyConnection:
    if path != ":memory:":
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(str(path), read_only=read_only)
    if not read_only:
        migrate(conn)
    return conn


def to_db(ts: dt.datetime | None) -> dt.datetime | None:
    if ts is None:
        return None
    if ts.tzinfo is not None:
        ts = ts.astimezone(UTC).replace(tzinfo=None)
    return ts


def from_db(ts: dt.datetime | None) -> dt.datetime | None:
    if ts is None:
        return None
    return ts.replace(tzinfo=UTC) if ts.tzinfo is None else ts


def rows_as_dicts(cursor: duckdb.DuckDBPyConnection) -> list[dict]:
    cols = [d[0] for d in cursor.description]
    return [dict(zip(cols, row, strict=True)) for row in cursor.fetchall()]
