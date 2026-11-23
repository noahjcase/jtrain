"""Migrations and database constraints (TESTING §11)."""

import duckdb
import pytest

from jtrainer import db as dbmod
from jtrainer.service import AttemptRepository

LATEST = len(dbmod.migrations())


def test_apply_to_empty_and_rerun_is_noop():
    conn = duckdb.connect()
    assert dbmod.migrate(conn) == list(range(1, LATEST + 1))
    assert dbmod.migrate(conn) == []
    assert dbmod.schema_version(conn) == LATEST


def _seed_v1(conn):
    conn.execute(
        "INSERT INTO clues (source, source_ref, round, clue_text, response) "
        "VALUES ('jwolle1', 'r1', 1, 'clue', 'resp')"
    )
    conn.execute(
        "INSERT INTO sessions (started_at, mode, app_version) VALUES "
        "('2026-01-01', 'random', '0.0.1')"
    )
    conn.execute(
        "INSERT INTO attempts (session_id, clue_id, shown_at, correct, grader, "
        "sampling_mode, sampling_prob) VALUES (1, 1, '2026-01-01', TRUE, 'exact', "
        "'random', 0.5)"
    )


@pytest.mark.parametrize("version", range(1, LATEST))
def test_each_historical_version_upgrades(version):
    """A database left at each released schema version upgrades to current with its data."""
    conn = duckdb.connect()
    dbmod.migrate(conn, target=version)
    assert dbmod.schema_version(conn) == version
    _seed_v1(conn)
    if version >= 2:
        conn.execute("INSERT INTO taxonomies VALUES ('v1', '2026-01-01', '')")
        conn.execute("INSERT INTO entities (canonical_name, canonical_key) VALUES ('X', 'x')")
        conn.execute("INSERT INTO clue_tags VALUES (1, 'v1', 'p', 'm', 't', NULL, 1, '2026-01-01')")
    applied = dbmod.migrate(conn)
    assert applied == list(range(version + 1, LATEST + 1))
    assert conn.execute("SELECT count(*) FROM attempts").fetchone()[0] == 1
    assert conn.execute("SELECT response FROM clues").fetchone()[0] == "resp"
    if version >= 2:
        assert conn.execute("SELECT count(*) FROM clue_tags").fetchone()[0] == 1


@pytest.mark.parametrize(
    ("sql", "params"),
    [
        (
            "INSERT INTO clues (source, source_ref, round, clue_text, response) VALUES "
            "('x', 'a', 4, 'c', 'r')",
            [],
        ),
        (
            "INSERT INTO clues (source, source_ref, round, board_row, clue_text, response) VALUES "
            "('x', 'b', 1, 6, 'c', 'r')",
            [],
        ),
        (
            "INSERT INTO attempts (session_id, clue_id, shown_at, confidence, sampling_mode) "
            "VALUES (1, 1, '2026-01-01', 4, 'random')",
            [],
        ),
        (
            "INSERT INTO attempts (session_id, clue_id, shown_at, sampling_mode) VALUES "
            "(1, 999, '2026-01-01', 'random')",
            [],
        ),
        (
            "INSERT INTO attempts (session_id, clue_id, shown_at, first_key_ms, submit_ms, "
            "sampling_mode) VALUES (1, 1, '2026-01-01', 500, 400, 'random')",
            [],
        ),
        (
            "INSERT INTO sessions (started_at, mode, app_version) VALUES ('2026-01-01', 'x', 'v')",
            [],
        ),
    ],
)
def test_constraints(sql, params):
    conn = dbmod.connect(":memory:")
    _seed_v1(conn)
    with pytest.raises(duckdb.ConstraintException):
        conn.execute(sql, params)


def test_board_row_null_allowed():
    conn = dbmod.connect(":memory:")
    conn.execute(
        "INSERT INTO clues (source, source_ref, round, board_row, clue_text, response) "
        "VALUES ('x', 'f', 3, NULL, 'c', 'r')"
    )


def test_attempt_repository_has_no_delete():
    names = [n for n in dir(AttemptRepository) if not n.startswith("__")]
    assert not [n for n in names if "delete" in n.lower() or "remove" in n.lower()]
