"""Ingest (TESTING §5): column mapping, flags, cleaning, idempotence, report."""

import datetime as dt

from conftest import SAMPLE_TSV, make_config

from jtrainer.ingest.jwolle1 import ingest_tsv


def _row(db, where: str, params=()):
    cur = db.execute(f"SELECT * FROM clues WHERE {where}", list(params))
    cols = [d[0] for d in cur.description]
    rows = cur.fetchall()
    assert len(rows) == 1, rows
    return dict(zip(cols, rows[0], strict=True))


def test_column_mapping_answer_is_clue_text(db):
    ingest_tsv(db, SAMPLE_TSV, make_config())
    row = _row(db, "clue_text = 'He was the 16th president of the United States'")
    assert row["response"] == "(Abraham) Lincoln"  # TSV 'question' column
    assert row["category"] == "U.S. PRESIDENTS"
    assert row["air_date"] == dt.date(1999, 5, 10)
    assert row["round"] == 1 and row["clue_value"] == 100 and row["board_row"] == 1
    assert row["response_key"] == "abraham lincoln"


def test_daily_double_fields(db):
    ingest_tsv(db, SAMPLE_TSV, make_config())
    row = _row(db, "response = 'Fyodor Dostoevsky'")
    assert row["is_daily_double"] is True
    assert row["dd_wager"] == 2500
    assert row["clue_value"] == 800
    assert row["board_row"] == 4  # DJ! $800 pre-2001: from board value, not the wager
    plain = _row(db, "response = 'gold'")
    assert plain["is_daily_double"] is False and plain["dd_wager"] is None


def test_off_grid_and_missing_values_counted(db):
    report = ingest_tsv(db, SAMPLE_TSV, make_config())
    assert _row(db, "response = 'an error'")["board_row"] is None
    missing = _row(db, "response = 'nothing'")
    assert missing["board_row"] is None and missing["clue_value"] is None
    assert report.board_row_missing == 2


def test_tournament_flags(db):
    ingest_tsv(db, SAMPLE_TSV, make_config())
    flagged = dict(
        db.execute(
            "SELECT response, is_tournament FROM clues WHERE category = 'ANIMALS'"
        ).fetchall()
    )
    assert flagged == {"a kitten": True, "a puppy": True, "a kid": True, "a joey": False}


def test_media_flags_and_near_miss(db):
    ingest_tsv(db, SAMPLE_TSV, make_config())
    media = dict(
        db.execute(
            "SELECT response, needs_media FROM clues WHERE air_date = DATE '2012-01-02'"
        ).fetchall()
    )
    assert media == {
        "carbon dioxide": True,
        "Vincent van Gogh": True,
        "the oboe": True,
        "Chile": True,
        "the Bounty": False,
    }


def test_cleaning(db):
    ingest_tsv(db, SAMPLE_TSV, make_config())
    responses = {r[0] for r in db.execute("SELECT response FROM clues").fetchall()}
    assert "Antônio Carlos Jobim" in responses  # HTML entity decoded
    assert '"Thriller"' in responses  # escaped quotes unescaped
    assert "Salvador Dalí" in responses  # Unicode preserved, not folded
    assert "Paris" in responses  # trailing whitespace stripped
    clue = _row(db, "response = 'Salvador Dalí'")["clue_text"]
    assert clue.startswith('His "The Persistence of Memory"')


def test_idempotent_and_dedup(db):
    first = ingest_tsv(db, SAMPLE_TSV, make_config())
    n = db.execute("SELECT count(*) FROM clues").fetchone()[0]
    assert first.skipped["duplicate_in_file"] == 1
    assert (
        db.execute(
            "SELECT count(*) FROM clues WHERE clue_text = 'This island nation is also a continent'"
        ).fetchone()[0]
        == 1
    )
    # same clue text with different responses are distinct clues
    assert (
        db.execute("SELECT count(*) FROM clues WHERE clue_text = 'What''s this?'").fetchone()[0]
        == 2
    )
    second = ingest_tsv(db, SAMPLE_TSV, make_config())
    assert db.execute("SELECT count(*) FROM clues").fetchone()[0] == n
    assert second.inserted == 0
    assert second.skipped["already_present"] == n


def test_report_balances(db):
    report = ingest_tsv(db, SAMPLE_TSV, make_config())
    assert report.rows_read == 54
    assert report.inserted + sum(report.skipped.values()) == report.rows_read
    assert report.inserted == db.execute("SELECT count(*) FROM clues").fetchone()[0]


def test_special_file_skipped_by_default_and_round_mapping(db, tmp_path):
    special = tmp_path / "extra.tsv"
    special.write_text(
        "round\tclue_value\tdaily_double_value\tcategory\tcomments\tanswer\tquestion\t"
        "air_date\tnotes\n"
        "single\t100\t0\tCAT\t\tA clue\tan answer\t1983-09-18\tpilot\n"
        "triple\t300\t0\tCAT\t\tA triple clue\tanother\t2008-01-01\tspecial\n"
    )
    report = ingest_tsv(db, special, make_config(), special=True)
    assert report.inserted == 1 and report.skipped["unsupported_round"] == 1
    assert _row(db, "response = 'an answer'")["is_special"] is True
