"""Polite scrape client and merge (TESTING §12), with fake_http and the fake clock."""

import itertools

import pytest
from conftest import FIXTURES, make_config

from jtrainer.ingest.flags import Flagger
from jtrainer.ingest.jarchive import parse_game
from jtrainer.ingest.scrape import Disallowed, PoliteClient, ScrapeError, merge_game


@pytest.fixture
def client(tmp_path, clock, fake_http, db):
    cfg = make_config().scrape
    return PoliteClient(cfg, tmp_path / "cache", clock, fake_http.transport, conn=db)


def test_robots_read_first_and_crawl_delay(client, fake_http):
    client.get("/showseason.php?season=41")
    assert fake_http.paths()[0] == "/robots.txt"
    assert client.interval_s == 5.0  # Crawl-delay 5 > configured 3


def test_requests_spaced_by_interval(client, fake_http):
    for gid in (8915, 8916, 8917):
        client.get(f"/showgame.php?game_id={gid}")
    times = [t for _, t in fake_http.requests]
    gaps = [b - a for a, b in itertools.pairwise(times)]
    assert all(g >= 5000 for g in gaps), gaps


def test_cached_page_never_refetched(client, fake_http, tmp_path, clock):
    client.get("/showgame.php?game_id=8915")
    n = len(fake_http.requests)
    assert client.get("/showgame.php?game_id=8915") is not None
    assert len(fake_http.requests) == n
    # a brand-new client on the same cache makes zero requests for it
    fresh = PoliteClient(make_config().scrape, tmp_path / "cache", clock, fake_http.transport)
    fresh.get("/showgame.php?game_id=8915")
    assert len(fake_http.requests) == n


def test_disallowed_paths_never_requested(client, fake_http):
    with pytest.raises(Disallowed):
        client.get("/private/secret.php")
    assert "/private/secret.php" not in fake_http.paths()


def test_backoff_and_bounded_retries(client, fake_http, clock):
    path = "/showgame.php?game_id=8915"
    fake_http.statuses[path] = [429, 503]
    assert client.get(path) is not None
    assert fake_http.paths().count(path) == 3
    fake_http.statuses["/showgame.php?game_id=8916"] = [500, 500, 500, 500]
    with pytest.raises(ScrapeError):
        client.get("/showgame.php?game_id=8916")
    assert fake_http.paths().count("/showgame.php?game_id=8916") == 3  # 1 + max_retries(2)


def test_404_recorded_not_retried(client, fake_http, db):
    assert client.get("/showgame.php?game_id=1") is None
    assert fake_http.paths().count("/showgame.php?game_id=1") == 1
    assert (
        db.execute(
            "SELECT status FROM scrape_log WHERE url = '/showgame.php?game_id=1'"
        ).fetchone()[0]
        == 404
    )


def test_merge_dedups_and_records_outcomes(db):
    flagger = Flagger([], [])
    g = parse_game((FIXTURES / "jarchive" / "game_complete.html").read_text(), game_id=8915)
    # pre-existing jwolle1 copy of one clue (cleaned text identical)
    clue = next(c for c in g.clues if (c.round, c.col, c.row) == (1, 1, 1))
    db.execute(
        "INSERT INTO clues (source, source_ref, air_date, round, board_row, clue_value, "
        "category, clue_text, response) VALUES ('jwolle1', 'jw:x', ?, 1, 1, 200, ?, ?, ?)",
        [g.air_date, clue.category, clue.clue_text, clue.response],
    )
    counts = merge_game(db, g, flagger)
    assert counts == {"matched": 1, "inserted": 60, "outcomes": 61}
    assert db.execute("SELECT count(*) FROM clues").fetchone()[0] == 61
    assert db.execute("SELECT count(*) FROM clue_outcomes").fetchone()[0] == 61
    again = merge_game(db, g, flagger)
    assert again["inserted"] == 0 and db.execute("SELECT count(*) FROM clues").fetchone()[0] == 61
