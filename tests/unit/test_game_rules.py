"""Game simulator rules (TESTING §13)."""

import datetime as dt

import numpy as np
import pytest
from conftest import make_config

from jtrainer.game import (
    Board,
    Cell,
    GameSpec,
    GameState,
    WagerError,
    archived_game,
    check_dd_wager,
    check_final_wager,
    coryat_delta,
    dd_wager_limits,
    plays_final,
    score_delta,
    synthetic_game,
)

POST = dt.date(2015, 6, 1)
PRE = dt.date(1999, 5, 10)


def test_dd_wager_limits():
    assert dd_wager_limits(0, 1, POST) == (5, 1000)
    assert dd_wager_limits(0, 2, POST) == (5, 2000)
    assert dd_wager_limits(0, 1, PRE) == (5, 500)
    assert dd_wager_limits(0, 2, PRE) == (5, 1000)
    assert dd_wager_limits(7400, 2, POST) == (5, 7400)
    assert dd_wager_limits(-600, 1, POST) == (5, 1000)


@pytest.mark.parametrize(("wager", "ok"), [(4, False), (5, True), (2000, True), (2001, False)])
def test_dd_wager_validation(wager, ok):
    if ok:
        assert check_dd_wager(wager, 1500, 2, POST) == wager
    else:
        with pytest.raises(WagerError):
            check_dd_wager(wager, 1500, 2, POST)


def test_final_wager_rules():
    assert check_final_wager(0, 5000) == 0
    assert check_final_wager(5000, 5000) == 5000
    with pytest.raises(WagerError):
        check_final_wager(5001, 5000)
    with pytest.raises(WagerError):
        check_final_wager(-1, 5000)
    assert not plays_final(0) and not plays_final(-200) and plays_final(1)
    with pytest.raises(WagerError, match="does not play"):
        check_final_wager(0, 0)


def test_scoring():
    assert score_delta("right", 800) == 800
    assert score_delta("wrong", 800) == -800
    assert score_delta("pass", 800) == 0


def test_coryat_rules():
    assert coryat_delta("right", 400, False, False) == 400
    assert coryat_delta("wrong", 400, False, False) == -400
    assert coryat_delta("right", 1200, True, False) == 1200  # DD at board value if right
    assert coryat_delta("wrong", 1200, True, False) == 0  # zero if wrong
    assert coryat_delta("right", 0, False, True) == 0  # Final excluded


def _tiny_spec() -> GameSpec:
    """A hand-scored mini game: J! 2 clues (one DD), DJ! 1 clue, Final."""
    j = Board(
        1,
        ["A"],
        {(0, 1): Cell(1, 1, 0, 1, 200, False, "A"), (0, 2): Cell(2, 1, 0, 2, 400, True, "A")},
    )
    dj = Board(2, ["B"], {(0, 1): Cell(3, 2, 0, 1, 400, False, "B")})
    return GameSpec("test", POST, {1: j, 2: dj}, Cell(4, 3, 0, 0, 0, False, "F"))


def test_hand_scored_game():
    s = GameState(_tiny_spec())
    s.start()
    s.apply(s.spec.boards[1].cells[(0, 1)], "wrong")  # -200      coryat -200
    s.apply(s.spec.boards[1].cells[(0, 2)], "right", 1000)  # +1000     coryat +400
    assert s.round == 2
    s.apply(s.spec.boards[2].cells[(0, 1)], "right")  # +400      coryat +400
    assert s.round == 3 and not s.finished
    s.apply(s.spec.final, "wrong", 700)  # -700      coryat 0
    assert s.finished
    assert s.score == -200 + 1000 + 400 - 700 == 500
    assert s.coryat == -200 + 400 + 400 == 600


def test_no_final_at_zero_or_below():
    s = GameState(_tiny_spec())
    s.start()
    s.apply(s.spec.boards[1].cells[(0, 1)], "wrong")
    s.apply(s.spec.boards[1].cells[(0, 2)], "wrong", 5)
    s.apply(s.spec.boards[2].cells[(0, 1)], "pass")
    assert s.score < 0 and s.finished


def test_dd_requires_wager_and_pass_counts_wrong():
    s = GameState(_tiny_spec())
    s.start()
    with pytest.raises(WagerError):
        s.apply(s.spec.boards[1].cells[(0, 2)], "right")
    s.apply(s.spec.boards[1].cells[(0, 2)], "pass", 500)
    assert s.score == -500 and s.coryat == 0


def test_revise_last_after_override():
    s = GameState(_tiny_spec())
    s.start()
    s.apply(s.spec.boards[1].cells[(0, 1)], "wrong")
    s.revise_last("right")
    assert s.score == 200 and s.coryat == 200


def test_archived_game_positions_as_aired(db_loaded):
    spec = archived_game(db_loaded, dt.date(1999, 5, 10))
    j = spec.boards[1]
    assert j.categories == ["U.S. PRESIDENTS", "WORLD CAPITALS"]
    responses = {
        pos: db_loaded.execute(
            "SELECT response FROM clues WHERE clue_id = ?", [c.clue_id]
        ).fetchone()[0]
        for pos, c in j.cells.items()
    }
    assert responses[(0, 1)] == "(Abraham) Lincoln"
    assert responses[(0, 5)] == "John F. Kennedy"
    assert responses[(1, 2)] == "Australia"
    assert [j.cells[(0, r)].value for r in range(1, 6)] == [100, 200, 300, 400, 500]
    assert spec.boards[2].categories == ["THE ELEMENTS"]
    final = db_loaded.execute(
        "SELECT response FROM clues WHERE clue_id = ?", [spec.final.clue_id]
    ).fetchone()[0]
    assert final == "Salvador Dalí"


@pytest.mark.slow
def test_full_board_synthetic(db_loaded):
    """Needs a corpus with many complete categories; runs on the real DB if present."""
    from pathlib import Path

    from jtrainer import db as dbmod

    path = Path(__file__).parents[2] / "data" / "jtrainer.duckdb"
    if not path.exists():
        pytest.skip("no ingested database at data/jtrainer.duckdb")
    conn = dbmod.connect(path, read_only=True)
    spec = synthetic_game(conn, make_config(), np.random.default_rng(1))
    for rnd, n_dd in ((1, 1), (2, 2)):
        b = spec.boards[rnd]
        assert len(b.categories) == 6 and len(b.cells) == 30
        assert sum(c.is_daily_double for c in b.cells.values()) == n_dd
    assert spec.final is not None
