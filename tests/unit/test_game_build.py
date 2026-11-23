"""Synthetic boards and game recording (fast, on generated data)."""

import datetime as dt

import numpy as np
from conftest import make_config

from jtrainer.game import (
    GameRecorder,
    GameState,
    archived_game,
    list_archived_games,
    synthetic_game,
)


def seed_categories(db, n=7):
    k = 0
    for rnd, values in ((1, [200, 400, 600, 800, 1000]), (2, [400, 800, 1200, 1600, 2000])):
        for c in range(n):
            for row, v in enumerate(values, start=1):
                k += 1
                db.execute(
                    "INSERT INTO clues (source, source_ref, air_date, round, board_row, "
                    "clue_value, category, clue_text, response) VALUES ('t', ?, ?, ?, ?, "
                    "?, ?, ?, ?)",
                    [
                        f"r{k}",
                        dt.date(2010, 1, 1 + c),
                        rnd,
                        row,
                        v,
                        f"CAT{rnd}-{c}",
                        f"clue {k}",
                        f"resp {k}",
                    ],
                )
    db.execute(
        "INSERT INTO clues (source, source_ref, air_date, round, category, clue_text, "
        "response) VALUES ('t', 'final', '2010-01-01', 3, 'FINAL', 'fc', 'fr')"
    )


def test_synthetic_board_shape(db):
    seed_categories(db)
    spec = synthetic_game(db, make_config(), np.random.default_rng(3))
    for rnd, n_dd in ((1, 1), (2, 2)):
        board = spec.boards[rnd]
        assert len(board.categories) == 6 and len(board.cells) == 30
        assert sum(c.is_daily_double for c in board.cells.values()) == n_dd
        dd_cols = [c.col for c in board.cells.values() if c.is_daily_double]
        assert len(set(dd_cols)) == len(dd_cols)
        assert [board.cells[(0, r)].value for r in range(1, 6)] == (
            [200, 400, 600, 800, 1000] if rnd == 1 else [400, 800, 1200, 1600, 2000]
        )
    assert spec.final is not None and spec.source == "synthetic"


def test_recorder_and_history(db_loaded, clock):
    spec = archived_game(db_loaded, dt.date(1999, 5, 10), choice_prob=0.5)
    db_loaded.execute(
        "INSERT INTO sessions (started_at, mode, app_version) VALUES ('2026-01-01', 'game', 't')"
    )
    rec = GameRecorder(db_loaded, clock)
    gid = rec.start(1, spec)
    state = GameState(spec)
    state.start()
    cell = spec.boards[1].cells[(0, 1)]
    state.apply(cell, "right")
    rec.record(gid, 1, state.log[-1])
    state.skip_round()
    assert state.round == 2
    rec.finish(gid, state)
    assert rec.history() == [(gid, "archive:1999-05-10", rec.history()[0][2], 100, 100)]


def test_list_archived_games(db_loaded):
    assert list_archived_games(db_loaded, make_config(), min_clues=10) == [
        (dt.date(1999, 5, 10), 13),
        (dt.date(2001, 11, 26), 11),
    ]
    assert list_archived_games(db_loaded, make_config(), min_clues=12) == [
        (dt.date(1999, 5, 10), 13)
    ]
