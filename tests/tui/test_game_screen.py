"""Game screen: play an archived board through the TUI."""

import datetime as dt

from helpers import make_app, run, text_of
from textual.widgets import Input

from jtrainer.game import archived_game
from jtrainer.tui.game import GameScreen


def test_play_clue_and_daily_double_wager(db_loaded, clock, rng):
    app = make_app(db_loaded, clock, rng)
    spec = archived_game(db_loaded, dt.date(1999, 5, 10))

    async def scenario(pilot):
        pilot.app.push_screen(GameScreen(pilot.app.service, spec))
        await pilot.pause()
        screen = pilot.app.screen
        assert screen.phase == "board" and screen.cursor == (0, 1)
        await pilot.press("enter")
        await pilot.pause()
        assert screen.phase == "answering"
        screen.query_one("#entry", Input).value = "Lincoln"
        await pilot.press("enter", "3")
        await pilot.pause()
        assert screen.state.score == 100 and screen.state.coryat == 100
        assert "+$100" in text_of(pilot.app, "#main")
        await pilot.press("enter")
        await pilot.pause()
        await pilot.press("down", "enter")  # $200 John Adams: pass
        await pilot.pause()
        await pilot.press("escape")
        await pilot.pause()
        assert screen.state.score == 100
        await pilot.press("ctrl+g")
        await pilot.pause()
        assert screen.phase == "done" and "Coryat" in text_of(pilot.app, "#main")

    run(app, scenario)
    game = db_loaded.execute("SELECT final_score, coryat FROM games").fetchone()
    assert game == (100, 100)
    outcomes = db_loaded.execute(
        "SELECT outcome, score_delta FROM game_clues ORDER BY attempt_id"
    ).fetchall()
    assert outcomes == [("right", 100), ("pass", 0)]
    modes = {r[0] for r in db_loaded.execute("SELECT sampling_mode FROM attempts").fetchall()}
    assert modes == {"game:archive"}


def test_daily_double_wager_validation(db_loaded, clock, rng):
    app = make_app(db_loaded, clock, rng)
    spec = archived_game(db_loaded, dt.date(1999, 5, 10))
    # make the $300 clue a Daily Double for this test
    spec.boards[1].cells[(0, 3)].is_daily_double = True

    async def scenario(pilot):
        pilot.app.push_screen(GameScreen(pilot.app.service, spec))
        await pilot.pause()
        screen = pilot.app.screen
        await pilot.press("down", "down", "enter")
        await pilot.pause()
        assert screen.phase == "wager"
        screen.query_one("#entry", Input).value = "9999"
        await pilot.press("enter")
        await pilot.pause()
        assert screen.phase == "wager" and "between" in text_of(pilot.app, "#prompt")
        screen.query_one("#entry", Input).value = "500"
        await pilot.press("enter")
        await pilot.pause()
        assert screen.phase == "answering"
        screen.query_one("#entry", Input).value = "John Quincy Adams"
        await pilot.press("enter", "3")
        await pilot.pause()
        assert screen.state.score == 500 and screen.state.coryat == 300

    run(app, scenario)
