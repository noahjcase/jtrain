"""TUI snapshots at a fixed size and fixed data (TESTING §10). Snapshot changes are
reviewed in the diff, never blindly regenerated."""

import pytest
from helpers import SIZE, make_app

pytestmark = pytest.mark.snapshot


def test_home(snap_compare, db_loaded, clock, rng):
    assert snap_compare(make_app(db_loaded, clock, rng), terminal_size=SIZE)


def test_quiz(snap_compare, db_loaded, clock, rng):
    assert snap_compare(make_app(db_loaded, clock, rng), terminal_size=SIZE, press=["r"])


def test_verdict(snap_compare, db_loaded, clock, rng):
    async def answer(pilot):
        await pilot.press("r")
        await pilot.pause()
        screen = pilot.app.screen
        screen.query_one("#response").value = screen.presented.clue["response"]
        await pilot.press("enter", "3")
        await pilot.pause()

    assert snap_compare(make_app(db_loaded, clock, rng), terminal_size=SIZE, run_before=answer)


def test_review_empty(snap_compare, db_loaded, clock, rng):
    assert snap_compare(make_app(db_loaded, clock, rng), terminal_size=SIZE, press=["v"])


def test_stats(snap_compare, db_loaded, clock, rng):
    app = make_app(db_loaded, clock, rng)
    svc = app.service
    svc.start_session("random")
    for i in range(6):
        p = svc.next_clue()
        svc.submit(p, p.clue["response"] if i % 3 else "zzzz", 1 + i % 3, 400, 2000 + 2500 * i)
    svc.end_session()
    assert snap_compare(app, terminal_size=SIZE, press=["s"])
