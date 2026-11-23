"""Review screen (TESTING §10)."""

from helpers import attempts, make_app, run, text_of

from jtrainer.tui.quiz import QuizScreen


def test_empty_queue_shows_empty_state(db_loaded, clock, rng):
    app = make_app(db_loaded, clock, rng)

    async def scenario(pilot):
        await pilot.press("v")
        await pilot.pause()
        assert isinstance(pilot.app.screen, QuizScreen)
        assert "Nothing due" in text_of(pilot.app, "#clue")

    run(app, scenario)


def test_shows_only_due_items(db_loaded, clock, rng):
    app = make_app(db_loaded, clock, rng)
    svc = app.service
    svc.start_session("random")
    missed = []
    for _ in range(3):
        p = svc.next_clue()
        svc.submit(p, "zzzz", 2, 100, 900)
        missed.append(p.clue["clue_id"])
    svc.end_session()
    clock.advance(40 * 86_400_000)
    due_entities = {d.entity_id for d in svc.review.due()}
    assert len(due_entities) == 3

    async def scenario(pilot):
        await pilot.press("v")
        await pilot.pause()
        for _ in range(3):
            screen = pilot.app.screen
            assert screen.presented is not None and screen.presented.pick.mode == "review"
            assert screen.presented.pick.entity_id in due_entities
            await pilot.press("escape")
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()

    run(app, scenario)
    review_rows = [a for a in attempts(db_loaded) if a["sampling_mode"] == "review"]
    assert len(review_rows) == 3
