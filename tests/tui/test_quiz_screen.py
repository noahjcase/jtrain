"""Quiz screen (TESTING §10)."""

from helpers import attempts, make_app, run, text_of, type_response

from jtrainer.tui.quiz import QuizScreen, SummaryScreen


def test_random_session_shows_clue(db_loaded, clock, rng):
    app = make_app(db_loaded, clock, rng)

    async def scenario(pilot):
        await pilot.press("r")
        await pilot.pause()
        screen = pilot.app.screen
        assert isinstance(screen, QuizScreen)
        clue = screen.presented.clue
        shown = text_of(pilot.app, "#clue")
        assert clue["category"] in shown
        assert ("Jeopardy!" in shown) and ("$" in shown or clue["round"] == 3)

    run(app, scenario)


def test_submit_with_confidence_writes_full_row_and_timing(db_loaded, clock, rng):
    app = make_app(db_loaded, clock, rng)

    async def scenario(pilot):
        await pilot.press("r")
        await pilot.pause()
        screen = pilot.app.screen
        response = screen.presented.clue["response"]
        clock.advance(1200)
        await type_response(pilot, response)
        clock.advance(4000)
        await pilot.press("enter")
        await pilot.pause()
        assert attempts(db_loaded) == []  # no row until confidence is given
        assert "How sure" in text_of(pilot.app, "#prompt")
        await pilot.press("2")
        await pilot.pause()
        rows = attempts(db_loaded)
        assert len(rows) == 1
        row = rows[0]
        for col in (
            "session_id",
            "clue_id",
            "shown_at",
            "first_key_ms",
            "submit_ms",
            "user_response",
            "confidence",
            "correct",
            "grader",
            "grader_detail",
            "sampling_mode",
            "sampling_prob",
        ):
            assert row[col] is not None, col
        assert row["first_key_ms"] == 1200
        assert row["submit_ms"] == 5200
        assert row["first_key_ms"] <= row["submit_ms"]
        assert row["confidence"] == 2 and row["correct"] is True

    run(app, scenario)


def test_verdict_shows_response_and_override_flips(db_loaded, clock, rng):
    app = make_app(db_loaded, clock, rng)

    async def scenario(pilot):
        await pilot.press("r")
        await pilot.pause()
        screen = pilot.app.screen
        response = screen.presented.clue["response"]
        await type_response(pilot, response)
        await pilot.press("enter", "3")
        await pilot.pause()
        verdict = text_of(pilot.app, "#verdict")
        assert "Correct" in verdict and response in verdict
        await pilot.press("o")
        await pilot.pause()
        row = attempts(db_loaded)[0]
        assert row["correct"] is False and row["overridden"] is True and row["grader"] == "user"
        assert "Incorrect" in text_of(pilot.app, "#verdict")

    run(app, scenario)


def test_pass_key_records_incorrect_without_grading(db_loaded, clock, rng):
    app = make_app(db_loaded, clock, rng)

    async def scenario(pilot):
        await pilot.press("r")
        await pilot.pause()
        clock.advance(2500)
        await pilot.press("escape")
        await pilot.pause()
        row = attempts(db_loaded)[0]
        assert row["correct"] is False and row["grader"] == "exact"
        assert row["user_response"] is None and row["confidence"] is None
        assert '"reason": "pass"' in row["grader_detail"]
        assert row["submit_ms"] == 2500

    run(app, scenario)


def test_blank_enter_passes(db_loaded, clock, rng):
    app = make_app(db_loaded, clock, rng)

    async def scenario(pilot):
        await pilot.press("r")
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        assert attempts(db_loaded)[0]["correct"] is False

    run(app, scenario)


def test_quit_mid_session_closes_session(db_loaded, clock, rng):
    app = make_app(db_loaded, clock, rng)

    async def scenario(pilot):
        await pilot.press("r")
        await pilot.pause()
        for _ in range(2):
            response = pilot.app.screen.presented.clue["response"]
            await type_response(pilot, response)
            await pilot.press("enter", "3")
            await pilot.pause()
            await pilot.press("enter")  # next clue
            await pilot.pause()
        await pilot.press("ctrl+g")
        await pilot.pause()
        assert isinstance(pilot.app.screen, SummaryScreen)
        assert "Accuracy" in text_of(pilot.app, "#summary")

    run(app, scenario)
    assert len(attempts(db_loaded)) == 2
    assert (
        db_loaded.execute("SELECT count(*) FROM sessions WHERE ended_at IS NULL").fetchone()[0] == 0
    )


def test_exit_app_mid_session_closes_session(db_loaded, clock, rng):
    app = make_app(db_loaded, clock, rng)

    async def scenario(pilot):
        await pilot.press("r")
        await pilot.pause()
        await pilot.press("escape")  # one pass
        await pilot.pause()
        pilot.app.exit()

    run(app, scenario)
    assert len(attempts(db_loaded)) == 1
    assert db_loaded.execute("SELECT ended_at IS NOT NULL FROM sessions").fetchone()[0]


def test_needs_review_asks_user(db_loaded, clock, rng):
    app = make_app(db_loaded, clock, rng, llm=None)

    async def scenario(pilot):
        await pilot.press("r")
        await pilot.pause()
        screen = pilot.app.screen
        # force a borderline response: a fuzzy partial of a multi-word response
        screen.presented.clue["response"] = "Andrew Jackson"
        await type_response(pilot, "Jackson")
        await pilot.press("enter", "2")
        await pilot.pause()
        assert "Were you right" in text_of(pilot.app, "#prompt")
        assert attempts(db_loaded)[0]["needs_review"] is True
        await pilot.press("y")
        await pilot.pause()
        row = attempts(db_loaded)[0]
        assert row["correct"] is True and row["grader"] == "user" and not row["needs_review"]

    run(app, scenario)


def test_session_ends_after_configured_length(db_loaded, clock, rng):
    app = make_app(db_loaded, clock, rng)
    app.service.config.sampling.session_length = 3

    async def scenario(pilot):
        await pilot.press("r")
        await pilot.pause()
        for _ in range(3):
            await pilot.press("escape")
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()
        assert isinstance(pilot.app.screen, SummaryScreen)

    run(app, scenario)


def test_fast_enter_then_confidence_is_not_typed_into_box(db_loaded, clock, rng):
    """'Enter, 2' pressed back to back grades with confidence 2 and never types the 2 into
    the answer box."""
    app = make_app(db_loaded, clock, rng)

    async def scenario(pilot):
        await pilot.press("r")
        await pilot.pause()
        for _ in range(5):
            screen = pilot.app.screen
            screen.query_one("#response").value = screen.presented.clue["response"]
            await pilot.press("enter", "2")  # no pause in between
            await pilot.pause()
            assert screen.state == "verdict"
            assert screen.query_one("#response").value == screen.presented.clue["response"]
            await pilot.press("enter")
            await pilot.pause()
        assert [a["confidence"] for a in attempts(db_loaded)] == [2] * 5

    run(app, scenario)
