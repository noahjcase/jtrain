"""Tag review screen (TESTING §10, milestone 2)."""

from helpers import make_app, run
from textual.widgets import Input, OptionList

from jtrainer.tui.tagreview import TagReviewScreen


def reviews(db):
    return db.execute(
        "SELECT clue_id, action, original_topic, new_topic, note FROM tag_reviews "
        "ORDER BY reviewed_at, clue_id"
    ).fetchall()


def test_accept_change_flag(db_loaded, clock, rng):
    app = make_app(db_loaded, clock, rng)
    seen = []

    async def scenario(pilot):
        await pilot.press("t")
        await pilot.pause()
        screen = pilot.app.screen
        assert isinstance(screen, TagReviewScreen)
        first = screen.current
        seen.append(first)
        await pilot.press("a")
        await pilot.pause()

        second = screen.current
        seen.append(second)
        await pilot.press("c")
        await pilot.pause()
        picker = pilot.app.screen.query_one(OptionList)
        idx = next(
            i for i in range(picker.option_count) if picker.get_option_at_index(i).id == "bible"
        )
        picker.highlighted = idx
        await pilot.press("enter")
        await pilot.pause()

        third = screen.current
        seen.append(third)
        await pilot.press("f")
        await pilot.pause()
        pilot.app.screen.query_one(Input).value = "needs a card-games topic"
        await pilot.press("enter")
        await pilot.pause()

    run(app, scenario)
    first, second, third = seen
    rows = reviews(db_loaded)
    assert rows[0] == (first["clue_id"], "accept", first["topic_id"], first["topic_id"], None)
    assert rows[1] == (second["clue_id"], "change", second["topic_id"], "bible", None)
    assert rows[2] == (
        third["clue_id"],
        "flag",
        third["topic_id"],
        None,
        "needs a card-games topic",
    )
    human = db_loaded.execute(
        "SELECT topic_id, model FROM clue_tags WHERE clue_id = ? AND model = 'human'",
        [second["clue_id"]],
    ).fetchall()
    assert human == [("bible", "human")]
