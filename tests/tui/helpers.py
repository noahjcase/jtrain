"""Helpers for driving the Textual app with Pilot under plain pytest."""

import asyncio

from conftest import make_config
from textual.widgets import Input, Static

from jtrainer.tui.app import JTrainApp

SIZE = (100, 30)


def make_app(db, clock, rng, llm=None) -> JTrainApp:
    return JTrainApp(db, make_config(), clock, rng, llm)


def run(app, scenario):
    async def main():
        async with app.run_test(size=SIZE) as pilot:
            await pilot.pause()
            await scenario(pilot)

    asyncio.run(main())


def text_of(app, selector: str) -> str:
    return str(app.screen.query_one(selector, Static).render())


def attempts(db) -> list[dict]:
    cur = db.execute("SELECT * FROM attempts ORDER BY attempt_id")
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, r, strict=True)) for r in cur.fetchall()]


async def type_response(pilot, text: str) -> None:
    """Type the first character as a keystroke (for first-key timing), then set the rest."""
    await pilot.press(text[0] if text[0].isalnum() else "x")
    inp = pilot.app.screen.query_one("#response", Input)
    inp.value = text
    await pilot.pause()
