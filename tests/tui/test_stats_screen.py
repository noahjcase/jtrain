"""Stats screen (TESTING §10): renders at 0, 1 and many attempts; values match the model."""

import numpy as np
import pytest
from conftest import make_config
from helpers import make_app, run, text_of

from jtrainer.model.stats import compute_stats
from jtrainer.tui.stats import StatsScreen


def simulate_history(app, n, seed=3):
    r = np.random.default_rng(seed)
    svc = app.service
    svc.start_session("random")
    for _ in range(n):
        p = svc.next_clue()
        right = r.random() < 0.6
        svc.submit(
            p,
            p.clue["response"] if right else "zzzz",
            int(r.integers(1, 4)),
            300,
            int(r.integers(1000, 20000)),
        )
    svc.end_session()


@pytest.mark.parametrize("n", [0, 1, 25])
def test_stats_render_and_match_model(db_loaded, clock, rng, n):
    app = make_app(db_loaded, clock, rng)
    if n:
        simulate_history(app, n)

    async def scenario(pilot):
        await pilot.press("s")
        await pilot.pause()
        screen = pilot.app.screen
        assert isinstance(screen, StatsScreen)
        direct = compute_stats(db_loaded, make_config())
        st = screen.stats
        assert st.overall == direct.overall
        assert st.brier == direct.brier
        assert [(t.topic_id, t.accuracy, t.rank) for t in st.topics] == [
            (t.topic_id, t.accuracy, t.rank) for t in direct.topics
        ]
        for a, b in zip(st.topics, direct.topics, strict=True):
            assert a.mu == pytest.approx(b.mu) and a.priority == pytest.approx(b.priority)
        summary = text_of(pilot.app, "#summary")
        assert f"attempts {n}" in summary
        assert "Brier" in summary

    run(app, scenario)
