"""Fast IRT checks (the recovery simulations live in tests/sim, marked slow)."""

import numpy as np
from conftest import make_config

from jtrainer.model.irt import (
    OUTCOMES,
    fit_difficulty,
    fit_from_db,
    fit_user_ability,
    outcome_index,
    outcome_probs,
    simulate_outcomes,
)


def test_outcome_index_covers_all():
    assert outcome_index(1, 0) == 0
    assert outcome_index(0, 3) == len(OUTCOMES) - 1
    assert outcome_index(2, 5) == outcome_index(1, 2)  # clipped


def test_probabilities_monotone_in_difficulty():
    p = np.array([0.9, 0.5, 0.1])
    probs = outcome_probs(p, 0.8, 0.1)
    anyone_right = probs[:, :3].sum(axis=1)
    assert np.all(np.diff(anyone_right) < 0)


def test_small_fit_runs_and_orders_levels():
    rng = np.random.default_rng(0)
    level = rng.integers(0, 11, 3000)
    b = np.linspace(-1.5, 1.5, 11)[level]
    nc, ni = simulate_outcomes(b, 0.8, 0.1, rng)
    outcome = np.array([outcome_index(c, i) for c, i in zip(nc, ni, strict=True)])
    fit = fit_difficulty(level, outcome)
    assert np.corrcoef(fit.level_mu, np.linspace(-1.5, 1.5, 11))[0, 1] > 0.9


def test_user_ability_small():
    fit = fit_user_ability(
        ["a"] * 20 + ["b"] * 20, np.array([1] * 15 + [0] * 5 + [0] * 15 + [1] * 5), np.zeros(40)
    )
    a, b = fit.theta[fit.topics.index("a")], fit.theta[fit.topics.index("b")]
    assert a > fit.theta_global > b


def test_fit_from_db_stores_difficulties(db_loaded, clock):
    cfg = make_config()
    assert fit_from_db(db_loaded, cfg, clock) is None  # no outcomes yet
    ids = [r[0] for r in db_loaded.execute("SELECT clue_id FROM clues").fetchall()]
    rng = np.random.default_rng(1)
    for cid in ids:
        nc = int(rng.random() < 0.7)
        db_loaded.execute(
            "INSERT INTO clue_outcomes VALUES (?, ?, ?, ?)",
            [cid, nc, int(rng.integers(0, 2)), nc == 0],
        )
    fit = fit_from_db(db_loaded, cfg, clock)
    assert fit is not None
    n = db_loaded.execute("SELECT count(*) FROM clue_difficulty").fetchone()[0]
    assert n == len(ids)
    # fitted difficulties now feed the ability model and samplers
    from jtrainer.sampling import Pool

    pool = Pool.from_db(db_loaded, cfg)
    stored = dict(db_loaded.execute("SELECT clue_id, b_mean FROM clue_difficulty").fetchall())
    i = 0
    assert pool.b[i] == stored[int(pool.ids[i])]
