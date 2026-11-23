"""Ability v1 recovery by simulation (TESTING §8.1). Marked slow."""

import datetime as dt

import numpy as np
import pytest
from conftest import make_config
from scipy import special, stats

from jtrainer.config import Ability
from jtrainer.model.ability import AbilityModel, level_difficulties, replay

pytestmark = pytest.mark.slow


def simulate(seed: int, n_topics: int = 10, n_attempts: int = 3000):
    rng = np.random.default_rng(seed)
    cfg = make_config()
    levels = np.array(level_difficulties(cfg.difficulty))
    # Evenly spread abilities (shuffled): with ~300 attempts per topic the standard error is
    # ~0.13, so truths closer than that could not be ranked by any estimator.
    true = rng.permutation(np.linspace(-1.8, 1.8, n_topics))
    topics = rng.integers(0, n_topics, n_attempts)
    lvl = rng.integers(0, len(levels), n_attempts)
    b = levels[lvl]
    y = rng.random(n_attempts) < special.expit(true[topics] - b)
    model = AbilityModel(Ability(prior_mean=0.0, prior_sd=1.5, drift_sd_per_day=0.0))
    for t, bb, yy in zip(topics, b, y, strict=True):
        model.update(f"t{t}", float(bb), bool(yy))
    return true, model, topics, lvl, y


def test_recovery_rank_and_coverage():
    inside = total = 0
    for seed in range(5):
        true, model, *_ = simulate(seed)
        est = np.array([model.get(f"t{i}").mu for i in range(len(true))])
        assert stats.spearmanr(true, est).statistic >= 0.9
        for i, th in enumerate(true):
            lo, hi = model.get(f"t{i}").interval(1.645)
            inside += lo <= th <= hi
            total += 1
    assert inside / total >= 0.85, inside / total


def test_reproducible_from_attempts_table(db):
    cfg = make_config()
    cfg.ability = Ability(prior_mean=0.0, prior_sd=1.5, drift_sd_per_day=0.0)
    true, model, topics, lvl, y = simulate(0, n_attempts=3000)
    db.execute("INSERT INTO taxonomies VALUES ('test-v1', '2026-01-01', '')")
    rows_by_level = [(1, r) for r in range(1, 6)] + [(2, r) for r in range(1, 6)] + [(3, None)]
    for i, (rnd, row) in enumerate(rows_by_level):
        for t in range(len(true)):
            cid = i * 100 + t + 1
            db.execute(
                "INSERT INTO clues (clue_id, source, source_ref, round, board_row, "
                "clue_text, response) VALUES (?, 'sim', ?, ?, ?, 'c', 'r')",
                [cid, f"s{cid}", rnd, row],
            )
            db.execute(
                "INSERT INTO clue_tags VALUES (?, 'test-v1', 'p', 'm', ?, NULL, NULL, "
                "'2026-01-01')",
                [cid, f"t{t}"],
            )
    db.execute(
        "INSERT INTO sessions (started_at, mode, app_version) VALUES "
        "('2026-01-01', 'random', 'sim')"
    )
    start = dt.datetime(2026, 1, 1)
    db.executemany(
        "INSERT INTO attempts (session_id, clue_id, shown_at, correct, grader, sampling_mode, "
        "sampling_prob) VALUES (1, ?, ?, ?, 'exact', 'random', 0.01)",
        [
            [int(lv) * 100 + int(t) + 1, start + dt.timedelta(seconds=k), bool(yy)]
            for k, (t, lv, yy) in enumerate(zip(topics, lvl, y, strict=True))
        ],
    )
    rebuilt, report = replay(db, cfg)
    assert report.used == 3000
    for i in range(len(true)):
        assert rebuilt.get(f"t{i}").mu == pytest.approx(model.get(f"t{i}").mu, abs=1e-12)
        assert rebuilt.get(f"t{i}").var == pytest.approx(model.get(f"t{i}").var, abs=1e-12)
