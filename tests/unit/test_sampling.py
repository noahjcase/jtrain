"""Sampling (TESTING §9)."""

import math
from collections import Counter

import numpy as np
import pytest
from conftest import make_config
from scipy import stats

from jtrainer.config import Ability, Eligibility
from jtrainer.model.ability import AbilityModel
from jtrainer.sampling import AdaptiveSampler, Pool, RandomSampler, ReviewSampler

MIX = {"priority": 0.6, "review": 0.2, "random": 0.2}


def make_pool(n: int, n_topics: int = 0, seed: int = 0) -> Pool:
    r = np.random.default_rng(seed)
    ids = np.arange(1, n + 1) * 10
    topic_idx = (np.arange(n) % n_topics) if n_topics else np.full(n, -1)
    b = r.normal(0, 1, n)
    return Pool(ids, topic_idx, b, [f"t{i}" for i in range(n_topics)])


def test_eligibility_filters(db_loaded):
    cfg = make_config()
    pool = Pool.from_db(db_loaded, cfg)
    flagged = {
        r[0]
        for r in db_loaded.execute(
            "SELECT clue_id FROM clues WHERE needs_media OR is_tournament OR is_special"
        ).fetchall()
    }
    assert flagged and not (flagged & set(pool.ids.tolist()))
    cfg.eligibility = Eligibility(include_media=True, include_tournament=True, include_special=True)
    everything = Pool.from_db(db_loaded, cfg)
    assert flagged <= set(everything.ids.tolist())


def test_random_uniform_chi_square(rng):
    pool = make_pool(50)
    sampler = RandomSampler(pool, rng)
    counts = Counter(sampler.draw().clue_id for _ in range(20_000))
    observed = [counts[i] for i in pool.ids]
    assert stats.chisquare(observed).pvalue > 0.001


def test_random_records_one_over_n(rng):
    pool = make_pool(50)
    sampler = RandomSampler(pool, rng)
    assert sampler.draw().prob == pytest.approx(1 / 50)
    for _ in range(10):
        pool.mark_served(sampler.draw().clue_id)
    assert sampler.draw().prob == pytest.approx(1 / 40)


def test_no_repeats_until_exhausted(rng):
    pool = make_pool(30)
    sampler = RandomSampler(pool, rng)
    seen = []
    for _ in range(30):
        pick = sampler.draw()
        pool.mark_served(pick.clue_id)
        seen.append(pick.clue_id)
    assert len(set(seen)) == 30
    assert sampler.draw() is not None  # exhausted: the pool resets instead of failing


def _adaptive(pool, rng, mix=MIX, review=None, prio=None, model=None):
    model = model or AbilityModel(Ability())
    prio = prio or {t: 1.0 for t in pool.topics}
    return AdaptiveSampler(pool, rng, mix, 0.8, lambda: model, lambda: prio, review)


def test_adaptive_mix_shares(rng):
    pool = make_pool(300, 4)
    review = lambda exclude: (99, [int(pool.ids[0]), int(pool.ids[1]), int(pool.ids[2])])  # noqa: E731
    sampler = _adaptive(pool, rng, review=review)
    modes = Counter(sampler.draw().mode for _ in range(10_000))
    for comp, share in MIX.items():
        assert abs(modes[f"adaptive:{comp}"] / 10_000 - share) < 0.02


def test_adaptive_redistributes_without_reviews(rng):
    pool = make_pool(60, 3)
    sampler = _adaptive(pool, rng, review=lambda exclude: None)
    st = sampler.state()
    assert set(st.shares) == {"priority", "random"}
    assert st.shares["priority"] == pytest.approx(0.75)
    assert st.shares["random"] == pytest.approx(0.25)
    modes = Counter(sampler.draw().mode for _ in range(4000))
    assert abs(modes["adaptive:priority"] / 4000 - 0.75) < 0.03
    # the recorded probability reflects the redistribution
    pick = sampler.draw()
    i = pool.index_of(pick.clue_id)
    expected = 0.75 * st.components["priority"][i] + 0.25 / 60
    assert pick.prob == pytest.approx(expected)


def test_untagged_corpus_falls_back_to_random(rng):
    pool = make_pool(40, 0)
    sampler = _adaptive(pool, rng, review=lambda exclude: None)
    st = sampler.state()
    assert st.shares == {"random": 1.0}
    assert sampler.draw().prob == pytest.approx(1 / 40)


def brute_force_marginal(pool: Pool, mix, mu, prio, kernel_sd, review_cands, served):
    """Independent enumeration of every (component, clue) outcome with plain Python."""
    avail = [i for i in range(len(pool)) if pool.ids[i] not in served]
    comps = {}
    # random
    comps["random"] = {i: 1 / len(avail) for i in avail}
    # priority: topic by priority among topics with available clues, then kernel weight
    topics = sorted({int(pool.topic_idx[i]) for i in avail if pool.topic_idx[i] >= 0})
    total_p = sum(prio[pool.topics[t]] for t in topics)
    q = {}
    for t in topics:
        members = [i for i in avail if pool.topic_idx[i] == t]
        weights = {
            i: math.exp(-((pool.b[i] - mu[pool.topics[t]]) ** 2) / (2 * kernel_sd**2))
            for i in members
        }
        z = sum(weights.values())
        for i in members:
            q[i] = prio[pool.topics[t]] / total_p * weights[i] / z
    comps["priority"] = q
    rc = [i for i in range(len(pool)) if int(pool.ids[i]) in review_cands and i in avail]
    if rc:
        comps["review"] = {i: 1 / len(rc) for i in rc}
    shares = {k: mix[k] for k in comps}
    s = sum(shares.values())
    marginal = {}
    for k, dist in comps.items():
        for i, p in dist.items():
            marginal[i] = marginal.get(i, 0) + shares[k] / s * p
    return marginal


def test_sampling_prob_exact_by_enumeration(rng):
    pool = Pool(
        np.array([1, 2, 3, 4, 5, 6]),
        np.array([0, 0, 0, 1, 1, -1]),
        np.array([-1.0, 0.0, 1.0, -0.5, 0.5, 0.0]),
        ["a", "b"],
    )
    model = AbilityModel(Ability())
    model.update("a", 0.0, True)
    mu = {"a": model.get("a").mu, "b": model.get("b").mu}
    prio = {"a": 0.3, "b": 0.1}
    review = lambda exclude: (7, [c for c in (2, 4) if c not in exclude])  # noqa: E731
    sampler = _adaptive(pool, rng, review=review, prio=prio, model=model)
    pool.mark_served(5)
    expected = brute_force_marginal(pool, MIX, mu, prio, 0.8, {2, 4}, {5})
    assert sum(expected.values()) == pytest.approx(1.0)
    seen = Counter()
    n = 30_000
    for _ in range(n):
        pick = sampler.draw()
        i = pool.index_of(pick.clue_id)
        assert pick.prob == pytest.approx(expected[i], rel=1e-9)
        seen[i] += 1
    assert seen[pool.index_of(5)] == 0  # served this session: never drawn
    for i, p in expected.items():
        assert abs(seen[i] / n - p) < 4 * math.sqrt(p * (1 - p) / n) + 1e-3


def test_review_sampler(rng):
    pool = make_pool(10)
    sampler = ReviewSampler(pool, rng, lambda exclude: (5, [10, 20, 30, 40]))
    pick = sampler.draw()
    assert pick.mode == "review" and pick.prob == 0.25 and pick.entity_id == 5
    assert ReviewSampler(pool, rng, lambda exclude: None).draw() is None
