"""Milestone 3 acceptance: adaptive mode visibly shifts toward high-priority topics."""

from collections import Counter

import numpy as np
import pytest
from conftest import make_config
from scipy import special

from jtrainer.config import Ability
from jtrainer.model.ability import AbilityModel
from jtrainer.model.priority import rank_topics
from jtrainer.sampling import AdaptiveSampler, Pool

pytestmark = pytest.mark.slow


def test_adaptive_shifts_toward_priority():
    rng = np.random.default_rng(11)
    cfg = make_config()
    n_topics, per = 6, 3000
    freq = np.array([0.30, 0.25, 0.15, 0.12, 0.10, 0.08])
    true = np.array([1.5, -1.0, 0.5, -1.5, 2.0, 0.0])  # topic 1 frequent and weak
    topic_idx = np.repeat(np.arange(n_topics), per)
    b = rng.normal(0.0, 1.2, len(topic_idx))
    pool = Pool(np.arange(1, len(b) + 1), topic_idx, b, [f"t{i}" for i in range(n_topics)])
    freqs = {f"t{i}": float(freq[i]) for i in range(n_topics)}
    model = AbilityModel(Ability(drift_sd_per_day=0.0))

    def priorities():
        return {tp.topic_id: tp.priority for tp in rank_topics(freqs, model, cfg)}

    sampler = AdaptiveSampler(
        pool, rng, {"priority": 0.8, "random": 0.2}, 0.8, lambda: model, priorities, None
    )
    early, late = Counter(), Counter()
    for k in range(1500):
        pick = sampler.draw()
        i = pool.index_of(pick.clue_id)
        t = int(topic_idx[i])
        model.update(f"t{t}", float(b[i]), bool(rng.random() < special.expit(true[t] - b[i])))
        pool.mark_served(pick.clue_id)
        (early if k < 300 else late)[t] += 1
    late_share = np.array([late[t] for t in range(n_topics)]) / sum(late.values())
    # after learning abilities, the weakest frequent topics get more than uniform-board share
    assert late_share[1] > 1 / n_topics and late_share[3] > 1 / n_topics
    # and strong topics get less than under uniform random sampling of the board
    assert late_share[4] < 1 / n_topics
    ranked = [tp.topic_id for tp in rank_topics(freqs, model, cfg)]
    assert ranked[0] == "t1"
