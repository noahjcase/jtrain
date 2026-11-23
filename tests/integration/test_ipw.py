"""Inverse-probability weighting (TESTING §9). This is why sampling_prob exists: adaptive
sampling oversamples clues near the user's ability, so naive accuracy is biased; IPW over
the recorded probabilities recovers accuracy on a uniform board. Must not be skipped.

The pool must be large relative to the number of draws: each draw's probability describes
the clues still unserved, so a heavily depleted pool would make the estimate describe a
shrinking remainder rather than the whole board. Real pools have ~495k clues."""

import numpy as np
from scipy import special

from jtrainer.config import Ability
from jtrainer.model.ability import AbilityModel
from jtrainer.model.stats import ipw_accuracy
from jtrainer.sampling import AdaptiveSampler, Pool


def test_ipw_recovers_uniform_board_accuracy():
    rng = np.random.default_rng(2024)
    n_topics, per_topic = 4, 4000  # pool >> draws, as in real use (~495k clues)
    true_theta = np.array([1.8, 0.8, -0.2, -1.2])
    topic_idx = np.repeat(np.arange(n_topics), per_topic)
    b = rng.normal(0.0, 1.6, n_topics * per_topic)
    pool = Pool(np.arange(1, len(b) + 1), topic_idx, b, [f"t{i}" for i in range(n_topics)])
    p_true = special.expit(true_theta[topic_idx] - b)
    uniform_acc = np.array([p_true[topic_idx == t].mean() for t in range(n_topics)])

    model = AbilityModel(Ability(prior_sd=1.5))
    prio = {f"t{i}": 0.25 * (1 - 0.5) for i in range(n_topics)}
    sampler = AdaptiveSampler(
        pool, rng, {"priority": 0.8, "random": 0.2}, 0.5, lambda: model, lambda: prio, None
    )
    records = {t: [] for t in range(n_topics)}
    for _ in range(1200):
        pick = sampler.draw()
        i = pool.index_of(pick.clue_id)
        y = bool(rng.random() < p_true[i])
        t = int(topic_idx[i])
        model.update(f"t{t}", float(b[i]), y)
        pool.mark_served(pick.clue_id)
        records[t].append((y, pick.prob))

    naive_err, ipw_err = [], []
    for t in range(n_topics):
        est = ipw_accuracy(records[t])
        naive_err.append(est.naive - uniform_acc[t])
        ipw_err.append(est.ipw - uniform_acc[t])
        assert abs(est.ipw - uniform_acc[t]) < 0.15, (t, est, uniform_acc[t])
    # Tolerances from 30 seeds of this simulation: mean |IPW error| had median 0.035 and
    # max 0.058; mean |naive error| had min 0.086 and median 0.106; per-topic |IPW error|
    # had p95 0.128.
    assert np.mean(np.abs(ipw_err)) < 0.07
    assert np.mean(np.abs(naive_err)) > 0.08  # naive is measurably biased
    assert np.mean(np.abs(ipw_err)) < np.mean(np.abs(naive_err))
