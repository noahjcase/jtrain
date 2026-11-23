"""IRT v2 recovery by simulation (TESTING §8.2). Marked slow."""

import numpy as np
import pytest
from scipy import special

from jtrainer.config import Difficulty
from jtrainer.model.ability import level_difficulties
from jtrainer.model.irt import (
    fit_difficulty,
    fit_difficulty_naive,
    fit_user_ability,
    outcome_index,
    outcome_probs,
    simulate_outcomes,
)

pytestmark = pytest.mark.slow


def simulate_clues(seed, n=20000, rho_k=0.75, rho_g=0.12, tau=0.8):
    rng = np.random.default_rng(seed)
    true_mu = np.array(level_difficulties(Difficulty()))
    level = rng.integers(0, len(true_mu), n)
    b = true_mu[level] + tau * rng.standard_normal(n)
    nc, ni = simulate_outcomes(b, rho_k, rho_g, rng)
    outcome = np.array([outcome_index(c, i) for c, i in zip(nc, ni, strict=True)])
    return true_mu, level, b, outcome


def test_outcome_probs_sum_to_one():
    p = np.linspace(0.01, 0.99, 25)
    assert np.allclose(outcome_probs(p, 0.8, 0.1).sum(axis=-1), 1.0)


def test_difficulty_recovery():
    true_mu, level, b, outcome = simulate_clues(0)
    fit = fit_difficulty(level, outcome)
    # level difficulties are identified well; single-airing clue difficulties are noisy
    assert np.corrcoef(fit.level_mu, true_mu)[0, 1] >= 0.9
    assert np.abs(fit.level_mu - true_mu).max() < 0.3
    assert abs(fit.rho_k - 0.75) < 0.1 and abs(fit.rho_g - 0.12) < 0.05
    post_mean, post_sd = fit.clue_posterior(level, outcome)
    assert np.corrcoef(post_mean, b)[0, 1] > 0.6
    assert (post_sd > 0).all()


def test_censoring_bias_smaller_than_naive():
    """'No one buzzed' triple stumpers inflate naive difficulty; the fitted model, which
    models reluctance to buzz, is much less biased."""
    true_mu, level, _, outcome = simulate_clues(1, rho_k=0.7)
    fitted = fit_difficulty(level, outcome).level_mu
    naive = fit_difficulty_naive(level, outcome)
    fitted_bias = np.abs(fitted - true_mu).mean()
    naive_bias = np.abs(naive - true_mu).mean()
    assert fitted_bias < naive_bias / 3, (fitted_bias, naive_bias)


def test_hierarchical_prior_shrinks_low_data_topics():
    rng = np.random.default_rng(5)
    theta = np.array([0.2, 0.6, -0.4, 0.0, 0.9, -0.1, 0.3, 0.5, 2.5, -2.5])
    n = [400] * 8 + [6, 6]
    topics, ys, bs = [], [], []
    for i, k in enumerate(n):
        b = rng.normal(0, 1, k)
        topics += [f"t{i}"] * k
        ys += list(rng.random(k) < special.expit(theta[i] - b))
        bs += list(b)
    fit = fit_user_ability(topics, np.array(ys), np.array(bs))
    idx = {t: i for i, t in enumerate(fit.topics)}
    # well-measured topics are recovered
    for i in range(8):
        assert abs(fit.theta[idx[f"t{i}"]] - theta[i]) < 0.35
    # low-data extreme topics are pulled toward the global mean
    for i in (8, 9):
        est = fit.theta[idx[f"t{i}"]]
        assert abs(est - fit.theta_global) < abs(theta[i] - fit.theta_global)
        assert fit.theta_sd[idx[f"t{i}"]] > fit.theta_sd[idx["t0"]]
