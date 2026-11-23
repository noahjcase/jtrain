"""Ability and difficulty, version 2 (DESIGN §8.1): offline Rasch / 1PL with pooling.

Two fits:

1. Clue difficulty from contestant outcomes (`clue_outcomes`). Each of three contestants
   independently knows the response with p = logistic(-b) (difficulty is relative to an
   average contestant, theta = 0). A contestant who knows buzzes with probability rho_k;
   one who doesn't buzzes (and is wrong) with probability rho_g. Buzzers ring in in random
   order until someone is right. Observed per clue: number wrong, and whether anyone was
   right. This models the censoring: a triple stumper with no wrong responses ("no one
   buzzed") is evidence of difficulty *and* of reluctance, not proof that no one knew.
   Clue difficulty is b = mu_level + tau * z, z ~ N(0, 1), with level = round x board row;
   level means, tau, rho_k and rho_g are fitted by marginal maximum likelihood with
   Gauss-Hermite quadrature, and each clue gets the posterior mean and sd of b.

2. User ability per topic: P(correct) = logistic(theta_topic - b_clue) with
   theta_topic ~ Normal(theta_global, sigma^2). theta_global and sigma are fitted by
   Laplace-approximate marginal likelihood (empirical Bayes); topic estimates are the
   resulting posterior modes, which shrink low-data topics toward the global mean.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import optimize, special

from jtrainer.model.ability import N_LEVELS

# Outcome categories: (anyone correct, number of wrong responses before that / in total)
OUTCOMES: list[tuple[int, int]] = [(1, 0), (1, 1), (1, 2), (0, 0), (0, 1), (0, 2), (0, 3)]
OUTCOME_INDEX = {o: i for i, o in enumerate(OUTCOMES)}
N_OUT = len(OUTCOMES)


def outcome_index(n_correct: int, n_incorrect: int) -> int:
    correct = 1 if n_correct > 0 else 0
    k = min(n_incorrect, 2 if correct else 3)
    return OUTCOME_INDEX[(correct, k)]


def _falling(n: int, k: int) -> float:
    out = 1.0
    for i in range(k):
        out *= n - i
    return out


# Precompute the multinomial bookkeeping for three contestants.
_TERMS: list[tuple[int, int, int, float, list[tuple[int, float]]]] = []
for _nk in range(4):
    for _ng in range(4 - _nk):
        _ns = 3 - _nk - _ng
        _coef = math.factorial(3) / (
            math.factorial(_nk) * math.factorial(_ng) * math.factorial(_ns)
        )
        if _nk == 0:
            _dist = [(OUTCOME_INDEX[(0, _ng)], 1.0)]
        else:
            _dist = [
                (OUTCOME_INDEX[(1, k)], _falling(_ng, k) * _nk / _falling(_nk + _ng, k + 1))
                for k in range(_ng + 1)
            ]
        _TERMS.append((_nk, _ng, _ns, _coef, _dist))


def outcome_probs(p_know: np.ndarray, rho_k: float, rho_g: float) -> np.ndarray:
    """P(outcome) for each p_know, shape (..., 7)."""
    p = np.asarray(p_know, dtype=float)
    a = p * rho_k
    g = (1 - p) * rho_g
    s = 1 - a - g
    out = np.zeros((*p.shape, N_OUT))
    for nk, ng, ns, coef, dist in _TERMS:
        w = coef * a**nk * g**ng * s**ns
        for idx, frac in dist:
            out[..., idx] += w * frac
    return out


def simulate_outcomes(
    b: np.ndarray, rho_k: float, rho_g: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate (n_correct, n_incorrect) per clue under the contestant model."""
    probs = outcome_probs(special.expit(-np.asarray(b)), rho_k, rho_g)
    u = rng.random(len(b))[:, None]
    idx = (probs.cumsum(axis=1) < u).sum(axis=1)
    idx = np.minimum(idx, N_OUT - 1)
    n_correct = np.array([OUTCOMES[i][0] for i in idx])
    n_incorrect = np.array([OUTCOMES[i][1] for i in idx])
    return n_correct, n_incorrect


@dataclass
class DifficultyFit:
    level_mu: np.ndarray  # (N_LEVELS,)
    tau: float
    rho_k: float
    rho_g: float
    loglik: float
    post_mean: np.ndarray  # (N_LEVELS, N_OUT): E[b | level, outcome]
    post_sd: np.ndarray

    def clue_posterior(
        self, level: np.ndarray, outcome: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        return self.post_mean[level, outcome], self.post_sd[level, outcome]


_GH_Z, _GH_W = np.polynomial.hermite_e.hermegauss(24)
_GH_W = _GH_W / _GH_W.sum()


def _count_table(level: np.ndarray, outcome: np.ndarray) -> np.ndarray:
    counts = np.zeros((N_LEVELS, N_OUT))
    np.add.at(counts, (level, outcome), 1)
    return counts


def fit_difficulty(
    level: np.ndarray, outcome: np.ndarray, prior_mu: np.ndarray | None = None
) -> DifficultyFit:
    """Fit the contestant model. `level` in 0..10, `outcome` in 0..6 (see OUTCOMES)."""
    counts = _count_table(np.asarray(level), np.asarray(outcome))
    present = counts.sum(axis=1) > 0
    prior_mu = np.zeros(N_LEVELS) if prior_mu is None else np.asarray(prior_mu, float)

    def unpack(x):
        return (
            x[:N_LEVELS],
            math.exp(x[N_LEVELS]),
            special.expit(x[N_LEVELS + 1]),
            special.expit(x[N_LEVELS + 2]),
        )

    def negloglik(x):
        mu, tau, rho_k, rho_g = unpack(x)
        b = mu[:, None] + tau * _GH_Z[None, :]  # (L, Q)
        probs = outcome_probs(special.expit(-b), rho_k, rho_g)  # (L, Q, O)
        marg = np.einsum("q,lqo->lo", _GH_W, probs)  # (L, O)
        ll = (counts * np.log(np.maximum(marg, 1e-300))).sum()
        # weak priors keep empty levels and the reluctance parameters well defined
        lp = -0.5 * (((mu - prior_mu) / 3.0) ** 2).sum()
        lp += -0.5 * ((x[N_LEVELS] - math.log(0.8)) / 1.0) ** 2
        lp += (
            -0.5 * ((x[N_LEVELS + 1] - 2.0) / 2.0) ** 2 - 0.5 * ((x[N_LEVELS + 2] + 2.0) / 2.0) ** 2
        )
        return -(ll + lp)

    x0 = np.concatenate([prior_mu, [math.log(0.8), 2.0, -2.0]])
    res = optimize.minimize(negloglik, x0, method="L-BFGS-B")
    mu, tau, rho_k, rho_g = unpack(res.x)
    mu = np.where(present, mu, prior_mu)
    b = mu[:, None] + tau * _GH_Z[None, :]
    probs = outcome_probs(special.expit(-b), rho_k, rho_g)  # (L, Q, O)
    joint = _GH_W[None, :, None] * probs
    norm = joint.sum(axis=1)  # (L, O)
    post_mean = (joint * b[:, :, None]).sum(axis=1) / norm
    post_var = (joint * b[:, :, None] ** 2).sum(axis=1) / norm - post_mean**2
    return DifficultyFit(
        mu, tau, rho_k, rho_g, -res.fun, post_mean, np.sqrt(np.maximum(post_var, 0))
    )


def fit_difficulty_naive(level: np.ndarray, outcome: np.ndarray) -> np.ndarray:
    """Naive level difficulties: treat a triple stumper as 'all three failed to know'.
    P(anyone correct) = 1 - (1 - p)^3 with p = logistic(-b). Ignores reluctance to buzz."""
    level = np.asarray(level)
    anyone = np.array([OUTCOMES[o][0] for o in np.asarray(outcome)])
    mu = np.zeros(N_LEVELS)
    for lv in range(N_LEVELS):
        m = level == lv
        if not m.any():
            continue
        rate = np.clip(anyone[m].mean(), 1e-4, 1 - 1e-4)
        p = 1 - (1 - rate) ** (1 / 3)
        mu[lv] = -special.logit(p)
    return mu


# ---------------------------------------------------------------------------
# User ability with hierarchical pooling


@dataclass
class HierarchicalAbility:
    theta_global: float
    sigma: float
    topics: list[str]
    theta: np.ndarray
    theta_sd: np.ndarray
    n: np.ndarray


def _topic_mode(y: np.ndarray, b: np.ndarray, g: float, s2: float) -> tuple[float, float]:
    """Newton iterations for the posterior mode of one topic's theta; returns (mode, H)."""
    th = g
    h = 1.0 / s2
    for _ in range(50):
        p = special.expit(th - b)
        grad = (y - p).sum() - (th - g) / s2
        h = (p * (1 - p)).sum() + 1.0 / s2
        step = grad / h
        th += step
        if abs(step) < 1e-10:
            break
    return th, h


def fit_user_ability(topic: list[str], correct: np.ndarray, b: np.ndarray) -> HierarchicalAbility:
    topics = sorted(set(topic))
    idx = {t: i for i, t in enumerate(topics)}
    t_idx = np.array([idx[t] for t in topic])
    y = np.asarray(correct, dtype=float)
    b = np.asarray(b, dtype=float)
    groups = [(y[t_idx == i], b[t_idx == i]) for i in range(len(topics))]

    def neg_marginal(x):
        g, log_s = x
        s2 = math.exp(2 * log_s)
        total = 0.0
        for yy, bb in groups:
            th, h = _topic_mode(yy, bb, g, s2)
            p = special.expit(th - bb)
            ll = (
                yy * np.log(np.maximum(p, 1e-300)) + (1 - yy) * np.log(np.maximum(1 - p, 1e-300))
            ).sum()
            lp = -0.5 * (th - g) ** 2 / s2 - 0.5 * math.log(2 * math.pi * s2)
            total += ll + lp + 0.5 * math.log(2 * math.pi) - 0.5 * math.log(h)
        # weak hyperprior on sigma keeps it away from 0 and infinity
        total += -0.5 * ((log_s - math.log(0.7)) / 1.0) ** 2
        return -total

    res = optimize.minimize(
        neg_marginal,
        np.array([0.0, math.log(0.7)]),
        method="Nelder-Mead",
        options={"xatol": 1e-6, "fatol": 1e-8, "maxiter": 2000},
    )
    g, log_s = res.x
    s2 = math.exp(2 * log_s)
    theta = np.zeros(len(topics))
    sd = np.zeros(len(topics))
    for i, (yy, bb) in enumerate(groups):
        theta[i], h = _topic_mode(yy, bb, g, s2)
        sd[i] = 1 / math.sqrt(h)
    return HierarchicalAbility(
        float(g), math.sqrt(s2), topics, theta, sd, np.array([len(yy) for yy, _ in groups])
    )


def store_difficulties(
    conn,
    fit: DifficultyFit,
    clue_ids: np.ndarray,
    level: np.ndarray,
    outcome: np.ndarray,
    fitted_at,
) -> int:
    from jtrainer.db import to_db

    mean, sd = fit.clue_posterior(np.asarray(level), np.asarray(outcome))
    rows = [
        [int(c), "irt-v2", float(m), float(s), to_db(fitted_at)]
        for c, m, s in zip(clue_ids, mean, sd, strict=True)
    ]
    conn.executemany(
        "INSERT INTO clue_difficulty VALUES (?, ?, ?, ?, ?) ON CONFLICT (clue_id) DO UPDATE SET "
        "model = excluded.model, b_mean = excluded.b_mean, b_sd = excluded.b_sd, "
        "fitted_at = excluded.fitted_at",
        rows,
    )
    return len(rows)


def fit_from_db(conn, config, clock) -> DifficultyFit | None:
    """Fit clue difficulties from all stored contestant outcomes and store them."""
    from jtrainer.model.ability import level_difficulties

    res = conn.execute(
        "SELECT o.clue_id, c.round, c.board_row, o.n_correct, o.n_incorrect "
        "FROM clue_outcomes o JOIN clues c USING (clue_id)"
    ).fetchall()
    if not res:
        return None
    from jtrainer.model.ability import level_index

    clue_ids = np.array([r[0] for r in res])
    level = np.array([level_index(r[1], r[2]) for r in res])
    outcome = np.array([outcome_index(r[3], r[4]) for r in res])
    fit = fit_difficulty(level, outcome, prior_mu=np.array(level_difficulties(config.difficulty)))
    store_difficulties(conn, fit, clue_ids, level, outcome, clock.now())
    return fit
