"""Topic ability, version 1 (DESIGN §8.1): an online Glicko-style rating per topic.

P(correct) = logistic(mu_topic - b_clue). After each attempt the topic's Gaussian belief
(mu, var) gets a one-step Laplace update:

    p       = logistic(mu - b)
    var'    = 1 / (1/var + p(1-p))
    mu'     = mu + var' (y - p)

so a surprising result moves the rating more, and uncertainty always shrinks (and stays
positive). Optional process noise (drift per day) lets the rating follow learning.
The model is a pure function of the attempts table: `replay` rebuilds it exactly.
"""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass, field

import duckdb

from jtrainer.config import Ability, Config, Difficulty
from jtrainer.db import from_db
from jtrainer.tagging.taxonomy import current_tags_sql

OVERALL = "_overall"
N_LEVELS = 11  # J rows 1-5, DJ rows 1-5, Final


def level_index(round_: int, board_row: int | None) -> int:
    if round_ == 3:
        return 10
    row = board_row if board_row in (1, 2, 3, 4, 5) else 3
    return (round_ - 1) * 5 + (row - 1)


def prior_difficulty(round_: int, board_row: int | None, cfg: Difficulty) -> float:
    if round_ == 3:
        return cfg.final
    row = board_row if board_row in (1, 2, 3, 4, 5) else 3
    b = cfg.j_row1 + (row - 1) * cfg.row_step
    return b + (cfg.dj_offset if round_ == 2 else 0.0)


def level_difficulties(cfg: Difficulty) -> list[float]:
    out = [prior_difficulty(1, r, cfg) for r in range(1, 6)]
    out += [prior_difficulty(2, r, cfg) for r in range(1, 6)]
    return [*out, cfg.final]


def logistic(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


@dataclass
class Rating:
    mu: float
    var: float
    n: int = 0
    last_at: dt.datetime | None = None

    @property
    def sd(self) -> float:
        return math.sqrt(self.var)

    def interval(self, z: float = 1.645) -> tuple[float, float]:
        return self.mu - z * self.sd, self.mu + z * self.sd


def expected_p(mu: float, var: float, b: float) -> float:
    """Posterior-mean probability of a correct response (probit approximation)."""
    return logistic((mu - b) / math.sqrt(1.0 + math.pi * var / 8.0))


@dataclass
class AbilityModel:
    cfg: Ability = field(default_factory=Ability)
    ratings: dict[str, Rating] = field(default_factory=dict)

    def get(self, topic: str) -> Rating:
        r = self.ratings.get(topic)
        if r is None:
            return Rating(self.cfg.prior_mean, self.cfg.prior_sd**2)
        return r

    def predict(self, topic: str, b: float) -> float:
        r = self.get(topic)
        return expected_p(r.mu, r.var, b)

    def _update_one(self, key: str, b: float, correct: bool, at: dt.datetime | None) -> float:
        r = self.ratings.get(key) or Rating(self.cfg.prior_mean, self.cfg.prior_sd**2)
        var = r.var
        if at is not None and r.last_at is not None and self.cfg.drift_sd_per_day > 0:
            days = max(0.0, (at - r.last_at).total_seconds() / 86400.0)
            var += self.cfg.drift_sd_per_day**2 * days
        p = logistic(r.mu - b)
        new_var = 1.0 / (1.0 / var + p * (1.0 - p))
        new_mu = r.mu + new_var * ((1.0 if correct else 0.0) - p)
        self.ratings[key] = Rating(new_mu, new_var, r.n + 1, at or r.last_at)
        return new_mu - r.mu

    def update(
        self, topic: str | None, b: float, correct: bool, at: dt.datetime | None = None
    ) -> float | None:
        """Update the topic rating (and the overall rating). Returns the topic's mu change."""
        self._update_one(OVERALL, b, correct, at)
        if topic is None:
            return None
        return self._update_one(topic, b, correct, at)


@dataclass
class ReplayReport:
    used: int = 0
    skipped_untagged: int = 0
    skipped_pending: int = 0


def clue_difficulty_sql(cfg: Difficulty) -> str:
    """SQL expression (over alias c, optional alias d for clue_difficulty) for clue b."""
    levels = level_difficulties(cfg)
    j = ", ".join(f"{x!r}" for x in levels[:5])
    dj = ", ".join(f"{x!r}" for x in levels[5:10])
    return (
        f"COALESCE(d.b_mean, CASE WHEN c.round = 3 THEN {cfg.final!r} "
        f"WHEN c.round = 1 THEN [{j}][COALESCE(c.board_row, 3)] "
        f"ELSE [{dj}][COALESCE(c.board_row, 3)] END)"
    )


def replay(
    conn: duckdb.DuckDBPyConnection, config: Config, version: str | None = None
) -> tuple[AbilityModel, ReplayReport]:
    """Rebuild the ability model from the attempts table alone."""
    version = version or config.taxonomy.version
    model = AbilityModel(config.ability)
    report = ReplayReport()
    rows = conn.execute(
        f"""
        SELECT a.correct, a.shown_at, t.topic_id, {clue_difficulty_sql(config.difficulty)} AS b
        FROM attempts a
        JOIN clues c USING (clue_id)
        LEFT JOIN clue_difficulty d USING (clue_id)
        LEFT JOIN ({current_tags_sql(version)}) t USING (clue_id)
        ORDER BY a.shown_at, a.attempt_id
        """
    ).fetchall()
    for correct, shown_at, topic, b in rows:
        if correct is None:
            report.skipped_pending += 1
            continue
        if topic is None:
            report.skipped_untagged += 1
        else:
            report.used += 1
        model.update(topic, float(b), bool(correct), from_db(shown_at))
    return model, report
