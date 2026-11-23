"""Clue samplers (DESIGN §9). Every pick carries the exact probability the sampler assigned.

- RandomSampler: uniform over eligible, not-yet-served clues; prob = 1/N_remaining.
- AdaptiveSampler: a mixture of components (priority, review, random). The recorded
  probability is the clue's *marginal* probability under the mixture, so inverse
  probability weighting over attempts recovers accuracy on a uniform board.
  Unavailable components (no tags yet, no due reviews) have their share redistributed
  proportionally over the available ones.
- ReviewSampler: due FSRS entities only; prob = 1/k over the entity's candidate clues.

No clue repeats within a session until the eligible pool is exhausted.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import duckdb
import numpy as np

from jtrainer.config import Config
from jtrainer.model.ability import AbilityModel, clue_difficulty_sql
from jtrainer.model.priority import eligibility_sql
from jtrainer.tagging.taxonomy import current_tags_sql


@dataclass
class Pick:
    clue_id: int
    mode: str
    prob: float
    entity_id: int | None = None


class Pool:
    """Eligible clues as parallel numpy arrays, plus the session's served mask."""

    def __init__(self, ids: np.ndarray, topic_idx: np.ndarray, b: np.ndarray, topics: list[str]):
        order = np.argsort(ids, kind="stable")
        self.ids = np.asarray(ids, dtype=np.int64)[order]
        self.topic_idx = np.asarray(topic_idx, dtype=np.int64)[order]
        self.b = np.asarray(b, dtype=np.float64)[order]
        self.topics = list(topics)
        self.served = np.zeros(len(self.ids), dtype=bool)

    @classmethod
    def from_db(cls, conn: duckdb.DuckDBPyConnection, config: Config) -> Pool:
        version = config.taxonomy.version
        res = conn.execute(
            f"""
            SELECT c.clue_id, COALESCE(t.topic_id, '') AS topic,
                   {clue_difficulty_sql(config.difficulty)} AS b
            FROM clues c
            LEFT JOIN clue_difficulty d USING (clue_id)
            LEFT JOIN ({current_tags_sql(version)}) t USING (clue_id)
            WHERE {eligibility_sql(config)}
            """
        ).fetchnumpy()
        topic_col = np.asarray(res["topic"], dtype=object)
        topics = sorted({t for t in topic_col if t})
        lookup = {t: i for i, t in enumerate(topics)}
        topic_idx = np.array([lookup.get(t, -1) for t in topic_col], dtype=np.int64)
        return cls(np.asarray(res["clue_id"]), topic_idx, np.asarray(res["b"], dtype=float), topics)

    def __len__(self) -> int:
        return len(self.ids)

    def index_of(self, clue_id: int) -> int | None:
        i = int(np.searchsorted(self.ids, clue_id))
        return i if i < len(self.ids) and self.ids[i] == clue_id else None

    def available(self) -> np.ndarray:
        if self.served.all():
            self.served[:] = False  # pool exhausted: allow repeats
        return ~self.served

    def mark_served(self, clue_id: int) -> None:
        i = self.index_of(clue_id)
        if i is not None:
            self.served[i] = True


class RandomSampler:
    mode = "random"

    def __init__(self, pool: Pool, rng: np.random.Generator):
        self.pool = pool
        self.rng = rng

    def draw(self) -> Pick | None:
        if len(self.pool) == 0:
            return None
        avail = self.pool.available()
        k = int(avail.sum())
        if k == len(self.pool) or k > len(self.pool) // 2:
            while True:  # rejection sampling: uniform over the available clues
                i = int(self.rng.integers(len(self.pool)))
                if avail[i]:
                    break
        else:
            i = int(np.flatnonzero(avail)[self.rng.integers(k)])
        return Pick(int(self.pool.ids[i]), self.mode, 1.0 / k)


# A review source returns (entity_id, candidate clue ids) for the most overdue entity,
# skipping clues in `exclude`, or None when nothing is due.
ReviewSource = Callable[[set[int]], tuple[int, list[int]] | None]


def scheduler_review_source(scheduler) -> ReviewSource:
    def source(exclude: set[int]) -> tuple[int, list[int]] | None:
        for item in scheduler.due():
            cands = scheduler.candidate_clues(item, exclude)
            if cands:
                return item.entity_id, cands
        return None

    return source


class ReviewSampler:
    mode = "review"

    def __init__(self, pool: Pool, rng: np.random.Generator, review_source: ReviewSource):
        self.pool = pool
        self.rng = rng
        self.review_source = review_source

    def draw(self) -> Pick | None:
        served = {int(x) for x in self.pool.ids[self.pool.served]}
        got = self.review_source(served)
        if got is None:
            return None
        entity_id, cands = got
        clue_id = cands[int(self.rng.integers(len(cands)))]
        return Pick(int(clue_id), self.mode, 1.0 / len(cands), entity_id)


@dataclass
class MixtureState:
    shares: dict[str, float]
    components: dict[str, np.ndarray]
    marginal: np.ndarray
    review_entity: int | None


class AdaptiveSampler:
    def __init__(
        self,
        pool: Pool,
        rng: np.random.Generator,
        mix: dict[str, float],
        kernel_sd: float,
        ability: Callable[[], AbilityModel],
        priorities: Callable[[], dict[str, float]],
        review_source: ReviewSource | None = None,
    ):
        self.pool = pool
        self.rng = rng
        self.mix = dict(mix)
        self.kernel_sd = kernel_sd
        self.ability = ability
        self.priorities = priorities
        self.review_source = review_source

    def _priority_component(self, avail: np.ndarray) -> np.ndarray | None:
        pool = self.pool
        if not pool.topics:
            return None
        prio = self.priorities()
        model = self.ability()
        mu = np.array([model.get(t).mu for t in pool.topics])
        m = avail & (pool.topic_idx >= 0)
        if not m.any():
            return None
        tidx = pool.topic_idx[m]
        w = np.exp(-((pool.b[m] - mu[tidx]) ** 2) / (2 * self.kernel_sd**2))
        w = np.maximum(w, 1e-12)
        w_topic = np.bincount(tidx, weights=w, minlength=len(pool.topics))
        pi = np.array([max(0.0, prio.get(t, 0.0)) for t in pool.topics])
        pi[w_topic == 0] = 0.0
        if pi.sum() <= 0:
            return None
        pi = pi / pi.sum()
        q = np.zeros(len(pool))
        q[m] = pi[tidx] * w / w_topic[tidx]
        return q

    def _review_component(self, avail: np.ndarray) -> tuple[np.ndarray, int] | None:
        if self.review_source is None:
            return None
        served = {int(x) for x in self.pool.ids[~avail]}
        got = self.review_source(served)
        if got is None:
            return None
        entity_id, cands = got
        idx = [i for c in cands if (i := self.pool.index_of(c)) is not None and avail[i]]
        if not idx:
            return None
        q = np.zeros(len(self.pool))
        q[idx] = 1.0 / len(idx)
        return q, entity_id

    def state(self) -> MixtureState:
        avail = self.pool.available()
        comps: dict[str, np.ndarray] = {}
        review_entity = None
        if self.mix.get("priority", 0) > 0:
            q = self._priority_component(avail)
            if q is not None:
                comps["priority"] = q
        if self.mix.get("review", 0) > 0:
            got = self._review_component(avail)
            if got is not None:
                comps["review"], review_entity = got
        if self.mix.get("random", 0) > 0 or not comps:
            comps["random"] = avail / avail.sum()
        raw = {k: self.mix.get(k, 0.0) for k in comps}
        if sum(raw.values()) <= 0:
            raw = {"random": 1.0}
        total = sum(raw.values())
        shares = {k: v / total for k, v in raw.items()}
        marginal = sum(shares[k] * comps[k] for k in comps)
        return MixtureState(shares, comps, marginal, review_entity)

    def draw(self) -> Pick | None:
        if len(self.pool) == 0:
            return None
        st = self.state()
        names = list(st.shares)
        probs = np.array([st.shares[n] for n in names])
        comp = names[int(self.rng.choice(len(names), p=probs / probs.sum()))]
        q = st.components[comp]
        cdf = np.cumsum(q)
        i = int(np.searchsorted(cdf, self.rng.random() * cdf[-1], side="right"))
        if i >= len(q) or q[i] <= 0:  # float edge at the top of the cdf
            i = int(np.flatnonzero(q > 0)[-1])
        entity = st.review_entity if comp == "review" else None
        return Pick(int(self.pool.ids[i]), f"adaptive:{comp}", float(st.marginal[i]), entity)
