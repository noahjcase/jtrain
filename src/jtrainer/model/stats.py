"""Statistics for the stats screen: IPW-corrected accuracy, ability, priority, timing,
confidence calibration. Each figure comes from a small pure function so the screen and
the tests compute identical values."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field

import duckdb

from jtrainer.config import Config
from jtrainer.model.ability import OVERALL, AbilityModel, ReplayReport, replay
from jtrainer.model.priority import TopicPriority, rank_topics, topic_frequencies
from jtrainer.tagging.taxonomy import current_tags_sql


@dataclass
class AccuracyEstimate:
    n: int
    naive: float | None
    ipw: float | None


def ipw_accuracy(rows: Iterable[tuple[bool, float | None]]) -> AccuracyEstimate:
    """Hajek inverse-probability-weighted accuracy from (correct, sampling_prob) pairs.

    Attempts without a sampling probability get weight 1."""
    n = 0
    hits = 0
    num = den = 0.0
    for correct, prob in rows:
        n += 1
        hits += bool(correct)
        w = 1.0 / prob if prob else 1.0
        num += w * bool(correct)
        den += w
    if n == 0:
        return AccuracyEstimate(0, None, None)
    return AccuracyEstimate(n, hits / n, num / den)


@dataclass
class CalibrationBin:
    confidence: int
    implied: float
    n: int
    accuracy: float | None


def calibration(
    rows: Iterable[tuple[int | None, bool]], probs: list[float]
) -> tuple[list[CalibrationBin], float | None]:
    """Accuracy per confidence key and the Brier score of the implied probabilities."""
    by = defaultdict(list)
    brier_terms = []
    for conf, correct in rows:
        if conf not in (1, 2, 3):
            continue
        by[conf].append(bool(correct))
        brier_terms.append((probs[conf - 1] - float(bool(correct))) ** 2)
    bins = []
    for conf in (1, 2, 3):
        vals = by.get(conf, [])
        bins.append(
            CalibrationBin(
                conf, probs[conf - 1], len(vals), sum(vals) / len(vals) if vals else None
            )
        )
    brier = sum(brier_terms) / len(brier_terms) if brier_terms else None
    return bins, brier


TIME_BINS_MS = (0, 3000, 6000, 10000, 15000, 30000)


def time_histogram(
    ms_values: Iterable[int | None], edges: tuple[int, ...] = TIME_BINS_MS
) -> list[tuple[str, int]]:
    counts = [0] * len(edges)
    for ms in ms_values:
        if ms is None:
            continue
        idx = max(i for i, e in enumerate(edges) if ms >= e)
        counts[idx] += 1
    labels = []
    for i, e in enumerate(edges):
        hi = edges[i + 1] if i + 1 < len(edges) else None
        labels.append(f"{e / 1000:g}-{hi / 1000:g}s" if hi else f"{e / 1000:g}s+")
    return list(zip(labels, counts, strict=True))


def median(values: list[float]) -> float | None:
    if not values:
        return None
    s = sorted(values)
    m = len(s) // 2
    return float(s[m]) if len(s) % 2 else (s[m - 1] + s[m]) / 2


@dataclass
class TopicRow:
    topic_id: str
    label: str
    accuracy: AccuracyEstimate
    mu: float
    sd: float
    frequency: float
    p_hat: float
    priority: float
    rank: int


@dataclass
class Stats:
    overall: AccuracyEstimate
    overall_mu: float
    overall_sd: float
    topics: list[TopicRow]
    untagged_attempts: int
    pending_review: int
    time_hist: list[tuple[str, int]]
    median_submit_ms: float | None
    calibration: list[CalibrationBin]
    brier: float | None
    by_round: dict[int, AccuracyEstimate] = field(default_factory=dict)
    replay: ReplayReport | None = None


def compute_stats(
    conn: duckdb.DuckDBPyConnection, config: Config, model: AbilityModel | None = None
) -> Stats:
    version = config.taxonomy.version
    if model is None:
        model, report = replay(conn, config, version)
    else:
        report = None
    rows = conn.execute(
        f"""
        SELECT a.correct, a.sampling_prob, a.confidence, a.submit_ms, c.round, t.topic_id
        FROM attempts a JOIN clues c USING (clue_id)
        LEFT JOIN ({current_tags_sql(version)}) t USING (clue_id)
        WHERE a.correct IS NOT NULL
        """
    ).fetchall()
    pending = conn.execute(
        "SELECT count(*) FROM attempts WHERE needs_review AND correct IS NULL"
    ).fetchone()[0]
    overall = ipw_accuracy((r[0], r[1]) for r in rows)
    by_topic: dict[str, list] = defaultdict(list)
    by_round: dict[int, list] = defaultdict(list)
    untagged = 0
    for r in rows:
        by_round[r[4]].append((r[0], r[1]))
        if r[5] is None:
            untagged += 1
        else:
            by_topic[r[5]].append((r[0], r[1]))
    freqs = topic_frequencies(conn, config, version)
    ranked: list[TopicPriority] = rank_topics(freqs, model, config) if freqs else []
    rank_of = {tp.topic_id: (i + 1, tp) for i, tp in enumerate(ranked)}
    labels = dict(
        conn.execute(
            "SELECT topic_id, label FROM topics WHERE taxonomy_version = ?", [version]
        ).fetchall()
    )
    topic_ids = sorted(set(labels) | set(by_topic))
    topic_rows = []
    for tid in topic_ids:
        rating = model.get(tid)
        rank, tp = rank_of.get(tid, (len(ranked) + 1, None))
        topic_rows.append(
            TopicRow(
                tid,
                labels.get(tid, tid),
                ipw_accuracy(by_topic.get(tid, [])),
                rating.mu,
                rating.sd,
                tp.frequency if tp else 0.0,
                tp.p_hat if tp else 0.0,
                tp.priority if tp else 0.0,
                rank,
            )
        )
    topic_rows.sort(key=lambda t: t.rank)
    bins, brier = calibration(((r[2], r[0]) for r in rows), config.confidence.probs)
    ov = model.get(OVERALL)
    return Stats(
        overall=overall,
        overall_mu=ov.mu,
        overall_sd=ov.sd,
        topics=topic_rows,
        untagged_attempts=untagged,
        pending_review=pending,
        time_hist=time_histogram(r[3] for r in rows),
        median_submit_ms=median([r[3] for r in rows if r[3] is not None]),
        calibration=bins,
        brier=brier,
        by_round={k: ipw_accuracy(v) for k, v in sorted(by_round.items())},
        replay=report,
    )
