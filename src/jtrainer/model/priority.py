"""Topic priority (DESIGN §8.2): priority = frequency x (1 - predicted accuracy)."""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass

import duckdb

from jtrainer.config import Config
from jtrainer.model.ability import AbilityModel, prior_difficulty
from jtrainer.tagging.taxonomy import current_tags_sql


@dataclass
class TopicPriority:
    topic_id: str
    frequency: float
    p_hat: float
    priority: float


def priority(frequency: float, p_hat: float) -> float:
    return frequency * (1.0 - p_hat)


def recency_weight(age_days: float, half_life_years: float) -> float:
    if half_life_years <= 0:
        return 1.0
    return math.exp(-math.log(2) * (age_days / 365.25) / half_life_years)


def topic_frequencies(
    conn: duckdb.DuckDBPyConnection,
    config: Config,
    version: str | None = None,
    ref_date: dt.date | None = None,
    half_life_years: float | None = None,
) -> dict[str, float]:
    """Each topic's recency-weighted share of eligible tagged clues. Sums to 1 over the
    taxonomy (topics with no clues get 0); empty dict if nothing is tagged."""
    version = version or config.taxonomy.version
    hl = config.priority.half_life_years if half_life_years is None else half_life_years
    topics = [
        r[0]
        for r in conn.execute(
            "SELECT topic_id FROM topics WHERE taxonomy_version = ?", [version]
        ).fetchall()
    ]
    if ref_date is None:
        ref_date = conn.execute("SELECT max(air_date) FROM clues").fetchone()[0]
    if ref_date is None:
        return {}
    if hl > 0:
        k = math.log(2) / (365.25 * hl)
        weight = f"exp(-{k!r} * greatest(0, date_diff('day', c.air_date, DATE '{ref_date}')))"
    else:
        weight = "1.0"
    rows = conn.execute(
        f"""
        SELECT t.topic_id, sum({weight}) FROM ({current_tags_sql(version)}) t
        JOIN clues c USING (clue_id)
        WHERE {eligibility_sql(config)}
        GROUP BY 1
        """
    ).fetchall()
    total = sum(w for _, w in rows)
    if not total:
        return {}
    freqs = dict.fromkeys(topics, 0.0)
    for topic, w in rows:
        freqs[topic] = w / total
    return freqs


def eligibility_sql(config: Config, alias: str = "c") -> str:
    e = config.eligibility
    parts = ["TRUE"]
    if not e.include_media:
        parts.append(f"NOT {alias}.needs_media")
    if not e.include_tournament:
        parts.append(f"NOT {alias}.is_tournament")
    if not e.include_special:
        parts.append(f"NOT {alias}.is_special")
    return " AND ".join(parts)


def rank_topics(
    freqs: dict[str, float], model: AbilityModel, config: Config
) -> list[TopicPriority]:
    b_target = prior_difficulty(
        config.priority.target_round, config.priority.target_row, config.difficulty
    )
    out = []
    for topic, f in freqs.items():
        p = model.predict(topic, b_target)
        out.append(TopicPriority(topic, f, p, priority(f, p)))
    out.sort(key=lambda t: (-t.priority, t.topic_id))
    return out
