"""Topic priority (TESTING §8.3)."""

import datetime as dt

import pytest
from conftest import make_config

from jtrainer.config import Ability
from jtrainer.model.ability import AbilityModel
from jtrainer.model.priority import priority, rank_topics, topic_frequencies


def test_priority_formula():
    assert priority(0.10, 0.70) == pytest.approx(0.03)
    assert priority(0.02, 0.10) == pytest.approx(0.018)


def test_frequent_modest_beats_rare_severe():
    # frequent topic, modest weakness: 0.10 * (1 - 0.6) = 0.04
    # rare topic, severe weakness:     0.02 * (1 - 0.1) = 0.018
    assert priority(0.10, 0.6) > priority(0.02, 0.1)
    # ...but not when the numbers say otherwise
    assert priority(0.10, 0.9) < priority(0.02, 0.1)


def test_rank_topics_orders_by_priority():
    cfg = make_config()
    model = AbilityModel(Ability())
    for _ in range(30):
        model.update("strong", -1.0, True)
        model.update("weak", 1.0, False)
    ranked = rank_topics({"strong": 0.5, "weak": 0.5}, model, cfg)
    assert [t.topic_id for t in ranked] == ["weak", "strong"]
    assert ranked[0].p_hat < ranked[1].p_hat


def test_frequencies_sum_to_one(db_loaded):
    cfg = make_config()
    freqs = topic_frequencies(db_loaded, cfg)
    assert sum(freqs.values()) == pytest.approx(1.0)
    topics = {
        r[0]
        for r in db_loaded.execute(
            "SELECT topic_id FROM topics WHERE taxonomy_version = 'test-v1'"
        ).fetchall()
    }
    assert set(freqs) == topics


def test_recency_weighting(db_loaded):
    cfg = make_config()
    ref = dt.date(2026, 1, 1)
    # 'bible' clues all aired on 2001-11-26; 'animals' eligible clue aired 2011
    with_decay = topic_frequencies(db_loaded, cfg, ref_date=ref, half_life_years=2.0)
    no_decay = topic_frequencies(db_loaded, cfg, ref_date=ref, half_life_years=0.0)
    assert with_decay["bible"] < no_decay["bible"]
    assert sum(with_decay.values()) == pytest.approx(1.0)


def test_no_tags_means_no_frequencies(db, config):
    assert topic_frequencies(db, config) == {}
