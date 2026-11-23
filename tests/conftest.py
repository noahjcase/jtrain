"""Shared fixtures (TESTING §4.1). Deterministic: injected RNG and clock, no network."""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterator
from pathlib import Path

import duckdb
import numpy as np
import pytest

from jtrainer import db as dbmod
from jtrainer.clock import FakeClock
from jtrainer.config import (
    LLM,
    Ability,
    Config,
    Difficulty,
    Eligibility,
    Grading,
    Priority,
    Review,
    Sampling,
    Scrape,
    Taxonomy,
)

# ---------------------------------------------------------------------------
# config, rng, clock


def make_config(tmp: Path | None = None) -> Config:
    """Thresholds and mixes set explicitly so tests don't drift with config.toml."""
    cfg = Config(
        eligibility=Eligibility(
            include_media=False, include_tournament=False, include_special=False
        ),
        taxonomy=Taxonomy(version="test-v1"),
        llm=LLM(judge_model="test-judge", tag_model="test-tagger", effort="low"),
        grading=Grading(fuzzy_high=92.0, fuzzy_low=60.0, min_len_fuzzy=5),
        difficulty=Difficulty(j_row1=-1.6, row_step=0.45, dj_offset=0.7, final=1.4),
        ability=Ability(prior_mean=0.0, prior_sd=1.5, drift_sd_per_day=0.0),
        priority=Priority(half_life_years=8.0, target_round=2, target_row=3),
        review=Review(desired_retention=0.9, slow_ms=12000),
        sampling=Sampling(
            session_length=50,
            difficulty_kernel_sd=0.8,
            mix={"priority": 0.6, "review": 0.2, "random": 0.2},
        ),
        scrape=Scrape(
            base_url="https://archive.test",
            min_interval_s=3.0,
            max_retries=2,
            backoff_s=10.0,
            user_agent="jtrainer-test/1.0",
        ),
    )
    if tmp is not None:
        cfg.root = tmp
    return cfg


@pytest.fixture
def config(tmp_path) -> Config:
    return make_config(tmp_path)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(12345)


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock(dt.datetime(2026, 1, 1, tzinfo=dt.UTC))


# ---------------------------------------------------------------------------
# database


@pytest.fixture
def db() -> Iterator[duckdb.DuckDBPyConnection]:
    conn = dbmod.connect(":memory:")
    yield conn
    conn.close()
