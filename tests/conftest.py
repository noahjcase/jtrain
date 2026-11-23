"""Shared fixtures (TESTING §4.1). Deterministic: injected RNG and clock, no network."""

from __future__ import annotations

import datetime as dt
import os
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
from jtrainer.llm import BatchResult, LLMRequest

FIXTURES = Path(__file__).parent / "fixtures"
SAMPLE_TSV = FIXTURES / "clues_sample.tsv"


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


# ---------------------------------------------------------------------------
# fake LLM


class FakeLLM:
    """Same interface as the real client. Canned responses are keyed by
    (prompt_version, input_hash); an unregistered input raises. Records every call."""

    def __init__(self):
        self.responses: dict[tuple[str, str], list[str]] = {}
        self.calls: list[LLMRequest] = []
        self.batches: dict[str, list[tuple[str, LLMRequest]]] = {}
        self.batch_outcomes: dict[str, BatchResult] = {}

    def register(self, request: LLMRequest, *texts: str) -> None:
        self.responses[(request.prompt_version, request.input_hash)] = list(texts)

    def complete(self, request: LLMRequest) -> str:
        self.calls.append(request)
        key = (request.prompt_version, request.input_hash)
        if key not in self.responses:
            raise AssertionError(f"FakeLLM: unregistered input for {request.prompt_version}")
        seq = self.responses[key]
        return seq.pop(0) if len(seq) > 1 else seq[0]

    # batch interface
    def submit_batch(self, items: list[tuple[str, LLMRequest]]) -> str:
        batch_id = f"batch-{len(self.batches) + 1}"
        self.batches[batch_id] = items
        return batch_id

    def batch_status(self, batch_id: str) -> str:
        return "ended"

    def batch_results(self, batch_id: str):
        for custom_id, req in reversed(self.batches[batch_id]):  # any order
            if custom_id in self.batch_outcomes:
                yield self.batch_outcomes[custom_id]
                continue
            seq = self.responses.get((req.prompt_version, req.input_hash))
            if seq is None:
                yield BatchResult(custom_id, "errored", error="unregistered")
            else:
                yield BatchResult(
                    custom_id, "succeeded", text=seq.pop(0) if len(seq) > 1 else seq[0]
                )


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM()


# ---------------------------------------------------------------------------
# live marker: skip (not fail) without a key


def pytest_collection_modifyitems(config, items):
    if os.environ.get("ANTHROPIC_API_KEY"):
        return
    skip = pytest.mark.skip(reason="ANTHROPIC_API_KEY not set")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip)
