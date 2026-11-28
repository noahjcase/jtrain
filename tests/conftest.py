"""Shared fixtures (TESTING §4.1). Deterministic: injected RNG and clock, no network."""

from __future__ import annotations

import datetime as dt
import os
import shutil
from collections.abc import Iterator
from pathlib import Path

import duckdb
import httpx
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
from jtrainer.db import to_db
from jtrainer.ingest.jwolle1 import ingest_tsv
from jtrainer.llm import BatchResult, LLMRequest
from jtrainer.tagging.entities import EntityResolver
from jtrainer.tagging.taxonomy import load_taxonomy, store_taxonomy

FIXTURES = Path(__file__).parent / "fixtures"
SAMPLE_TSV = FIXTURES / "clues_sample.tsv"
TEST_TAXONOMY = FIXTURES / "taxonomy_test.yaml"


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
            recency_half_life_years=0.0,
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


# Deterministic topic assignment for the sample clues, by category.
CATEGORY_TOPICS = {
    "U.S. PRESIDENTS": "us_presidents",
    "FAMOUS NAMES": "us_presidents",
    "PEOPLE": "us_presidents",
    "WORLD CAPITALS": "world_geography",
    "GEOGRAPHY": "world_geography",
    "MAPS": "world_geography",
    "BRAZIL": "arts",
    "THE ELEMENTS": "science",
    "SCIENCE": "science",
    "ART": "arts",
    "OPERA": "arts",
    "MUSIC": "arts",
    "LITERATURE": "arts",
    "NOVELS": "arts",
    "80s MUSIC": "arts",
    "PAINTINGS": "arts",
    "INSTRUMENTS": "arts",
    "MUSEUMS": "arts",
    "THE BIBLE": "bible",
    "ANIMALS": "animals",
    "POTPOURRI": "wordplay_other",
    "ODDITIES": "wordplay_other",
    "SHIPS": "world_geography",
    "POKER": "wordplay_other",
    "FRUIT": "science",
    "WORLD LEADERS": "us_presidents",
}


def tag_all(
    conn: duckdb.DuckDBPyConnection,
    version: str = "test-v1",
    when: dt.datetime | None = None,
    topics: dict[str, str] | None = None,
) -> None:
    """Tag every clue under `version` with a fixture 'model', entity from its response."""
    resolver = EntityResolver(conn)
    topics = topics or CATEGORY_TOPICS
    when = when or dt.datetime(2025, 6, 1)
    for clue_id, category, response in conn.execute(
        "SELECT clue_id, category, response FROM clues ORDER BY clue_id"
    ).fetchall():
        entity = resolver.resolve(response, (), source="llm")
        conn.execute(
            "INSERT INTO clue_tags VALUES (?, ?, 'fixture-v1', 'fixture', ?, NULL, ?, ?)",
            [clue_id, version, topics.get(category, "wordplay_other"), entity, to_db(when)],
        )


@pytest.fixture(scope="session")
def loaded_template(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("template") / "loaded.duckdb"
    conn = dbmod.connect(path)
    ingest_tsv(conn, SAMPLE_TSV, make_config())
    store_taxonomy(conn, load_taxonomy(TEST_TAXONOMY))
    tag_all(conn)
    conn.execute("CHECKPOINT")
    conn.close()
    return path


@pytest.fixture
def db_loaded(loaded_template, tmp_path) -> Iterator[duckdb.DuckDBPyConnection]:
    """Sample clues ingested, the test taxonomy stored, every clue tagged. A fresh copy of a
    session-scoped template per test."""
    path = tmp_path / "loaded.duckdb"
    shutil.copy(loaded_template, path)
    conn = dbmod.connect(path)
    yield conn
    conn.close()


def clue_id_by_response(conn, response: str) -> int:
    return conn.execute(
        "SELECT clue_id FROM clues WHERE response = ? ORDER BY clue_id", [response]
    ).fetchone()[0]


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
# fake HTTP


class FakeHTTP:
    """httpx.MockTransport serving fixtures/jarchive, recording requests with fake-clock times.
    `statuses[path]` can queue status codes to return before serving the file."""

    ROUTES = {
        "/robots.txt": "robots.txt",
        "/showseason.php?season=41": "season_41.html",
        "/showgame.php?game_id=8915": "game_complete.html",
        "/showgame.php?game_id=8916": "game_tiebreaker.html",
        "/showgame.php?game_id=8917": "game_unrevealed_old_layout.html",
    }

    def __init__(self, clock: FakeClock):
        self.clock = clock
        self.requests: list[tuple[str, int]] = []  # (path, monotonic ms)
        self.statuses: dict[str, list[int]] = {}
        self.transport = httpx.MockTransport(self.handle)

    def handle(self, request: httpx.Request) -> httpx.Response:
        path = request.url.raw_path.decode()
        self.requests.append((path, self.clock.monotonic_ms()))
        queued = self.statuses.get(path)
        if queued:
            return httpx.Response(queued.pop(0), text="")
        name = self.ROUTES.get(path)
        if name is None:
            return httpx.Response(404, text="not found")
        return httpx.Response(200, text=(FIXTURES / "jarchive" / name).read_text())

    def paths(self) -> list[str]:
        return [p for p, _ in self.requests]


@pytest.fixture
def fake_http(clock) -> FakeHTTP:
    return FakeHTTP(clock)


# ---------------------------------------------------------------------------
# live marker: skip (not fail) without a key


def pytest_collection_modifyitems(config, items):
    if os.environ.get("ANTHROPIC_API_KEY"):
        return
    skip = pytest.mark.skip(reason="ANTHROPIC_API_KEY not set")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip)
