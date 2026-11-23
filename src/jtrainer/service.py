"""Study service: sessions, clue selection, grading, attempt logging, overrides.

The TUI and the integration tests both drive the app through this layer.
Attempts are append-only: the repository has no delete; corrections are overrides.
"""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass
from typing import Any

import duckdb
import numpy as np

from jtrainer import __version__
from jtrainer.clock import Clock, SystemClock
from jtrainer.config import Config
from jtrainer.db import from_db, rows_as_dicts, to_db
from jtrainer.grading import Judge, Verdict, grade, is_pass
from jtrainer.llm import CachedLLM, LLMClient
from jtrainer.model.ability import clue_difficulty_sql, replay
from jtrainer.model.priority import rank_topics, topic_frequencies
from jtrainer.model.review import ReviewScheduler
from jtrainer.sampling import (
    AdaptiveSampler,
    Pick,
    Pool,
    RandomSampler,
    ReviewSampler,
    scheduler_review_source,
)
from jtrainer.tagging.entities import aliases_for_clue, entity_for_clue
from jtrainer.tagging.taxonomy import current_tags_sql

MODES = ("random", "adaptive", "review", "game")


class AttemptRepository:
    """Data access for `attempts`. Deliberately exposes no delete."""

    COLUMNS = (
        "session_id",
        "clue_id",
        "shown_at",
        "first_key_ms",
        "submit_ms",
        "user_response",
        "confidence",
        "correct",
        "grader",
        "grader_detail",
        "overridden",
        "needs_review",
        "sampling_mode",
        "sampling_prob",
        "wager",
    )

    def __init__(self, conn: duckdb.DuckDBPyConnection):
        self.conn = conn

    def insert(self, **values: Any) -> int:
        values = {**values, "shown_at": to_db(values["shown_at"])}
        if isinstance(values.get("grader_detail"), dict):
            values["grader_detail"] = json.dumps(values["grader_detail"], sort_keys=True)
        cols = ", ".join(self.COLUMNS)
        marks = ", ".join("?" * len(self.COLUMNS))
        return self.conn.execute(
            f"INSERT INTO attempts ({cols}) VALUES ({marks}) RETURNING attempt_id",
            [values.get(c) for c in self.COLUMNS],
        ).fetchone()[0]

    def get(self, attempt_id: int) -> dict:
        rows = rows_as_dicts(
            self.conn.execute("SELECT * FROM attempts WHERE attempt_id = ?", [attempt_id])
        )
        if not rows:
            raise KeyError(f"no attempt {attempt_id}")
        row = rows[0]
        row["grader_detail"] = json.loads(row["grader_detail"]) if row["grader_detail"] else {}
        return row

    def set_verdict(
        self,
        attempt_id: int,
        *,
        correct: bool | None,
        grader: str,
        detail: dict,
        overridden: bool,
        needs_review: bool,
    ) -> None:
        self.conn.execute(
            "UPDATE attempts SET correct = ?, grader = ?, grader_detail = ?, overridden = ?, "
            "needs_review = ? WHERE attempt_id = ?",
            [
                correct,
                grader,
                json.dumps(detail, sort_keys=True),
                overridden,
                needs_review,
                attempt_id,
            ],
        )

    def for_session(self, session_id: int) -> list[dict]:
        return rows_as_dicts(
            self.conn.execute(
                "SELECT * FROM attempts WHERE session_id = ? ORDER BY attempt_id", [session_id]
            )
        )


@dataclass
class Presented:
    clue: dict
    pick: Pick
    shown_at: dt.datetime


@dataclass
class AttemptOutcome:
    attempt_id: int
    verdict: Verdict
    clue: dict


def clue_record(conn: duckdb.DuckDBPyConnection, clue_id: int) -> dict:
    rows = rows_as_dicts(conn.execute("SELECT * FROM clues WHERE clue_id = ?", [clue_id]))
    if not rows:
        raise KeyError(f"no clue {clue_id}")
    return rows[0]


class StudyService:
    def __init__(
        self,
        conn: duckdb.DuckDBPyConnection,
        config: Config,
        clock: Clock | None = None,
        rng: np.random.Generator | None = None,
        llm_client: LLMClient | None = None,
    ):
        self.conn = conn
        self.config = config
        self.clock = clock or SystemClock()
        self.rng = rng if rng is not None else np.random.default_rng()
        self.attempts = AttemptRepository(conn)
        self.judge = None
        if llm_client is not None:
            self.judge = Judge(
                CachedLLM(llm_client, conn, self.clock),
                config.llm.judge_model,
                config.llm.judge_max_tokens,
            )
        self.review = ReviewScheduler(conn, config, self.clock)
        self.ability, _ = replay(conn, config)
        self.session_id: int | None = None
        self.mode: str | None = None
        self._pool: Pool | None = None
        self._sampler = None
        self._freqs: dict[str, float] | None = None

    # -- sessions ---------------------------------------------------------------

    def start_session(self, mode: str) -> int:
        if mode not in MODES:
            raise ValueError(f"unknown mode {mode!r}")
        if self.session_id is not None:
            self.end_session()
        self.session_id = self.conn.execute(
            "INSERT INTO sessions (started_at, mode, app_version) VALUES (?, ?, ?) "
            "RETURNING session_id",
            [to_db(self.clock.now()), mode, __version__],
        ).fetchone()[0]
        self.mode = mode
        self._sampler = None
        if self._pool is not None:
            self._pool.served[:] = False
        return self.session_id

    def end_session(self) -> None:
        if self.session_id is None:
            return
        self.conn.execute(
            "UPDATE sessions SET ended_at = ? WHERE session_id = ?",
            [to_db(self.clock.now()), self.session_id],
        )
        self.session_id = None
        self.mode = None
        self._sampler = None

    @property
    def pool(self) -> Pool:
        if self._pool is None:
            self._pool = Pool.from_db(self.conn, self.config)
        return self._pool

    def topic_priorities(self) -> dict[str, float]:
        if self._freqs is None:
            self._freqs = topic_frequencies(self.conn, self.config)
        return {
            tp.topic_id: tp.priority for tp in rank_topics(self._freqs, self.ability, self.config)
        }

    def _make_sampler(self):
        if self.mode == "random":
            return RandomSampler(self.pool, self.rng)
        if self.mode == "review":
            return ReviewSampler(self.pool, self.rng, scheduler_review_source(self.review))
        if self.mode == "adaptive":
            return AdaptiveSampler(
                self.pool,
                self.rng,
                self.config.sampling.mix,
                self.config.sampling.difficulty_kernel_sd,
                lambda: self.ability,
                self.topic_priorities,
                scheduler_review_source(self.review),
            )
        raise ValueError(f"mode {self.mode!r} has no clue sampler")

    def session_attempt_count(self) -> int:
        if self.session_id is None:
            return 0
        return self.conn.execute(
            "SELECT count(*) FROM attempts WHERE session_id = ?", [self.session_id]
        ).fetchone()[0]

    def session_done(self) -> bool:
        return (
            self.mode in ("random", "adaptive")
            and self.session_attempt_count() >= self.config.sampling.session_length
        )

    # -- clues ------------------------------------------------------------------

    def next_clue(self) -> Presented | None:
        if self.session_id is None:
            raise RuntimeError("no active session")
        if self._sampler is None:
            self._sampler = self._make_sampler()
        pick = self._sampler.draw()
        if pick is None:
            return None
        self.pool.mark_served(pick.clue_id)
        return Presented(clue_record(self.conn, pick.clue_id), pick, self.clock.now())

    def present(self, clue_id: int, mode: str, prob: float = 1.0) -> Presented:
        """Present a specific clue (game mode chooses clues by board position)."""
        return Presented(
            clue_record(self.conn, clue_id), Pick(clue_id, mode, prob), self.clock.now()
        )

    # -- grading ----------------------------------------------------------------

    def grade_response(self, clue: dict, user_response: str | None) -> Verdict:
        aliases = aliases_for_clue(self.conn, clue["clue_id"], self.config.taxonomy.version)
        return grade(
            clue["response"],
            user_response,
            self.config.grading,
            aliases=aliases,
            judge=self.judge,
            clue_text=clue["clue_text"],
            category=clue.get("category") or "",
        )

    def submit(
        self,
        presented: Presented,
        user_response: str | None,
        confidence: int | None,
        first_key_ms: int | None,
        submit_ms: int | None,
        *,
        passed: bool = False,
        wager: int | None = None,
    ) -> AttemptOutcome:
        if self.session_id is None:
            raise RuntimeError("no active session")
        if passed or is_pass(user_response):
            verdict = Verdict(False, "exact", {"reason": "pass"})
            confidence = None
        else:
            if confidence not in (1, 2, 3):
                raise ValueError("a confidence of 1, 2 or 3 is required before grading")
            verdict = self.grade_response(presented.clue, user_response)
        if first_key_ms is not None and submit_ms is not None:
            first_key_ms = min(first_key_ms, submit_ms)
        attempt_id = self.attempts.insert(
            session_id=self.session_id,
            clue_id=presented.clue["clue_id"],
            shown_at=presented.shown_at,
            first_key_ms=first_key_ms,
            submit_ms=submit_ms,
            user_response=(user_response or "").strip() or None,
            confidence=confidence,
            correct=verdict.correct,
            grader=verdict.grader,
            grader_detail=verdict.detail,
            overridden=False,
            needs_review=verdict.needs_review,
            sampling_mode=presented.pick.mode,
            sampling_prob=presented.pick.prob,
            wager=wager,
        )
        if verdict.correct is not None:
            self._after_verdict(
                attempt_id, presented.clue["clue_id"], verdict.correct, presented.shown_at
            )
        return AttemptOutcome(attempt_id, verdict, presented.clue)

    def _clue_topic_and_b(self, clue_id: int) -> tuple[str | None, float]:
        row = self.conn.execute(
            f"SELECT t.topic_id, {clue_difficulty_sql(self.config.difficulty)} FROM clues c "
            "LEFT JOIN clue_difficulty d USING (clue_id) "
            f"LEFT JOIN ({current_tags_sql(self.config.taxonomy.version)}) t USING (clue_id) "
            "WHERE c.clue_id = ?",
            [clue_id],
        ).fetchone()
        return row[0], float(row[1])

    def _after_verdict(
        self, attempt_id: int, clue_id: int, correct: bool, shown_at: dt.datetime
    ) -> None:
        topic, b = self._clue_topic_and_b(clue_id)
        self.ability.update(topic, b, correct, shown_at)
        self.review.on_attempt(attempt_id)

    def _after_change(self, attempt_id: int) -> None:
        """An earlier verdict changed: rebuild derived state from attempts."""
        row = self.attempts.get(attempt_id)
        self.ability, _ = replay(self.conn, self.config)
        entity = entity_for_clue(
            self.conn, row["clue_id"], self.config.taxonomy.version, self.review.resolver
        )
        if entity is not None:
            self.review.rebuild_entity(entity)

    def override(self, attempt_id: int) -> bool:
        """Flip the verdict. The machine grader's original verdict is kept in grader_detail."""
        row = self.attempts.get(attempt_id)
        if row["correct"] is None:
            raise ValueError("attempt is awaiting review; use resolve()")
        detail = row["grader_detail"]
        if not row["overridden"]:
            detail = {
                "original": {"grader": row["grader"], "correct": row["correct"], "detail": detail},
                "overrides": [],
            }
        new = not row["correct"]
        detail["overrides"].append({"at": self.clock.now().isoformat(), "to": new})
        self.attempts.set_verdict(
            attempt_id,
            correct=new,
            grader="user",
            detail=detail,
            overridden=True,
            needs_review=False,
        )
        self._after_change(attempt_id)
        return new

    def resolve(self, attempt_id: int, correct: bool) -> None:
        """The user decides an attempt the pipeline could not grade."""
        row = self.attempts.get(attempt_id)
        if not row["needs_review"]:
            raise ValueError("attempt does not need review")
        detail = {
            **row["grader_detail"],
            "resolved_by_user": correct,
            "pipeline_grader": row["grader"],
        }
        self.attempts.set_verdict(
            attempt_id,
            correct=correct,
            grader="user",
            detail=detail,
            overridden=False,
            needs_review=False,
        )
        self._after_change(attempt_id)

    # -- summaries ----------------------------------------------------------------

    def session_summary(self, session_id: int | None = None) -> dict:
        sid = session_id or self.session_id
        rows = self.attempts.for_session(sid) if sid else []
        graded = [r for r in rows if r["correct"] is not None]
        n_right = sum(1 for r in graded if r["correct"])
        times = sorted(r["submit_ms"] for r in rows if r["submit_ms"] is not None)
        return {
            "session_id": sid,
            "attempts": len(rows),
            "graded": len(graded),
            "correct": n_right,
            "accuracy": n_right / len(graded) if graded else None,
            "pending_review": sum(1 for r in rows if r["needs_review"]),
            "passes": sum(1 for r in rows if r["user_response"] is None),
            "median_ms": times[len(times) // 2] if times else None,
        }

    def counts(self) -> dict:
        c = self.conn
        return {
            "clues": c.execute("SELECT count(*) FROM clues").fetchone()[0],
            "eligible": len(self.pool),
            "tagged": c.execute(
                f"SELECT count(*) FROM ({current_tags_sql(self.config.taxonomy.version)})"
            ).fetchone()[0],
            "attempts": c.execute("SELECT count(*) FROM attempts").fetchone()[0],
            "due": self.review.due_count(),
            "queued": self.review.queue_size(),
        }

    def last_session_at(self) -> dt.datetime | None:
        row = self.conn.execute("SELECT max(started_at) FROM sessions").fetchone()
        return from_db(row[0]) if row else None
