"""Spaced review (DESIGN §8.3): FSRS per answer *entity*, presented through varied clues.

An entity enters the queue on a miss, a correct response with confidence 1 (a guess), or a
correct response slower than `review.slow_ms`. Once queued, every later attempt on any clue
for that entity is a review. `review_state` is a cache that can always be rebuilt from the
attempts table (`rebuild_entity`), which is how overrides propagate.
"""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass

import duckdb
import fsrs
import numpy as np

from jtrainer.clock import Clock, SystemClock
from jtrainer.config import Config
from jtrainer.db import from_db, to_db
from jtrainer.model.priority import eligibility_sql
from jtrainer.tagging.entities import EntityResolver, clues_for_entity, entity_for_clue


@dataclass
class DueItem:
    entity_id: int
    due_at: dt.datetime
    last_clue_id: int | None
    canonical_name: str


def qualifies(correct: bool, confidence: int | None, submit_ms: int | None, slow_ms: int) -> bool:
    if not correct:
        return True
    if confidence == 1:
        return True
    return submit_ms is not None and submit_ms > slow_ms


def rating_for(
    correct: bool, confidence: int | None, submit_ms: int | None, slow_ms: int
) -> fsrs.Rating:
    if not correct:
        return fsrs.Rating.Again
    if confidence == 1 or (submit_ms is not None and submit_ms > slow_ms):
        return fsrs.Rating.Hard
    if confidence == 3:
        return fsrs.Rating.Easy
    return fsrs.Rating.Good


def card_to_json(card: fsrs.Card) -> str:
    return card.to_json()


def card_from_json(text: str) -> fsrs.Card:
    return fsrs.Card.from_json(text)


class ReviewScheduler:
    def __init__(self, conn: duckdb.DuckDBPyConnection, config: Config, clock: Clock | None = None):
        self.conn = conn
        self.config = config
        self.cfg = config.review
        self.clock = clock or SystemClock()
        self.version = config.taxonomy.version
        self.resolver = EntityResolver(conn)
        self.scheduler = fsrs.Scheduler(
            desired_retention=self.cfg.desired_retention,
            learning_steps=tuple(dt.timedelta(minutes=m) for m in self.cfg.learning_steps_minutes),
            relearning_steps=tuple(
                dt.timedelta(minutes=m) for m in self.cfg.relearning_steps_minutes
            ),
            enable_fuzzing=False,
        )

    # -- state ------------------------------------------------------------------

    def card(self, entity_id: int) -> fsrs.Card | None:
        row = self.conn.execute(
            "SELECT fsrs_card FROM review_state WHERE entity_id = ?", [entity_id]
        ).fetchone()
        return card_from_json(row[0]) if row else None

    def _save(self, entity_id: int, card: fsrs.Card, attempt_id: int, clue_id: int) -> None:
        self.conn.execute(
            "INSERT INTO review_state VALUES (?, ?, ?, ?, ?) ON CONFLICT (entity_id) DO UPDATE "
            "SET fsrs_card = excluded.fsrs_card, due_at = excluded.due_at, "
            "last_attempt_id = excluded.last_attempt_id, last_clue_id = excluded.last_clue_id",
            [entity_id, card_to_json(card), to_db(card.due), attempt_id, clue_id],
        )

    def _apply(self, entity_id: int, attempt: dict, card: fsrs.Card | None) -> fsrs.Card | None:
        correct = attempt["correct"]
        if correct is None:
            return card
        args = (bool(correct), attempt["confidence"], attempt["submit_ms"], self.cfg.slow_ms)
        if card is None:
            if not qualifies(*args):
                return None
            at = from_db(attempt["shown_at"])
            card = fsrs.Card(card_id=entity_id, due=at)
        rating = rating_for(*args)
        card, _ = self.scheduler.review_card(
            card, rating, review_datetime=from_db(attempt["shown_at"])
        )
        self._save(entity_id, card, attempt["attempt_id"], attempt["clue_id"])
        return card

    def _attempt(self, attempt_id: int) -> dict:
        row = self.conn.execute(
            "SELECT attempt_id, clue_id, shown_at, correct, confidence, submit_ms "
            "FROM attempts WHERE attempt_id = ?",
            [attempt_id],
        ).fetchone()
        keys = ["attempt_id", "clue_id", "shown_at", "correct", "confidence", "submit_ms"]
        return dict(zip(keys, row, strict=True))

    def on_attempt(self, attempt_id: int) -> int | None:
        """Update review state after a graded attempt. Returns the entity id if queued."""
        attempt = self._attempt(attempt_id)
        entity_id = entity_for_clue(self.conn, attempt["clue_id"], self.version, self.resolver)
        if entity_id is None:
            return None
        card = self._apply(entity_id, attempt, self.card(entity_id))
        return entity_id if card is not None else None

    def rebuild_entity(self, entity_id: int) -> fsrs.Card | None:
        """Recompute an entity's card from its attempts (used after overrides)."""
        self.conn.execute("DELETE FROM review_state WHERE entity_id = ?", [entity_id])
        clue_ids = clues_for_entity(self.conn, entity_id, self.version)
        if not clue_ids:
            return None
        placeholders = ", ".join("?" * len(clue_ids))
        rows = self.conn.execute(
            "SELECT attempt_id, clue_id, shown_at, correct, confidence, submit_ms FROM attempts "
            f"WHERE clue_id IN ({placeholders}) ORDER BY shown_at, attempt_id",
            clue_ids,
        ).fetchall()
        keys = ["attempt_id", "clue_id", "shown_at", "correct", "confidence", "submit_ms"]
        card = None
        for row in rows:
            card = self._apply(entity_id, dict(zip(keys, row, strict=True)), card)
        return card

    # -- queue ------------------------------------------------------------------

    def due(self, now: dt.datetime | None = None, limit: int = 1000) -> list[DueItem]:
        now = now or self.clock.now()
        rows = self.conn.execute(
            "SELECT r.entity_id, r.due_at, r.last_clue_id, e.canonical_name FROM review_state r "
            "JOIN entities e USING (entity_id) WHERE r.due_at <= ? "
            "ORDER BY r.due_at, r.entity_id LIMIT ?",
            [to_db(now), limit],
        ).fetchall()
        return [DueItem(r[0], from_db(r[1]), r[2], r[3]) for r in rows]

    def due_count(self, now: dt.datetime | None = None) -> int:
        now = now or self.clock.now()
        return self.conn.execute(
            "SELECT count(*) FROM review_state WHERE due_at <= ?", [to_db(now)]
        ).fetchone()[0]

    def queue_size(self) -> int:
        return self.conn.execute("SELECT count(*) FROM review_state").fetchone()[0]

    def candidate_clues(
        self, item: DueItem, exclude: set[int] | frozenset = frozenset()
    ) -> list[int]:
        """Clues to present for a due entity: prefer never-attempted clues, then any clue
        other than the most recently shown one, then (single-clue entity) that clue."""
        clue_ids = clues_for_entity(self.conn, item.entity_id, self.version)
        if not clue_ids:
            return []
        placeholders = ", ".join("?" * len(clue_ids))
        eligible = {
            r[0]
            for r in self.conn.execute(
                f"SELECT clue_id FROM clues c WHERE clue_id IN ({placeholders}) AND "
                f"{eligibility_sql(self.config)}",
                clue_ids,
            ).fetchall()
        }
        attempted = {
            r[0]
            for r in self.conn.execute(
                f"SELECT DISTINCT clue_id FROM attempts WHERE clue_id IN ({placeholders})", clue_ids
            ).fetchall()
        }
        pool = [c for c in clue_ids if c not in exclude]
        tiers = [
            [c for c in pool if c in eligible and c not in attempted],
            [c for c in pool if c in eligible and c != item.last_clue_id],
            [c for c in pool if c != item.last_clue_id],
            pool,
        ]
        return next((t for t in tiers if t), [])

    def pick(
        self,
        rng: np.random.Generator,
        exclude: set[int] | frozenset = frozenset(),
        now: dt.datetime | None = None,
    ) -> tuple[DueItem, int, int] | None:
        """(due item, clue_id, number of candidates) for the most overdue entity."""
        for item in self.due(now):
            cands = self.candidate_clues(item, exclude)
            if cands:
                return item, int(cands[int(rng.integers(len(cands)))]), len(cands)
        return None


def describe_card(card: fsrs.Card) -> dict:
    return json.loads(card.to_json())
