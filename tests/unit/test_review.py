"""Spaced review (TESTING §8.4)."""

import datetime as dt

import fsrs
import numpy as np
import pytest
from conftest import clue_id_by_response, make_config

from jtrainer.model.review import (
    ReviewScheduler,
    card_from_json,
    card_to_json,
    qualifies,
    rating_for,
)
from jtrainer.tagging.entities import entity_for_clue

SLOW = 12000


@pytest.mark.parametrize(
    ("correct", "conf", "ms", "enters"),
    [
        (False, 3, 2000, True),  # a miss
        (True, 1, 2000, True),  # correct but a guess
        (True, 3, SLOW + 1, True),  # correct but slow
        (True, 3, 2000, False),  # fast, confident, correct
        (True, 2, 2000, False),
    ],
)
def test_queue_entry_rules(correct, conf, ms, enters):
    assert qualifies(correct, conf, ms, SLOW) is enters


def test_rating_map():
    assert rating_for(False, 3, 1000, SLOW) == fsrs.Rating.Again
    assert rating_for(True, 1, 1000, SLOW) == fsrs.Rating.Hard
    assert rating_for(True, 2, 1000, SLOW) == fsrs.Rating.Good
    assert rating_for(True, 3, 1000, SLOW) == fsrs.Rating.Easy


def test_card_json_round_trip():
    card = fsrs.Card(card_id=7, due=dt.datetime(2026, 1, 1, tzinfo=dt.UTC))
    card, _ = fsrs.Scheduler(enable_fuzzing=False).review_card(
        card, fsrs.Rating.Good, review_datetime=dt.datetime(2026, 1, 1, tzinfo=dt.UTC)
    )
    text = card_to_json(card)
    assert card_to_json(card_from_json(text)) == text


class Harness:
    def __init__(self, db, clock):
        self.db = db
        self.clock = clock
        self.cfg = make_config()
        self.sched = ReviewScheduler(db, self.cfg, clock)
        db.execute(
            "INSERT INTO sessions (started_at, mode, app_version) VALUES "
            "('2026-01-01', 'random', 'test')"
        )

    def attempt(self, response, correct, conf=2, ms=3000) -> int:
        cid = clue_id_by_response(self.db, response)
        aid = self.db.execute(
            "INSERT INTO attempts (session_id, clue_id, shown_at, submit_ms, confidence, "
            "correct, grader, sampling_mode, sampling_prob) VALUES (1, ?, ?, ?, ?, ?, 'exact', "
            "'random', 0.1) RETURNING attempt_id",
            [cid, self.clock.now().replace(tzinfo=None), ms, conf, correct],
        ).fetchone()[0]
        self.sched.on_attempt(aid)
        return aid

    def entity(self, response):
        return entity_for_clue(
            self.db, clue_id_by_response(self.db, response), self.cfg.taxonomy.version
        )


def test_miss_queues_fast_correct_does_not(db_loaded, clock):
    h = Harness(db_loaded, clock)
    h.attempt("gold", False)
    h.attempt("iron", True, conf=3, ms=1500)
    assert h.sched.queue_size() == 1
    assert h.sched.card(h.entity("gold")) is not None
    assert h.sched.card(h.entity("iron")) is None


def test_due_order_and_not_yet_due(db_loaded, clock):
    h = Harness(db_loaded, clock)
    h.attempt("gold", False)
    clock.advance(3_600_000)
    h.attempt("iron", False)
    assert h.sched.due() == []  # nothing due immediately
    clock.advance(3 * 86_400_000)
    due = h.sched.due()
    assert [d.entity_id for d in due] == [h.entity("gold"), h.entity("iron")]
    assert due[0].due_at <= due[1].due_at


def test_advancing_past_due_makes_due(db_loaded, clock):
    h = Harness(db_loaded, clock)
    h.attempt("gold", False)
    card = h.sched.card(h.entity("gold"))
    assert h.sched.due_count() == 0
    clock.advance((card.due - clock.now()).total_seconds() * 1000 + 1)
    assert h.sched.due_count() == 1


def test_review_presents_different_clue(db_loaded, clock, rng):
    h = Harness(db_loaded, clock)
    # Two clues share the Lincoln entity: "(Abraham) Lincoln" and "Abraham Lincoln" (a DD)
    h.attempt("(Abraham) Lincoln", False)
    clock.advance(30 * 86_400_000)
    _item, clue_id, n = h.sched.pick(rng)
    assert clue_id == clue_id_by_response(db_loaded, "Abraham Lincoln")
    assert n == 1


def test_review_avoids_most_recent_clue(db_loaded, clock, rng):
    h = Harness(db_loaded, clock)
    h.attempt("(Abraham) Lincoln", False)
    clock.advance(30 * 86_400_000)
    h.attempt("Abraham Lincoln", False)  # now both clues have been seen
    clock.advance(30 * 86_400_000)
    for seed in range(10):
        _, clue_id, _ = h.sched.pick(np.random.default_rng(seed))
        assert clue_id == clue_id_by_response(db_loaded, "(Abraham) Lincoln")


def test_single_clue_entity_falls_back(db_loaded, clock, rng):
    h = Harness(db_loaded, clock)
    h.attempt("hydrargyrum", False)
    clock.advance(30 * 86_400_000)
    _, clue_id, n = h.sched.pick(rng)
    assert clue_id == clue_id_by_response(db_loaded, "hydrargyrum") and n == 1


def test_rebuild_matches_incremental(db_loaded, clock):
    h = Harness(db_loaded, clock)
    h.attempt("gold", False)
    clock.advance(5 * 86_400_000)
    h.attempt("gold", True, conf=3, ms=2000)
    eid = h.entity("gold")
    before = card_to_json(h.sched.card(eid))
    rebuilt = h.sched.rebuild_entity(eid)
    assert card_to_json(rebuilt) == before
