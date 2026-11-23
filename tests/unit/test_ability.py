"""Ability v1 update rule (TESTING §8.1)."""

import datetime as dt

import pytest
from conftest import clue_id_by_response, make_config

from jtrainer.config import Ability, Difficulty
from jtrainer.model.ability import OVERALL, AbilityModel, prior_difficulty, replay

D = Difficulty()


def test_correct_raises_incorrect_lowers():
    m = AbilityModel(Ability())
    m.update("t", 0.0, True)
    assert m.get("t").mu > 0
    m2 = AbilityModel(Ability())
    m2.update("t", 0.0, False)
    assert m2.get("t").mu < 0


def test_surprise_moves_more():
    hard_right = AbilityModel(Ability()).update("t", 2.0, True)
    easy_right = AbilityModel(Ability()).update("t", -2.0, True)
    assert hard_right > easy_right > 0
    easy_wrong = AbilityModel(Ability()).update("t", -2.0, False)
    hard_wrong = AbilityModel(Ability()).update("t", 2.0, False)
    assert easy_wrong < hard_wrong < 0


def test_uncertainty_shrinks_and_stays_positive():
    m = AbilityModel(Ability())
    prev = m.get("t").var
    for i in range(200):
        m.update("t", (i % 5) - 2.0, i % 3 != 0)
        var = m.get("t").var
        assert 0 < var < prev
        prev = var


def test_drift_grows_uncertainty_over_time():
    m = AbilityModel(Ability(drift_sd_per_day=0.1))
    t0 = dt.datetime(2026, 1, 1, tzinfo=dt.UTC)
    m.update("t", 0.0, True, t0)
    v_after = m.get("t").var
    m.update("t", 0.0, True, t0 + dt.timedelta(days=100))
    still = AbilityModel(Ability(drift_sd_per_day=0.0))
    still.update("t", 0.0, True, t0)
    still.update("t", 0.0, True, t0 + dt.timedelta(days=100))
    assert m.get("t").var > still.get("t").var and v_after > 0


def test_other_topics_unchanged():
    m = AbilityModel(Ability())
    m.update("a", 0.0, True)
    before = m.get("b")
    m.update("a", 0.0, False)
    assert m.get("b") == before
    assert "b" not in m.ratings


def test_difficulty_prior_monotone():
    for rnd in (1, 2):
        rows = [prior_difficulty(rnd, r, D) for r in range(1, 6)]
        assert rows == sorted(rows) and len(set(rows)) == 5
    assert prior_difficulty(2, 1, D) > prior_difficulty(1, 1, D)
    assert prior_difficulty(3, None, D) > prior_difficulty(2, 3, D)
    assert prior_difficulty(1, None, D) == prior_difficulty(1, 3, D)


def _attempt(db, clue_id, correct, when, session=1):
    db.execute(
        "INSERT INTO attempts (session_id, clue_id, shown_at, correct, grader, "
        "sampling_mode, sampling_prob) VALUES (?, ?, ?, ?, 'exact', 'random', 0.1)",
        [session, clue_id, when, correct],
    )


def test_untagged_attempts_skipped_and_counted(db_loaded):
    db = db_loaded
    db.execute(
        "INSERT INTO sessions (started_at, mode, app_version) VALUES "
        "('2026-01-01', 'random', 'test')"
    )
    cid = clue_id_by_response(db, "gold")
    db.execute("DELETE FROM clue_tags WHERE clue_id = ?", [cid])
    tagged = clue_id_by_response(db, "iron")
    _attempt(db, cid, True, dt.datetime(2026, 1, 1, 0, 0, 1))
    _attempt(db, tagged, True, dt.datetime(2026, 1, 1, 0, 0, 2))
    model, report = replay(db, make_config())
    assert report.skipped_untagged == 1 and report.used == 1
    assert model.get(OVERALL).n == 2  # untagged still informs the overall rating
    assert model.get("science").n == 1


def test_replay_matches_incremental(db_loaded):
    db = db_loaded
    cfg = make_config()
    db.execute(
        "INSERT INTO sessions (started_at, mode, app_version) VALUES "
        "('2026-01-01', 'random', 'test')"
    )
    clues = db.execute(
        "SELECT c.clue_id, c.round, c.board_row, t.topic_id FROM clues c JOIN "
        "clue_tags t USING (clue_id) ORDER BY clue_id LIMIT 12"
    ).fetchall()
    inc = AbilityModel(cfg.ability)
    for i, (cid, rnd, row, topic) in enumerate(clues):
        when = dt.datetime(2026, 1, 1, 0, 0, i)
        _attempt(db, cid, i % 3 != 0, when)
        inc.update(
            topic,
            prior_difficulty(rnd, row, cfg.difficulty),
            i % 3 != 0,
            when.replace(tzinfo=dt.UTC),
        )
    model, _ = replay(db, cfg)
    for topic in inc.ratings:
        assert model.get(topic).mu == pytest.approx(inc.get(topic).mu)
        assert model.get(topic).var == pytest.approx(inc.get(topic).var)
