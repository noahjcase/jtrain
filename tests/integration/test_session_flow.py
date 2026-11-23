"""End-to-end sessions through the service layer (TESTING §11)."""

import json

import pytest
from conftest import clue_id_by_response, make_config

from jtrainer.grading import Judge
from jtrainer.service import StudyService


@pytest.fixture
def service(db_loaded, clock, rng, fake_llm):
    return StudyService(db_loaded, make_config(), clock, rng, fake_llm)


def answer_wrong(service, p, conf=2):
    return service.submit(p, "zzzz qqqq", conf, 1000, 2500)


def test_twenty_attempts_across_modes(service, db_loaded, clock):
    db = db_loaded
    outcomes = []
    service.start_session("random")
    for i in range(8):
        p = service.next_clue()
        clock.advance(3000)
        if i % 2:
            outcomes.append(service.submit(p, p.clue["response"], 3, 800, 3000))
        else:
            outcomes.append(answer_wrong(service, p))
    service.end_session()

    service.start_session("adaptive")
    for i in range(8):
        p = service.next_clue()
        assert p.pick.mode.startswith("adaptive:")
        clock.advance(3000)
        outcomes.append(
            service.submit(p, None, None, None, 3000, passed=True)
            if i == 0
            else service.submit(p, p.clue["response"], 2, 900, 3000)
        )
    service.end_session()

    clock.advance(40 * 86_400_000)  # missed entities come due
    service.start_session("review")
    reviewed = 0
    while reviewed < 4 and (p := service.next_clue()) is not None:
        assert p.pick.mode == "review"
        outcomes.append(service.submit(p, p.clue["response"], 3, 700, 2000))
        reviewed += 1
    service.end_session()
    assert reviewed == 4

    assert db.execute("SELECT count(*) FROM attempts").fetchone()[0] == 20
    assert db.execute("SELECT count(*) FROM sessions WHERE ended_at IS NULL").fetchone()[0] == 0
    rows = db.execute(
        "SELECT sampling_mode, sampling_prob, grader, correct FROM attempts"
    ).fetchall()
    assert all(0 < r[1] <= 1 for r in rows)
    assert {r[0] for r in rows} >= {"random", "review"}
    assert all(r[3] is not None for r in rows)
    # ability updated for touched topics; review queue holds the missed entities
    assert any(r.n > 0 for t, r in service.ability.ratings.items() if not t.startswith("_"))
    assert service.review.queue_size() >= 4


def test_attempt_row_fully_populated(service, db_loaded):
    service.start_session("random")
    p = service.next_clue()
    out = service.submit(p, p.clue["response"], 2, 1200, 5200)
    row = service.attempts.get(out.attempt_id)
    for col in (
        "session_id",
        "clue_id",
        "shown_at",
        "first_key_ms",
        "submit_ms",
        "user_response",
        "confidence",
        "correct",
        "grader",
        "sampling_mode",
        "sampling_prob",
    ):
        assert row[col] is not None, col
    assert row["first_key_ms"] == 1200 and row["submit_ms"] == 5200


def test_confidence_required(service):
    service.start_session("random")
    p = service.next_clue()
    with pytest.raises(ValueError, match="confidence"):
        service.submit(p, "something", None, 100, 200)


def test_override_and_double_override(service, db_loaded):
    service.start_session("random")
    p = service.next_clue()
    out = service.submit(p, p.clue["response"], 3, 500, 1500)
    assert out.verdict.correct is True and out.verdict.grader == "exact"
    assert service.override(out.attempt_id) is False
    row = service.attempts.get(out.attempt_id)
    assert row["correct"] is False and row["overridden"] and row["grader"] == "user"
    assert row["grader_detail"]["original"] == {
        "grader": "exact",
        "correct": True,
        "detail": out.verdict.detail,
    }
    assert service.override(out.attempt_id) is True
    row = service.attempts.get(out.attempt_id)
    assert row["correct"] is True and row["overridden"]
    assert row["grader_detail"]["original"]["grader"] == "exact"
    assert [o["to"] for o in row["grader_detail"]["overrides"]] == [False, True]


def test_override_updates_review_queue(service, db_loaded):
    service.start_session("random")
    p = service.next_clue()
    out = service.submit(p, p.clue["response"], 3, 500, 1500)  # fast, sure, right
    assert service.review.queue_size() == 0
    service.override(out.attempt_id)  # actually wrong
    assert service.review.queue_size() == 1
    service.override(out.attempt_id)
    assert service.review.queue_size() == 0


def test_judge_band_and_resolve(db_loaded, clock, rng, fake_llm):
    service = StudyService(db_loaded, make_config(), clock, rng, None)  # no LLM configured
    service.start_session("random")
    p = service.present(clue_id_by_response(db_loaded, "Australia"), "random", 0.5)
    out = service.submit(p, "Austria", 2, 300, 900)
    assert out.verdict.needs_review and out.verdict.correct is None
    service.resolve(out.attempt_id, False)
    row = service.attempts.get(out.attempt_id)
    assert row["correct"] is False and row["grader"] == "user" and not row["needs_review"]


def test_judge_called_for_band(db_loaded, clock, rng, fake_llm):
    service = StudyService(db_loaded, make_config(), clock, rng, fake_llm)
    service.start_session("random")
    clue_id = clue_id_by_response(db_loaded, "Australia")
    p = service.present(clue_id, "random", 0.5)
    judge: Judge = service.judge
    fake_llm.register(
        judge.request(
            clue_text=p.clue["clue_text"],
            category=p.clue["category"],
            response="Australia",
            user_response="Austria",
        ),
        json.dumps({"verdict": "incorrect", "rationale": "Different country."}),
    )
    out = service.submit(p, "Austria", 2, 300, 900)
    assert out.verdict.grader == "llm" and out.verdict.correct is False


def test_session_summary(service, clock):
    service.start_session("random")
    for i in range(4):
        p = service.next_clue()
        service.submit(
            p,
            p.clue["response"] if i < 3 else None,
            2 if i < 3 else None,
            100,
            1000 + i,
            passed=i == 3,
        )
    s = service.session_summary()
    assert s["attempts"] == 4 and s["correct"] == 3 and s["accuracy"] == 0.75
    assert s["passes"] == 1
