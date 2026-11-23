"""Grading pipeline (TESTING §6.2-6.3)."""

import json
from pathlib import Path

import pytest
import yaml

from jtrainer.config import Grading
from jtrainer.grading import Judge, grade, is_pass
from jtrainer.llm import CachedLLM, LLMUnavailable

GOLDEN = Path(__file__).parents[1] / "fixtures" / "grading_golden.yaml"
CFG = Grading(fuzzy_high=92.0, fuzzy_low=60.0, min_len_fuzzy=5)


class Boom:
    """An LLM client that must not be called."""

    def complete(self, request):
        raise AssertionError("LLM must not be called")


def make_judge(db, client, clock=None):
    return Judge(CachedLLM(client, db, clock), "test-judge")


def judge_request(judge, response, user, clue="clue", category="CAT"):
    return judge.request(clue_text=clue, category=category, response=response, user_response=user)


def test_exact_match_no_fuzzy_or_llm(db):
    v = grade("(Abraham) Lincoln", "who is Lincoln?", CFG, judge=make_judge(db, Boom()))
    assert v.correct is True and v.grader == "exact" and not v.needs_review


def test_alias_match(db):
    v = grade(
        "Abraham Lincoln", "Honest Abe", CFG, aliases=["Honest Abe"], judge=make_judge(db, Boom())
    )
    assert v.correct is True and v.grader == "exact"


def test_fuzzy_high_correct(db):
    v = grade("Dostoevsky", "Dostoyevsky", CFG, judge=make_judge(db, Boom()))
    assert v.correct is True and v.grader == "fuzzy" and v.detail["score"] >= 92


def test_fuzzy_low_incorrect_no_llm(db):
    v = grade("Paris", "Rome", CFG, judge=make_judge(db, Boom()))
    assert v.correct is False and v.grader == "fuzzy" and not v.needs_review


def test_band_calls_llm_once_and_stores_rationale(db, fake_llm):
    judge = make_judge(db, fake_llm)
    fake_llm.register(
        judge_request(judge, "Australia", "Austria"),
        json.dumps({"verdict": "incorrect", "rationale": "Different country."}),
    )
    v = grade("Australia", "Austria", CFG, judge=judge, clue_text="clue", category="CAT")
    assert len(fake_llm.calls) == 1
    assert v.correct is False and v.grader == "llm"
    assert v.detail["rationale"] == "Different country."


def test_band_repeat_served_from_cache(db, fake_llm):
    judge = make_judge(db, fake_llm)
    fake_llm.register(
        judge_request(judge, "Tchaikovsky", "Chaikovsky"),
        json.dumps({"verdict": "correct", "rationale": "Spelling variant."}),
    )
    for _ in range(3):
        v = grade("Tchaikovsky", "Chaikovsky", CFG, judge=judge, clue_text="clue", category="CAT")
        assert v.correct is True and v.grader == "llm"
    assert len(fake_llm.calls) == 1
    # a fresh judge over the same database also hits the cache
    judge2 = make_judge(db, fake_llm)
    grade("Tchaikovsky", "Chaikovsky", CFG, judge=judge2, clue_text="clue", category="CAT")
    assert len(fake_llm.calls) == 1


def test_malformed_retry_then_needs_review(db, fake_llm):
    judge = make_judge(db, fake_llm)
    fake_llm.register(judge_request(judge, "Iran", "Iraq"), "not json", "still not json")
    v = grade("Iran", "Iraq", CFG, judge=judge, clue_text="clue", category="CAT")
    assert len(fake_llm.calls) == 2  # one retry
    assert v.needs_review and v.correct is None  # never a silent default verdict
    assert v.detail["review_reason"] == "malformed_llm_output"


def test_malformed_then_valid(db, fake_llm):
    judge = make_judge(db, fake_llm)
    fake_llm.register(
        judge_request(judge, "Iran", "Iraq"),
        "oops",
        '{"verdict": "incorrect", "rationale": "Different country."}',
    )
    v = grade("Iran", "Iraq", CFG, judge=judge, clue_text="clue", category="CAT")
    assert v.correct is False and v.grader == "llm" and len(fake_llm.calls) == 2


def test_llm_unavailable_needs_review(db):
    class Down:
        def complete(self, request):
            raise LLMUnavailable("no network")

    v = grade("Iran", "Iraq", CFG, judge=make_judge(db, Down()))
    assert v.needs_review and v.detail["review_reason"] == "llm_unavailable"


def test_no_judge_needs_review():
    v = grade("Iran", "Iraq", CFG, judge=None)
    assert v.needs_review and v.correct is None


@pytest.mark.parametrize("user", ["", "   ", None, "pass", "PASS", "?"])
def test_blank_and_pass_incorrect_without_grading(db, user):
    v = grade("Paris", user, CFG, judge=make_judge(db, Boom()))
    assert v.correct is False and v.grader == "exact" and v.detail["reason"] == "pass"
    assert is_pass(user)


def test_thresholds_from_config_move_the_band():
    # "Nostradamas" vs "Nostradamus" scores ~90.9
    assert grade("Nostradamus", "Nostradamas", CFG).needs_review
    lenient = Grading(fuzzy_high=90.0, fuzzy_low=60.0, min_len_fuzzy=5)
    assert grade("Nostradamus", "Nostradamas", lenient).correct is True
    # low threshold: "Paris"/"Parma" (~60) is incorrect at low=60? raise low and it moves
    strict_low = Grading(fuzzy_high=92.0, fuzzy_low=20.0, min_len_fuzzy=5)
    assert grade("Paris", "Rome", strict_low).needs_review


def test_near_misses_never_fuzzy_correct():
    for resp, user in [
        ("Australia", "Austria"),
        ("Russia", "Prussia"),
        ("Henry V", "Henry IV"),
        ("Sweden", "Swede"),
        ("1812", "1814"),
    ]:
        v = grade(resp, user, CFG)
        assert v.correct is not True, (resp, user)


def golden_cases():
    return yaml.safe_load(GOLDEN.read_text())


def test_golden_set_size():
    assert len(golden_cases()) >= 100


def test_golden_no_llm():
    """Steps 1-3 only: no case may be graded confidently wrong. Each either matches the
    expected verdict or falls into the LLM band."""
    wrong = []
    for case in golden_cases():
        v = grade(case["response"], case["user_response"], CFG)
        if not v.needs_review and v.correct != case["expected"]:
            wrong.append((case["response"], case["user_response"], v.correct, v.detail))
    assert not wrong, wrong
