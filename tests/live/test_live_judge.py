"""Live LLM judge (TESTING §14 and §6.3 golden_live). Run manually: uv run pytest -m live"""

from pathlib import Path

import pytest
import yaml

from jtrainer import db as dbmod
from jtrainer.clock import SystemClock
from jtrainer.config import Config
from jtrainer.grading import Judge, grade
from jtrainer.llm import AnthropicClient, CachedLLM

pytestmark = pytest.mark.live

ROOT = Path(__file__).parents[2]
GOLDEN = yaml.safe_load((ROOT / "tests" / "fixtures" / "grading_golden.yaml").read_text())
FLOOR = 0.95


@pytest.fixture(scope="module")
def judge(tmp_path_factory):
    cfg = Config.load(ROOT / "config.toml")
    conn = dbmod.connect(tmp_path_factory.mktemp("live") / "judge.duckdb")
    return cfg, Judge(
        CachedLLM(AnthropicClient(cfg.llm.effort), conn, SystemClock()), cfg.llm.judge_model
    )


def test_live_judge_borderline(judge):
    _cfg, j = judge
    borderline = [c for c in GOLDEN if "judge" in c["note"]][:10]
    agree = 0
    for c in borderline:
        ok, _ = j.judge(
            clue_text=c["clue_text"],
            category="",
            response=c["response"],
            user_response=c["user_response"],
        )
        agree += ok == c["expected"]
    assert agree >= 9


def test_golden_live(judge):
    cfg, j = judge
    right = 0
    misses = []
    for c in GOLDEN:
        v = grade(c["response"], c["user_response"], cfg.grading, judge=j, clue_text=c["clue_text"])
        if v.correct == c["expected"]:
            right += 1
        else:
            misses.append((c["response"], c["user_response"], v.correct, v.detail))
    accuracy = right / len(GOLDEN)
    print(f"golden live accuracy {accuracy:.1%}")
    assert accuracy >= FLOOR, misses
