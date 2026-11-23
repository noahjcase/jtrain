"""Response grading pipeline (DESIGN §7).

normalize -> exact (incl. entity aliases) -> fuzzy (two thresholds) -> LLM judge for the
band in between -> user override. Blank responses and explicit passes are incorrect.
If the judge is unavailable or returns malformed output twice, the attempt is marked
`needs_review` for the user to decide; there is never a silent default verdict.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from rapidfuzz import fuzz

from jtrainer.config import Grading
from jtrainer.llm import (
    CachedLLM,
    LLMRequest,
    LLMUnavailable,
    MalformedResponse,
    load_prompt,
    parse_json_object,
)
from jtrainer.text import normalize, number_variants, response_variants

PASS_WORDS = {"pass", "skip", "idk", "i dont know", "dont know", "no idea", "?"}

NUMBER_WORDS = {
    *"one two three four five six seven eight nine ten eleven twelve thirteen".split(),
    *"fourteen fifteen sixteen seventeen eighteen nineteen twenty thirty forty".split(),
    *"fifty sixty seventy eighty ninety hundred thousand million billion".split(),
    *"first second third fourth fifth sixth seventh eighth ninth tenth".split(),
    "eleventh",
    "twelfth",
}

_ROMAN_RE = re.compile(r"^(?=[mdclxvi])m*(c[md]|d?c{0,3})(x[cl]|l?x{0,3})(i[xv]|v?i{0,3})$")


@dataclass
class Verdict:
    correct: bool | None
    grader: str  # 'exact' | 'fuzzy' | 'llm' | 'user'
    detail: dict[str, Any] = field(default_factory=dict)
    needs_review: bool = False


def is_pass(user_response: str | None) -> bool:
    if user_response is None or not user_response.strip():
        return True
    return normalize(user_response) in PASS_WORDS or user_response.strip() == "?"


def _numberish(tok: str) -> bool:
    if any(ch.isdigit() for ch in tok):
        return True
    return bool(_ROMAN_RE.match(tok)) and (len(tok) >= 2 or tok in {"v", "x"})


def _numberish_tokens(s: str) -> set[str]:
    return {t for t in s.split() if _numberish(t)}


_MINOR_WORDS = {"and", "of", "the", "for", "in", "on", "a", "an", "to", "at"}


def _initials(s: str) -> set[str]:
    """Initialisms of a phrase, with and without minor words ('NASA', 'UN', 'USA')."""
    toks = s.split()
    full = "".join(t[0] for t in toks)
    major = "".join(t[0] for t in toks if t not in _MINOR_WORDS)
    return {full, major} - {""}


def band_triggers(user_norm: str, resp_variants: set[str]) -> list[str]:
    """Reasons a low fuzzy score must not be read as a confident 'incorrect'."""
    reasons = []
    utoks = set(user_norm.split())
    compact = user_norm.replace(" ", "")
    for r in resp_variants:
        rtoks = r.split()
        if utoks and utoks < set(rtoks):
            reasons.append("partial_name")
        elif 0 < len(utoks) < len(rtoks) and all(
            max(fuzz.ratio(u, t) for t in rtoks) >= 80 for u in utoks
        ):
            reasons.append("partial_name")  # e.g. a variant spelling of the surname alone
        if 2 <= len(compact) <= 6 and len(rtoks) >= 2 and compact in _initials(r):
            reasons.append("initials")
        r_compact = r.replace(" ", "")
        if 2 <= len(r_compact) <= 6 and len(utoks) >= 2 and r_compact in _initials(user_norm):
            reasons.append("initials")
        u_num = _numberish_tokens(user_norm) or utoks & NUMBER_WORDS
        r_num = _numberish_tokens(r) or set(rtoks) & NUMBER_WORDS
        if u_num and r_num:
            reasons.append("numbers")
    return sorted(set(reasons))


def _fuzzy_correct_allowed(u: str, r: str, cfg: Grading) -> bool:
    if len(r.replace(" ", "")) < cfg.min_len_fuzzy:
        return False
    if not u or u[0] != r[0]:
        return False
    return _numberish_tokens(u) == _numberish_tokens(r)


class Judge:
    """LLM judge for the fuzzy band, cached through CachedLLM."""

    SCHEMA = {
        "type": "object",
        "properties": {
            "verdict": {"type": "string", "enum": ["correct", "incorrect"]},
            "rationale": {"type": "string"},
        },
        "required": ["verdict", "rationale"],
        "additionalProperties": False,
    }

    def __init__(self, llm: CachedLLM, model: str, max_tokens: int = 2048):
        self.llm = llm
        self.model = model
        self.max_tokens = max_tokens
        self.prompt = load_prompt("judge")

    def request(
        self, *, clue_text: str, category: str, response: str, user_response: str
    ) -> LLMRequest:
        system, user = self.prompt.render(
            clue_text=clue_text,
            category=category,
            response=response,
            user_response=user_response,
        )
        return LLMRequest(
            self.prompt.version, self.model, system, user, self.max_tokens, self.SCHEMA
        )

    @staticmethod
    def parse(text: str) -> tuple[bool, str]:
        obj = parse_json_object(text)
        verdict = obj.get("verdict")
        if verdict not in ("correct", "incorrect"):
            raise MalformedResponse(f"bad verdict {verdict!r}", text)
        return verdict == "correct", str(obj.get("rationale", ""))[:500]

    def judge(self, **kwargs: str) -> tuple[bool, str]:
        return self.llm.complete(self.request(**kwargs), self.parse, retries=1)


def grade(
    response: str,
    user_response: str | None,
    cfg: Grading,
    *,
    aliases: Sequence[str] = (),
    judge: Judge | None = None,
    clue_text: str = "",
    category: str = "",
) -> Verdict:
    if is_pass(user_response):
        return Verdict(False, "exact", {"reason": "pass"})
    assert user_response is not None
    user_norm = normalize(user_response)
    user_vars = number_variants(user_norm) if user_norm else set()
    resp_vars = response_variants(response, aliases)

    matched = user_vars & resp_vars
    if matched:
        return Verdict(True, "exact", {"matched": sorted(matched)[0]})

    best_score, best_pair = 0.0, ("", "")
    for u in user_vars:
        for r in resp_vars:
            s = fuzz.token_sort_ratio(u, r)
            if s > best_score:
                best_score, best_pair = s, (u, r)
    detail: dict[str, Any] = {"score": round(best_score, 1), "closest": best_pair[1]}
    triggers = band_triggers(user_norm, resp_vars)
    if triggers:
        detail["band_triggers"] = triggers

    if best_score >= cfg.fuzzy_high and _fuzzy_correct_allowed(*best_pair, cfg):
        return Verdict(True, "fuzzy", detail)
    if best_score < cfg.fuzzy_low and not triggers:
        return Verdict(False, "fuzzy", detail)

    # The band: ask the judge.
    detail["band"] = True
    if judge is None:
        detail["review_reason"] = "no_judge"
        return Verdict(None, "fuzzy", detail, needs_review=True)
    try:
        ok, rationale = judge.judge(
            clue_text=clue_text,
            category=category,
            response=response,
            user_response=user_response,
        )
    except MalformedResponse as exc:
        detail.update(review_reason="malformed_llm_output", raw=exc.raw[:500])
        return Verdict(None, "llm", detail, needs_review=True)
    except LLMUnavailable as exc:
        detail.update(review_reason="llm_unavailable", error=str(exc)[:200])
        return Verdict(None, "fuzzy", detail, needs_review=True)
    detail["rationale"] = rationale
    return Verdict(ok, "llm", detail)
