"""Normalization (TESTING §6.1): table-driven cases plus Hypothesis properties."""

import pytest
from hypothesis import given
from hypothesis import strategies as st

from jtrainer.text import (
    clean_text,
    normalize,
    response_key,
    response_variants,
    year_words,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("PARIS", "paris"),
        ("Paris!", "paris"),
        ("  New   York  ", "new york"),
        ("rock & roll", "rock and roll"),
        ("U.S.A.", "usa"),
        ("O'Brien", "obrien"),
        ("Jean-Paul Sartre", "jean paul sartre"),
    ],
)
def test_case_punctuation_whitespace(raw, expected):
    assert normalize(raw) == expected


@pytest.mark.parametrize(
    "prefix",
    [
        "what is",
        "What is",
        "who is",
        "who are",
        "what are",
        "what's",
        "What’s",
        "where is",
        "where are",
        "who was",
        "what were",
        "who's",
        "when is",
    ],
)
@pytest.mark.parametrize("suffix", ["", "?", " ?"])
def test_prefixes_stripped(prefix, suffix):
    assert normalize(f"{prefix} the Louvre{suffix}") == "louvre"


@pytest.mark.parametrize("article", ["the", "a", "an", "The", "A", "An"])
def test_leading_articles_stripped(article):
    assert normalize(f"{article} Bounty") == "bounty"


def test_inner_articles_kept():
    assert normalize("Gone with the Wind") == "gone with the wind"


def test_parenthetical_optional():
    variants = response_variants("(Abraham) Lincoln")
    assert normalize("Lincoln") in variants
    assert normalize("Abraham Lincoln") in variants


def test_parenthetical_trailing():
    variants = response_variants("the Louvre (Museum)")
    assert {"louvre", "louvre museum"} <= variants


def test_or_alternatives():
    variants = response_variants("Mars (or the Red Planet)")
    assert "mars" in variants and "red planet" in variants
    assert "malvinas islands" in response_variants("Falkland (or Malvinas) Islands")


def test_accent_folding_decision():
    # Decision: accents are folded for matching. "Dali" matches "Dalí".
    assert normalize("Dalí") == normalize("Dali") == "dali"
    assert normalize("Antônio") == "antonio"


def test_numbers_expand():
    assert "nineteen eighty four" in response_variants("1984")
    assert "nineteen eighty four" in response_variants("Nineteen Eighty-Four")
    assert year_words(2005) is None and year_words(1905) == "nineteen oh five"


def test_response_key_resolves_parentheticals():
    assert response_key("(Abraham) Lincoln") == response_key("Abraham Lincoln")
    assert response_key("Mars (or the Red Planet)") == "mars"


def test_clean_text():
    assert clean_text('\\"Thriller\\"') == '"Thriller"'
    assert clean_text("Ant&ocirc;nio  ") == "Antônio"
    assert clean_text(None) == ""


@given(st.text())
def test_property_idempotent(x):
    assert normalize(normalize(x)) == normalize(x)


@given(st.text())
def test_property_prefix_invariant(x):
    assert normalize("what is " + x) == normalize(x)


@given(st.text(alphabet=st.characters(), max_size=200))
def test_property_never_raises(x):
    normalize(x)
    response_variants(x)
    response_key(x)
