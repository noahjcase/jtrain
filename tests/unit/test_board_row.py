"""Board-row derivation (TESTING §5): table-driven and property tests."""

import datetime as dt

import pytest
from hypothesis import given
from hypothesis import strategies as st

from jtrainer.ingest.board import BOARD_VALUES, VALUE_DOUBLING_DATE, board_row, season_for

PRE = dt.date(1999, 5, 10)
ON = VALUE_DOUBLING_DATE
POST = dt.date(2015, 6, 1)


@pytest.mark.parametrize(
    ("round_", "values", "date"),
    [
        (1, [100, 200, 300, 400, 500], PRE),
        (2, [200, 400, 600, 800, 1000], PRE),
        (1, [200, 400, 600, 800, 1000], ON),
        (2, [400, 800, 1200, 1600, 2000], ON),
        (1, [200, 400, 600, 800, 1000], POST),
        (2, [400, 800, 1200, 1600, 2000], POST),
    ],
)
def test_values_map_to_rows(round_, values, date):
    assert [board_row(round_, v, date) for v in values] == [1, 2, 3, 4, 5]


def test_doubling_date_boundary():
    assert board_row(1, 100, dt.date(2001, 11, 23)) == 1
    assert board_row(1, 100, ON) is None
    assert board_row(1, 200, ON) == 1


def test_same_value_different_rows_across_eras():
    assert board_row(1, 200, PRE) == 2
    assert board_row(1, 200, POST) == 1


def test_final_is_null():
    assert board_row(3, 0, POST) is None
    assert board_row(3, 1000, POST) is None


@pytest.mark.parametrize("value", [None, 0, 300, 700, 2500, -200])
def test_off_grid_or_missing_is_null(value):
    assert board_row(1, value, POST) is None


def test_missing_date_is_null():
    assert board_row(1, 200, None) is None


@given(
    st.sampled_from(sorted(BOARD_VALUES)),
    st.integers(0, 4),
    st.dates(dt.date(1984, 9, 10), dt.date(2030, 1, 1)),
)
def test_property_valid_values_in_range(key, idx, date):
    round_, post = key
    if post != (date >= VALUE_DOUBLING_DATE):
        date = POST if post else PRE
    value = BOARD_VALUES[key][idx]
    assert board_row(round_, value, date) == idx + 1


def test_season():
    assert season_for(dt.date(1984, 9, 10)) == 1
    assert season_for(dt.date(1985, 6, 1)) == 1
    assert season_for(dt.date(2024, 9, 9)) == 41
