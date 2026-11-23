"""Board-row derivation. Board values doubled on 2001-11-26."""

from __future__ import annotations

import datetime as dt

VALUE_DOUBLING_DATE = dt.date(2001, 11, 26)

BOARD_VALUES: dict[tuple[int, bool], tuple[int, ...]] = {
    # (round, post_doubling) -> values for rows 1..5
    (1, False): (100, 200, 300, 400, 500),
    (2, False): (200, 400, 600, 800, 1000),
    (1, True): (200, 400, 600, 800, 1000),
    (2, True): (400, 800, 1200, 1600, 2000),
}


def is_post_doubling(air_date: dt.date) -> bool:
    return air_date >= VALUE_DOUBLING_DATE


def board_values(round_: int, air_date: dt.date) -> tuple[int, ...]:
    return BOARD_VALUES[(round_, is_post_doubling(air_date))]


def board_row(round_: int, clue_value: int | None, air_date: dt.date | None) -> int | None:
    """Row 1-5 for a Jeopardy/Double Jeopardy clue from its *board* value.

    Daily Doubles must pass the board value (`clue_value`), never the wager.
    Final Jeopardy, missing values, missing dates and off-grid values give None.
    """
    if round_ not in (1, 2) or clue_value is None or air_date is None:
        return None
    values = board_values(round_, air_date)
    try:
        return values.index(clue_value) + 1
    except ValueError:
        return None


def season_for(air_date: dt.date) -> int:
    """Approximate season number; seasons start in September (season 1 = 1984-85)."""
    return air_date.year - 1983 if air_date.month >= 8 else air_date.year - 1984
