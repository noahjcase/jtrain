"""J! Archive parser (TESTING §12), against hand-built fixture pages."""

import datetime as dt

from conftest import FIXTURES

from jtrainer.ingest.jarchive import parse_game, parse_season

J = FIXTURES / "jarchive"


def game(name):
    return parse_game((J / name).read_text(), game_id=1)


def test_complete_game_has_61_clues():
    g = game("game_complete.html")
    assert len(g.regular_clues) == 61
    assert sum(c.round == 1 for c in g.clues) == 30
    assert sum(c.round == 2 for c in g.clues) == 30
    assert sum(c.round == 3 for c in g.clues) == 1
    assert g.air_date == dt.date(2024, 9, 9) and g.show_number == 9201


def test_unrevealed_clues_old_layout():
    g = game("game_unrevealed_old_layout.html")
    assert len(g.regular_clues) == 58
    positions = {(c.round, c.col, c.row) for c in g.clues if c.round != 3}
    assert (1, 6, 5) not in positions and (2, 1, 5) not in positions


def test_fields_extracted():
    g = game("game_complete.html")
    by_pos = {(c.round, c.col, c.row): c for c in g.clues if c.round != 3}
    c = by_pos[(1, 2, 1)]
    assert c.value == 200 and c.category == "J CATEGORY 2"
    assert c.category_comment == "(Ken: A comment.)"
    assert c.clue_text == 'J clue 2-1 about "things"'
    assert c.response == "J response 2-1"
    dd = by_pos[(1, 3, 4)]
    assert dd.is_daily_double and dd.dd_wager == 1500 and dd.value == 800
    assert {(c.round, c.col, c.row) for c in g.clues if c.is_daily_double} == {
        (1, 3, 4),
        (2, 2, 3),
        (2, 5, 5),
    }
    assert by_pos[(2, 5, 5)].value == 2000


def test_contestant_outcomes_and_triple_stumpers():
    g = game("game_complete.html")
    by_pos = {(c.round, c.col, c.row): c for c in g.clues if c.round != 3}
    # outcome pattern from the fixture generator: k = (7c + 3r + offset) % 6
    two_wrong = by_pos[(1, 6, 4)]  # k=0: two wrong, triple stumper
    assert (two_wrong.n_correct, two_wrong.n_incorrect, two_wrong.triple_stumper) == (0, 2, True)
    silent = by_pos[(1, 1, 2)]  # k=1: nobody buzzed
    assert (silent.n_correct, silent.n_incorrect, silent.triple_stumper) == (0, 0, True)
    wrong_then_right = by_pos[(1, 1, 3)]  # k=4? compute below
    k = (7 * 1 + 3 * 3) % 6
    assert k == 4 and wrong_then_right.n_correct == 1 and not wrong_then_right.triple_stumper
    final = next(c for c in g.clues if c.round == 3)
    assert (final.n_correct, final.n_incorrect, final.triple_stumper) == (1, 2, False)
    assert final.response == "the final response"


def test_tiebreaker():
    g = game("game_tiebreaker.html")
    tb = [c for c in g.clues if c.is_tiebreaker]
    assert len(tb) == 1 and tb[0].response == "the tiebreaker response"
    assert len(g.regular_clues) == 61


def test_season_page():
    games = parse_season((J / "season_41.html").read_text())
    assert [(s.game_id, s.show_number, s.air_date) for s in games] == [
        (8915, 9201, dt.date(2024, 9, 9)),
        (8916, 9202, dt.date(2024, 9, 10)),
        (8917, 9203, dt.date(2024, 9, 11)),
    ]
