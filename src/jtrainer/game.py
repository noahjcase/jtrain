"""Game simulator (milestone 6): boards, Daily Doubles, Final Jeopardy, wagering, Coryat.

Single player, no buzzer: on each clue the player answers or passes. Passing a regular
clue scores nothing (as not buzzing would).
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

import duckdb
import numpy as np

from jtrainer.clock import Clock, SystemClock
from jtrainer.config import Config
from jtrainer.db import to_db
from jtrainer.ingest.board import board_values
from jtrainer.model.priority import eligibility_sql

MIN_DD_WAGER = 5


class WagerError(ValueError):
    pass


# ---------------------------------------------------------------------------
# Rules (pure)


def top_board_value(round_: int, air_date: dt.date) -> int:
    return board_values(round_, air_date)[-1]


def dd_wager_limits(score: int, round_: int, air_date: dt.date) -> tuple[int, int]:
    """Daily Double: minimum $5; maximum the greater of the score and the round's top value."""
    return MIN_DD_WAGER, max(score, top_board_value(round_, air_date))


def check_dd_wager(wager: int, score: int, round_: int, air_date: dt.date) -> int:
    lo, hi = dd_wager_limits(score, round_, air_date)
    if not lo <= wager <= hi:
        raise WagerError(f"Daily Double wager must be between ${lo:,} and ${hi:,}")
    return wager


def plays_final(score: int) -> bool:
    return score > 0


def check_final_wager(wager: int, score: int) -> int:
    if not plays_final(score):
        raise WagerError("a player at $0 or below does not play Final Jeopardy!")
    if not 0 <= wager <= score:
        raise WagerError(f"Final Jeopardy! wager must be between $0 and ${score:,}")
    return wager


def score_delta(outcome: str, amount: int) -> int:
    if outcome == "right":
        return amount
    if outcome == "wrong":
        return -amount
    if outcome == "pass":
        return 0
    raise ValueError(f"unknown outcome {outcome!r}")


def coryat_delta(
    outcome: str, board_value: int | None, is_daily_double: bool, is_final: bool
) -> int:
    """Coryat: regular clues at board value (+right, -wrong); Daily Doubles at board value
    if right and zero if wrong; Final Jeopardy excluded."""
    if is_final or board_value is None:
        return 0
    if is_daily_double:
        return board_value if outcome == "right" else 0
    return score_delta(outcome, board_value)


# ---------------------------------------------------------------------------
# Boards


@dataclass
class Cell:
    clue_id: int
    round: int
    col: int  # 0-5
    row: int  # 1-5
    value: int
    is_daily_double: bool
    category: str


@dataclass
class Board:
    round: int
    categories: list[str]
    cells: dict[tuple[int, int], Cell] = field(default_factory=dict)  # (col, row) -> Cell

    def remaining(self, played: set[int]) -> list[Cell]:
        return [c for c in self.cells.values() if c.clue_id not in played]


@dataclass
class GameSpec:
    source: str
    air_date: dt.date  # determines value era and DD limits
    boards: dict[int, Board]
    final: Cell | None
    choice_prob: float | None = None  # probability this game was chosen (for logging)


def list_archived_games(
    conn: duckdb.DuckDBPyConnection, config: Config, min_clues: int = 55
) -> list[tuple[dt.date, int]]:
    """Air dates of regular (non-tournament, non-special) games with enough clues."""
    return [
        tuple(r)
        for r in conn.execute(
            """
        SELECT air_date, count(*) FROM clues
        WHERE NOT is_tournament AND NOT is_special
        GROUP BY air_date HAVING count(*) >= ? AND count(*) FILTER (WHERE round = 3) >= 1
        ORDER BY air_date
        """,
            [min_clues],
        ).fetchall()
    ]


def archived_game(
    conn: duckdb.DuckDBPyConnection, air_date: dt.date, choice_prob: float | None = None
) -> GameSpec:
    """Replay an archived game: clues in their aired board positions. Category order is the
    order categories first appear in the source file."""
    rows = conn.execute(
        """
        SELECT clue_id, round, board_row, clue_value, is_daily_double, category, source_line
        FROM clues WHERE air_date = ? AND NOT is_special ORDER BY source_line, clue_id
        """,
        [air_date],
    ).fetchall()
    if not rows:
        raise KeyError(f"no game aired on {air_date}")
    boards: dict[int, Board] = {}
    final = None
    for clue_id, rnd, row, _value, dd, category, _ in rows:
        if rnd == 3:
            final = final or Cell(clue_id, 3, 0, 0, 0, False, category)
            continue
        board = boards.setdefault(rnd, Board(rnd, []))
        if category not in board.categories:
            if len(board.categories) >= 6:
                continue
            board.categories.append(category)
        col = board.categories.index(category)
        if row is None or (col, row) in board.cells:
            continue
        board.cells[(col, row)] = Cell(
            clue_id, rnd, col, row, board_values(rnd, air_date)[row - 1], bool(dd), category
        )
    return GameSpec(f"archive:{air_date}", air_date, boards, final, choice_prob)


def synthetic_game(
    conn: duckdb.DuckDBPyConnection, config: Config, rng: np.random.Generator
) -> GameSpec:
    """A board of six complete, eligible current-era categories per round, Daily Doubles
    placed like the show (one in J!, two in DJ! in different columns, mostly lower rows)."""
    elig = eligibility_sql(config)
    boards: dict[int, Board] = {}
    era_date = dt.date(2001, 11, 26)
    for rnd, n_dd in ((1, 1), (2, 2)):
        cats = conn.execute(
            f"""
            SELECT category, list(clue_id ORDER BY board_row) AS ids
            FROM clues c WHERE round = ? AND air_date >= DATE '2001-11-26' AND {elig}
              AND board_row IS NOT NULL
            GROUP BY air_date, category
            HAVING count(*) = 5 AND count(DISTINCT board_row) = 5
            ORDER BY min(clue_id)
            """,
            [rnd],
        ).fetchall()
        if len(cats) < 6:
            raise RuntimeError("not enough complete categories for a synthetic board")
        chosen = rng.choice(len(cats), size=6, replace=False)
        board = Board(rnd, [])
        dd_cols = set(int(x) for x in rng.choice(6, size=n_dd, replace=False))
        dd_rows = {col: int(rng.choice([2, 3, 4, 5], p=[0.1, 0.3, 0.35, 0.25])) for col in dd_cols}
        for col, ci in enumerate(chosen):
            category, ids = cats[int(ci)]
            board.categories.append(category)
            for row, clue_id in enumerate(ids, start=1):
                board.cells[(col, row)] = Cell(
                    int(clue_id),
                    rnd,
                    col,
                    row,
                    board_values(rnd, era_date)[row - 1],
                    dd_rows.get(col) == row,
                    category,
                )
        boards[rnd] = board
    finals = conn.execute(
        f"SELECT clue_id, category FROM clues c WHERE round = 3 AND {elig} ORDER BY clue_id"
    ).fetchall()
    fid, fcat = finals[int(rng.integers(len(finals)))]
    return GameSpec("synthetic", era_date, boards, Cell(fid, 3, 0, 0, 0, False, fcat), None)


# ---------------------------------------------------------------------------
# Game state


@dataclass
class GameState:
    spec: GameSpec
    score: int = 0
    coryat: int = 0
    round: int = 1
    played: set[int] = field(default_factory=set)
    log: list[dict] = field(default_factory=list)
    finished: bool = False

    @property
    def board(self) -> Board | None:
        return self.spec.boards.get(self.round)

    def wager_limits(self, cell: Cell) -> tuple[int, int]:
        if cell.round == 3:
            return 0, max(self.score, 0)
        return dd_wager_limits(self.score, cell.round, self.spec.air_date)

    def check_wager(self, cell: Cell, wager: int) -> int:
        if cell.round == 3:
            return check_final_wager(wager, self.score)
        return check_dd_wager(wager, self.score, cell.round, self.spec.air_date)

    def apply(self, cell: Cell, outcome: str, wager: int | None = None) -> int:
        is_final = cell.round == 3
        if is_final or cell.is_daily_double:
            if wager is None:
                raise WagerError("this clue needs a wager")
            amount = self.check_wager(cell, wager)
            # On a Daily Double or Final, not answering counts as wrong.
            outcome = "wrong" if outcome == "pass" else outcome
        else:
            amount = cell.value
        delta = score_delta(outcome, amount)
        self.score += delta
        self.coryat += coryat_delta(outcome, cell.value, cell.is_daily_double, is_final)
        self.played.add(cell.clue_id)
        self.log.append(
            {
                "clue_id": cell.clue_id,
                "round": cell.round,
                "col": cell.col,
                "row": cell.row,
                "value": cell.value,
                "outcome": outcome,
                "delta": delta,
                "wager": wager,
            }
        )
        self._advance()
        return delta

    def revise_last(self, outcome: str) -> int:
        """Change the most recent clue's outcome (after a verdict override)."""
        entry = self.log[-1]
        is_final = entry["round"] == 3
        amount = entry["wager"] if entry["wager"] is not None else entry["value"]
        if (is_final or entry["wager"] is not None) and outcome == "pass":
            outcome = "wrong"
        old_coryat = coryat_delta(
            entry["outcome"], entry["value"], entry["wager"] is not None and not is_final, is_final
        )
        new_delta = score_delta(outcome, amount)
        self.score += new_delta - entry["delta"]
        self.coryat += (
            coryat_delta(
                outcome, entry["value"], entry["wager"] is not None and not is_final, is_final
            )
            - old_coryat
        )
        entry.update(outcome=outcome, delta=new_delta)
        if not self.finished:
            self._advance()  # a changed score can change Final Jeopardy! eligibility
        return new_delta

    def _advance(self) -> None:
        while True:
            if self.round in (1, 2):
                board = self.board
                if board is not None and board.remaining(self.played):
                    return
                self.round += 1
                continue
            if self.round == 3:
                final = self.spec.final
                if final is None or final.clue_id in self.played or not plays_final(self.score):
                    self.finished = True
                    self.round = 4
                return
            self.finished = True
            return

    def start(self) -> None:
        self._advance()

    def skip_round(self) -> None:
        """Move on with clues left unplayed (as when time runs out on the show)."""
        if self.round in (1, 2):
            self.round += 1
            self._advance()


class GameRecorder:
    """Persists games and their per-clue outcomes."""

    def __init__(self, conn: duckdb.DuckDBPyConnection, clock: Clock | None = None):
        self.conn = conn
        self.clock = clock or SystemClock()

    def start(self, session_id: int, spec: GameSpec) -> int:
        return self.conn.execute(
            "INSERT INTO games (session_id, source, started_at) VALUES (?, ?, ?) RETURNING game_id",
            [session_id, spec.source, to_db(self.clock.now())],
        ).fetchone()[0]

    def record(self, game_id: int, attempt_id: int, entry: dict) -> None:
        self.conn.execute(
            "INSERT INTO game_clues VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [
                game_id,
                attempt_id,
                entry["round"],
                entry["col"],
                entry["row"],
                entry["value"],
                entry["outcome"],
                entry["delta"],
            ],
        )

    def revise(self, game_id: int, attempt_id: int, entry: dict) -> None:
        self.conn.execute(
            "UPDATE game_clues SET outcome = ?, score_delta = ? WHERE game_id = ? "
            "AND attempt_id = ?",
            [entry["outcome"], entry["delta"], game_id, attempt_id],
        )

    def finish(self, game_id: int, state: GameState) -> None:
        self.conn.execute(
            "UPDATE games SET ended_at = ?, final_score = ?, coryat = ? WHERE game_id = ?",
            [to_db(self.clock.now()), state.score, state.coryat, game_id],
        )

    def history(self, limit: int = 20) -> list[tuple]:
        return self.conn.execute(
            "SELECT game_id, source, started_at, final_score, coryat FROM games "
            "WHERE ended_at IS NOT NULL ORDER BY started_at DESC LIMIT ?",
            [limit],
        ).fetchall()
