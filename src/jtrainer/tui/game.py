"""Game screen (milestone 6): play a full archived game or a synthesized board."""

from __future__ import annotations

from rich.markup import escape
from rich.table import Table
from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.screen import Screen
from textual.widgets import Footer, Input, Static

from jtrainer.game import (
    Cell,
    GameRecorder,
    GameSpec,
    GameState,
    WagerError,
    archived_game,
    list_archived_games,
    synthetic_game,
)
from jtrainer.service import StudyService
from jtrainer.tui.quiz import render_clue, render_verdict
from jtrainer.tui.widgets import AnswerInput


class GameScreen(Screen):
    BINDINGS = [
        Binding("escape", "back", "Back / pass"),
        Binding("ctrl+g", "end_game", "End game"),
    ]
    DEFAULT_CSS = """
    GameScreen { padding: 0 1; }
    #score { height: 1; }
    #main { height: auto; min-height: 14; }
    #entry { margin-top: 1; }
    #prompt { height: auto; margin-top: 1; }
    """

    def __init__(self, service: StudyService, spec: GameSpec | None = None):
        super().__init__()
        self.service = service
        self.spec = spec
        self.state: GameState | None = None
        self.game_id: int | None = None
        self.recorder = GameRecorder(service.conn, service.clock)
        self.phase = "choose"  # choose|board|wager|answering|confidence|verdict|review|done
        self.cursor = (0, 1)
        self.cell: Cell | None = None
        self.wager: int | None = None
        self.presented = None
        self.outcome = None
        self.pending: str | None = None
        self.shown_ms = 0
        self.first_key_ms: int | None = None
        self.submit_ms: int | None = None

    def compose(self) -> ComposeResult:
        yield Static("", id="score")
        yield Static("", id="main")
        yield AnswerInput(id="entry")
        yield Static("", id="prompt")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#entry", Input).display = False
        if self.spec is not None:
            self.begin(self.spec)
        else:
            self._main(
                "[b]Play a game[/b]\n\n  [b]a[/b]  a random archived game, as aired\n"
                "  [b]s[/b]  a synthesized board of current-era categories"
            )
            self._prompt("")

    # -- small helpers -----------------------------------------------------------

    def _main(self, renderable) -> None:
        self.query_one("#main", Static).update(renderable)

    def _prompt(self, text: str) -> None:
        self.query_one("#prompt", Static).update(text)

    def _score(self) -> None:
        if self.state is None:
            return
        names = {1: "Jeopardy!", 2: "Double Jeopardy!", 3: "Final Jeopardy!", 4: "Game over"}
        s = self.state
        self.query_one("#score", Static).update(
            f"[b]{names[s.round]}[/b] · score [b]${s.score:,}[/b] · Coryat ${s.coryat:,} · "
            f"{escape(s.spec.source)}"
        )

    def _input(self, placeholder: str) -> None:
        inp = self.query_one("#entry", Input)
        inp.display = True
        inp.disabled = False
        inp.value = ""
        inp.placeholder = placeholder
        inp.focus()

    def _hide_input(self) -> None:
        inp = self.query_one("#entry", Input)
        inp.disabled = True
        inp.display = False
        self.set_focus(None)

    # -- game lifecycle ------------------------------------------------------------

    def begin(self, spec: GameSpec) -> None:
        self.spec = spec
        session_id = self.service.start_session("game")
        self.game_id = self.recorder.start(session_id, spec)
        self.state = GameState(spec)
        self.state.start()
        self.show_board()

    def choose(self, kind: str) -> None:
        rng = self.service.rng
        if kind == "a":
            games = list_archived_games(self.service.conn, self.service.config)
            if not games:
                self._prompt("[red]No archived games found. Run `jtrain ingest` first.[/red]")
                return
            air_date, _ = games[int(rng.integers(len(games)))]
            self.begin(archived_game(self.service.conn, air_date, 1.0 / len(games)))
        else:
            self.begin(synthetic_game(self.service.conn, self.service.config, rng))

    def show_board(self) -> None:
        s = self.state
        assert s is not None
        self._score()
        if s.finished:
            self.finish()
            return
        if s.round == 3:
            assert s.spec.final is not None
            self.cell = s.spec.final
            self.ask_wager()
            return
        self.phase = "board"
        board = s.board
        assert board is not None
        if (self.cursor[0], self.cursor[1]) not in board.cells or board.cells[
            self.cursor
        ].clue_id in s.played:
            remaining = sorted(board.remaining(s.played), key=lambda c: (c.col, c.row))
            if remaining:
                self.cursor = (remaining[0].col, remaining[0].row)
        table = Table(show_header=True, expand=True, show_lines=True, padding=(0, 0))
        for name in board.categories:
            table.add_column(Text(name, justify="center", style="bold"), justify="center", ratio=1)
        for row in range(1, 6):
            cells = []
            for col in range(len(board.categories)):
                cell = board.cells.get((col, row))
                if cell is None or cell.clue_id in s.played:
                    cells.append(Text(""))
                    continue
                style = "reverse bold" if (col, row) == self.cursor else "bold yellow"
                cells.append(Text(f"${cell.value:,}", style=style))
            table.add_row(*cells)
        self._main(table)
        self._hide_input()
        self._prompt("[dim]Arrows move · Enter selects · Ctrl+G ends the game[/dim]")

    def select(self) -> None:
        s = self.state
        assert s is not None and s.board is not None
        cell = s.board.cells.get(self.cursor)
        if cell is None or cell.clue_id in s.played:
            return
        self.cell = cell
        if cell.is_daily_double:
            self.ask_wager()
        else:
            self.wager = None
            self.ask_clue()

    def ask_wager(self) -> None:
        assert self.state is not None and self.cell is not None
        lo, hi = self.state.wager_limits(self.cell)
        self.phase = "wager"
        title = "FINAL JEOPARDY!" if self.cell.round == 3 else "DAILY DOUBLE!"
        self._main(
            f"[b yellow]{title}[/b yellow]\n\nCategory: [b]"
            f"{escape(self.cell.category)}[/b]\n\nYour score: ${self.state.score:,}"
        )
        self._input(f"Wager (${lo:,} to ${hi:,})")
        self._prompt("")

    def ask_clue(self) -> None:
        assert self.cell is not None and self.state is not None
        mode = "game:synthetic" if self.spec and self.spec.source == "synthetic" else "game:archive"
        prob = self.spec.choice_prob if self.spec else None
        self.presented = self.service.present(self.cell.clue_id, mode, prob)
        clue = dict(self.presented.clue)
        clue["clue_value"] = self.cell.value
        self._main(render_clue(clue))
        self.phase = "answering"
        self.first_key_ms = None
        self.submit_ms = None
        self.shown_ms = self.service.clock.monotonic_ms()
        self._input("Your response (Enter to submit, blank or Escape to pass)")
        self._prompt("")

    # -- events --------------------------------------------------------------------

    def on_input_changed(self, event: Input.Changed) -> None:
        if self.phase == "answering" and event.value and self.first_key_ms is None:
            self.first_key_ms = self.service.clock.monotonic_ms() - self.shown_ms

    def submit_text(self, value: str) -> None:
        text = value.strip()
        if self.phase == "wager":
            assert self.state is not None and self.cell is not None
            try:
                wager = int(text.replace("$", "").replace(",", ""))
                self.wager = self.state.check_wager(self.cell, wager)
            except (ValueError, WagerError) as exc:
                msg = str(exc) if isinstance(exc, WagerError) else "Enter a whole number."
                self._prompt(f"[red]{escape(msg)}[/red]")
                return
            self.ask_clue()
        elif self.phase == "answering":
            self.submit_ms = self.service.clock.monotonic_ms() - self.shown_ms
            if self.first_key_ms is None:
                self.first_key_ms = self.submit_ms
            if not text or text.lower() == "pass":
                self.grade(None, None, passed=True)
                return
            self.pending = text
            self.phase = "confidence"
            self._hide_input()
            self._prompt("[b]How sure are you?[/b]  1 guess · 2 maybe · 3 sure")

    def on_key(self, event) -> None:
        if self.intercept_key(event.key):
            event.stop()

    def intercept_key(self, key: str) -> bool:
        """Handle a key for the current phase; True if consumed (see AnswerInput)."""
        if self.phase == "choose" and key in ("a", "s"):
            self.choose(key)
            return True
        if self.phase == "board":
            moves = {"left": (-1, 0), "right": (1, 0), "up": (0, -1), "down": (0, 1)}
            if key in moves:
                dx, dy = moves[key]
                col = min(5, max(0, self.cursor[0] + dx))
                row = min(5, max(1, self.cursor[1] + dy))
                self.cursor = (col, row)
                self.show_board()
                return True
            if key == "enter":
                self.select()
                return True
        if self.phase == "confidence" and key in ("1", "2", "3"):
            self.grade(self.pending, int(key))
            return True
        if self.phase == "verdict":
            if key in ("enter", "space"):
                self.show_board()
                return True
            if key == "o":
                self.override()
                return True
        if self.phase == "review" and key in ("y", "n"):
            self.service.resolve(self.outcome.attempt_id, key == "y")
            self.outcome.verdict.correct = key == "y"
            self.outcome.verdict.needs_review = False
            self.outcome.verdict.grader = "user"
            self.apply_outcome()
            return True
        if self.phase == "done" and key in ("enter", "escape"):
            self.app.go_home()
            return True
        return False

    def grade(self, response: str | None, confidence: int | None, passed: bool = False) -> None:
        self.outcome = self.service.submit(
            self.presented,
            response,
            confidence,
            self.first_key_ms,
            self.submit_ms,
            passed=passed,
            wager=self.wager,
        )
        self.pending = response
        self._hide_input()
        if self.outcome.verdict.needs_review:
            self.phase = "review"
            self._main(render_verdict(self.outcome, response))
            self._prompt("[b]Were you right?[/b]  y yes · n no")
            return
        self.apply_outcome()

    def _outcome_name(self) -> str:
        v = self.outcome.verdict
        if v.detail.get("reason") == "pass":
            return "pass"
        return "right" if v.correct else "wrong"

    def apply_outcome(self) -> None:
        assert self.state is not None and self.cell is not None
        delta = self.state.apply(self.cell, self._outcome_name(), self.wager)
        self.recorder.record(self.game_id, self.outcome.attempt_id, self.state.log[-1])
        self.show_verdict(delta)

    def show_verdict(self, delta: int) -> None:
        self.phase = "verdict"
        sign = "+" if delta >= 0 else "-"
        self._main(render_verdict(self.outcome, self.pending) + f"\n\n[b]{sign}${abs(delta):,}[/b]")
        self._score()
        self._prompt("[dim]Enter continue · o override verdict[/dim]")

    def override(self) -> None:
        if self.outcome is None or self._outcome_name() == "pass":
            return
        new = self.service.override(self.outcome.attempt_id)
        self.outcome.verdict.correct = new
        self.outcome.verdict.grader = "user"
        delta = self.state.revise_last("right" if new else "wrong")
        self.recorder.revise(self.game_id, self.outcome.attempt_id, self.state.log[-1])
        self.show_verdict(delta)

    def finish(self) -> None:
        assert self.state is not None
        self.state.finished = True
        self.recorder.finish(self.game_id, self.state)
        self.service.end_session()
        self.phase = "done"
        self._hide_input()
        s = self.state
        right = sum(1 for e in s.log if e["outcome"] == "right")
        wrong = sum(1 for e in s.log if e["outcome"] == "wrong")
        self._main(
            f"[b]Game over[/b]\n\nFinal score: [b]${s.score:,}[/b]\n"
            f"Coryat score: [b]${s.coryat:,}[/b]\n\n{right} right · {wrong} wrong · "
            f"{len(s.log) - right - wrong} passed"
        )
        self._score()
        self._prompt("[dim]Enter to return home[/dim]")

    # -- actions ---------------------------------------------------------------------

    def action_back(self) -> None:
        if self.phase == "answering":
            self.submit_ms = self.service.clock.monotonic_ms() - self.shown_ms
            self.grade(None, None, passed=True)
        elif self.phase in ("choose", "done"):
            self.app.go_home()

    def action_end_game(self) -> None:
        if self.state is not None and self.phase not in ("done",):
            self.finish()
        elif self.state is None:
            self.app.go_home()
