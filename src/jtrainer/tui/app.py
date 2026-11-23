"""Textual application: home menu and screen routing."""

from __future__ import annotations

import duckdb
import numpy as np
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.screen import Screen
from textual.widgets import Footer, Static

from jtrainer.clock import Clock, SystemClock
from jtrainer.config import Config
from jtrainer.llm import LLMClient
from jtrainer.service import StudyService

MENU = [
    ("r", "random", "Random quiz", "uniform over eligible clues (unbiased baseline)"),
    ("a", "adaptive", "Adaptive study", "weighted toward topics where improvement pays"),
    ("v", "review", "Spaced review", "missed facts, through different clues"),
    ("g", "game", "Play a game", "a full board with wagering and Coryat score"),
    ("s", "stats", "Stats", "accuracy, ability, priorities, calibration"),
    ("t", "tags", "Tag review", "hand-check topic tags (milestone 2)"),
]


class HomeScreen(Screen):
    BINDINGS = [Binding(k, f"open('{target}')", label, show=False) for k, target, label, _ in MENU]
    BINDINGS += [Binding("q", "app.quit", "Quit")]
    DEFAULT_CSS = """
    HomeScreen { padding: 1 4; }
    #title { text-style: bold; color: $accent; height: 2; }
    #menu { height: auto; }
    #counts { margin-top: 1; color: $text-muted; height: auto; }
    """

    def compose(self) -> ComposeResult:
        yield Static("jtrain · Jeopardy! trainer", id="title")
        lines = []
        for key, _, label, desc in MENU:
            lines.append(f"  [b]{key}[/b]  {label:<16} [dim]{desc}[/dim]")
        lines.append("  [b]q[/b]  Quit")
        yield Static("\n".join(lines), id="menu")
        yield Static("", id="counts")
        yield Footer()

    def on_screen_resume(self) -> None:
        self.refresh_counts()

    def on_mount(self) -> None:
        self.refresh_counts()

    def refresh_counts(self) -> None:
        c = self.app.service.counts()
        text = (
            f"{c['clues']:,} clues · {c['eligible']:,} eligible · {c['tagged']:,} tagged · "
            f"{c['attempts']:,} attempts · {c['due']} reviews due ({c['queued']} queued)"
        )
        self.query_one("#counts", Static).update(text)

    def action_open(self, target: str) -> None:
        self.app.open_screen(target)


class JTrainApp(App):
    TITLE = "jtrain"
    ENABLE_COMMAND_PALETTE = False
    CSS = "Screen { background: $surface; }"

    def __init__(
        self,
        conn: duckdb.DuckDBPyConnection,
        config: Config,
        clock: Clock | None = None,
        rng: np.random.Generator | None = None,
        llm_client: LLMClient | None = None,
    ):
        super().__init__()
        self.conn = conn
        self.config = config
        self.service = StudyService(conn, config, clock or SystemClock(), rng, llm_client)

    def on_mount(self) -> None:
        self.push_screen(HomeScreen())

    def go_home(self) -> None:
        while len(self.screen_stack) > 2 and not isinstance(self.screen, HomeScreen):
            self.pop_screen()
        if not isinstance(self.screen, HomeScreen):
            self.switch_screen(HomeScreen())

    def action_go_home(self) -> None:
        self.go_home()

    def open_screen(self, target: str) -> None:
        from jtrainer.tui.game import GameScreen
        from jtrainer.tui.quiz import QuizScreen
        from jtrainer.tui.stats import StatsScreen
        from jtrainer.tui.tagreview import TagReviewScreen

        if target in ("random", "adaptive", "review"):
            self.push_screen(QuizScreen(self.service, target))
        elif target == "game":
            self.push_screen(GameScreen(self.service))
        elif target == "stats":
            self.push_screen(StatsScreen(self.service))
        elif target == "tags":
            self.push_screen(TagReviewScreen(self.service))

    def on_unmount(self) -> None:
        self.service.end_session()
