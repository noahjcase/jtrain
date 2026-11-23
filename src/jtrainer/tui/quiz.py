"""Quiz screen (random / adaptive / review modes) and the session summary.

Flow per clue: type a response, Enter -> press 1/2/3 for confidence (before the reveal)
-> verdict. Blank Enter or Escape passes. On the verdict: Enter/Space next, o override,
q end session. If the grader could not decide: y (I was right) / n (I was wrong).
"""

from __future__ import annotations

from rich.markup import escape
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import Screen
from textual.widgets import Footer, Input, Static

from jtrainer.service import AttemptOutcome, Presented, StudyService
from jtrainer.tui.widgets import AnswerInput

ROUND_NAMES = {1: "Jeopardy!", 2: "Double Jeopardy!", 3: "Final Jeopardy!"}
CONFIDENCE_NAMES = {1: "guess", 2: "maybe", 3: "sure"}


def clue_header(clue: dict) -> str:
    parts = [ROUND_NAMES.get(clue["round"], "?")]
    if clue["round"] != 3 and clue.get("clue_value"):
        parts.append(f"${clue['clue_value']:,}")
    if clue.get("is_daily_double"):
        parts.append("Daily Double")
    if clue.get("air_date"):
        parts.append(f"aired {clue['air_date']}")
    return " · ".join(parts)


def render_clue(clue: dict) -> str:
    cat = escape(clue.get("category") or "")
    comment = clue.get("category_comment")
    lines = [f"[b]{cat}[/b]"]
    if comment:
        lines.append(f"[i]{escape(comment)}[/i]")
    lines.append(f"[dim]{escape(clue_header(clue))}[/dim]")
    lines.append("")
    lines.append(escape(clue["clue_text"]))
    return "\n".join(lines)


def render_verdict(outcome: AttemptOutcome, user_response: str | None) -> str:
    v = outcome.verdict
    if v.needs_review:
        head = "[b yellow]? Couldn't grade this automatically[/b yellow]"
    elif v.correct:
        head = "[b green]✓ Correct[/b green]"
    else:
        head = "[b red]✗ Incorrect[/b red]"
    lines = [
        head,
        f"Correct response: [b]{escape(outcome.clue['response'])}[/b]",
        f"Your response:    {escape(user_response or '(pass)')}",
    ]
    grader = v.grader + (" (override)" if v.grader == "user" else "")
    extra = ""
    if "score" in v.detail:
        extra = f" · match score {v.detail['score']}"
    if v.detail.get("rationale"):
        extra += f" · {escape(v.detail['rationale'])}"
    lines.append(f"[dim]graded by {grader}{extra}[/dim]")
    return "\n".join(lines)


class QuizScreen(Screen):
    BINDINGS = [
        Binding("escape", "pass_clue", "Pass", show=True),
        Binding("ctrl+g", "end_session", "End session", show=True),
    ]

    DEFAULT_CSS = """
    QuizScreen { padding: 1 2; }
    #status { color: $text-muted; height: 1; }
    #clue { border: round $primary; padding: 1 2; height: auto; min-height: 7; }
    #response { margin-top: 1; }
    #prompt { height: auto; margin-top: 1; }
    #verdict { height: auto; margin-top: 1; }
    #empty { padding: 2 4; }
    """

    def __init__(self, service: StudyService, mode: str):
        super().__init__()
        self.service = service
        self.mode = mode
        self.state = "idle"  # answering | confidence | verdict | review_needed | empty
        self.presented: Presented | None = None
        self.outcome: AttemptOutcome | None = None
        self.shown_ms = 0
        self.first_key_ms: int | None = None
        self.submit_ms: int | None = None
        self.pending_response: str | None = None

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static("", id="status")
            yield Static("", id="clue")
            yield AnswerInput(
                placeholder="Your response (Enter to submit, blank Enter to pass)", id="response"
            )
            yield Static("", id="verdict")
            yield Static("", id="prompt")
        yield Footer()

    def on_mount(self) -> None:
        self.service.start_session(self.mode)
        self.next_clue()

    # -- helpers -------------------------------------------------------------------

    @property
    def clock(self):
        return self.service.clock

    def _elapsed(self) -> int:
        return self.clock.monotonic_ms() - self.shown_ms

    def _status(self) -> str:
        n = self.service.session_attempt_count()
        s = self.service.session_summary()
        acc = f"{s['accuracy']:.0%}" if s["accuracy"] is not None else "–"
        total = self.service.config.sampling.session_length
        progress = f"{n}/{total}" if self.mode in ("random", "adaptive") else f"{n} done"
        return f"{self.mode.title()} · {progress} · accuracy {acc}"

    def _set(self, widget_id: str, text: str) -> None:
        self.query_one(f"#{widget_id}", Static).update(text)

    def next_clue(self) -> None:
        if self.service.session_done():
            self.finish()
            return
        presented = self.service.next_clue()
        inp = self.query_one("#response", Input)
        if presented is None:
            self.state = "empty"
            inp.display = False
            msg = (
                "[b]Nothing due for review.[/b]\n\n"
                f"{self.service.review.queue_size()} entities are in the review queue. "
                "Missed clues come back here when they are due.\n\nPress Escape or "
                "Ctrl+G to go back."
                if self.mode == "review"
                else "[b]No eligible clues.[/b] Run `jtrain ingest` first."
            )
            self._set("clue", msg)
            self._set("prompt", "")
            self._set("verdict", "")
            self._set("status", self._status())
            return
        self.presented = presented
        self.outcome = None
        self.pending_response = None
        self.first_key_ms = None
        self.submit_ms = None
        self.shown_ms = self.clock.monotonic_ms()
        self.state = "answering"
        inp.display = True
        inp.disabled = False
        inp.value = ""
        inp.focus()
        self._set("clue", render_clue(presented.clue))
        self._set("prompt", "[dim]Type your response. Enter submits; Escape passes.[/dim]")
        self._set("verdict", "")
        self._set("status", self._status())

    # -- events ------------------------------------------------------------------

    def on_input_changed(self, event: Input.Changed) -> None:
        if self.state == "answering" and event.value and self.first_key_ms is None:
            self.first_key_ms = self._elapsed()

    def submit_text(self, value: str) -> None:
        if self.state != "answering":
            return
        self.submit_ms = self._elapsed()
        if self.first_key_ms is None:
            self.first_key_ms = self.submit_ms
        text = value.strip()
        if not text or text.lower() == "pass":
            self._grade(None, None, passed=True)
            return
        self.pending_response = text
        self.state = "confidence"
        inp = self.query_one("#response", Input)
        inp.disabled = True
        self.set_focus(None)
        self._set("prompt", "[b]How sure are you?[/b]  1 guess · 2 maybe · 3 sure")

    def on_key(self, event) -> None:
        if self.intercept_key(event.key):
            event.stop()

    def intercept_key(self, key: str) -> bool:
        """Handle a key for the current state; True if consumed. Also called by the answer
        box first, so keys typed right after Enter reach the confidence prompt."""
        if self.state == "confidence" and key in ("1", "2", "3"):
            self._grade(self.pending_response, int(key))
            return True
        if self.state == "verdict":
            if key in ("enter", "space", "n"):
                self.next_clue()
                return True
            if key == "o":
                self.override()
                return True
            if key == "q":
                self.finish()
                return True
        if self.state == "review_needed" and key in ("y", "n"):
            self.resolve(key == "y")
            return True
        return False

    def _grade(self, response: str | None, confidence: int | None, passed: bool = False) -> None:
        assert self.presented is not None
        self.outcome = self.service.submit(
            self.presented, response, confidence, self.first_key_ms, self.submit_ms, passed=passed
        )
        self._show_verdict(response)

    def _show_verdict(self, response: str | None) -> None:
        assert self.outcome is not None
        self.query_one("#response", Input).disabled = True
        self.set_focus(None)
        self._set("verdict", render_verdict(self.outcome, response))
        if self.outcome.verdict.needs_review:
            self.state = "review_needed"
            self._set("prompt", "[b]Were you right?[/b]  y yes · n no")
        else:
            self.state = "verdict"
            self._set("prompt", "[dim]Enter next · o override verdict · q end session[/dim]")
        self._set("status", self._status())

    def override(self) -> None:
        if self.outcome is None:
            return
        new = self.service.override(self.outcome.attempt_id)
        self.outcome.verdict.correct = new
        self.outcome.verdict.grader = "user"
        self._show_verdict(self.pending_response)

    def resolve(self, correct: bool) -> None:
        assert self.outcome is not None
        self.service.resolve(self.outcome.attempt_id, correct)
        self.outcome.verdict.correct = correct
        self.outcome.verdict.needs_review = False
        self.outcome.verdict.grader = "user"
        self._show_verdict(self.pending_response)

    # -- actions -----------------------------------------------------------------

    def action_pass_clue(self) -> None:
        if self.state == "answering":
            self.submit_ms = self._elapsed()
            self._grade(None, None, passed=True)
        elif self.state == "confidence":
            # back out of the confidence prompt: treat as a pass
            self._grade(None, None, passed=True)
        elif self.state == "empty":
            self.finish()

    def action_end_session(self) -> None:
        self.finish()

    def finish(self) -> None:
        summary = self.service.session_summary()
        self.service.end_session()
        self.app.switch_screen(SummaryScreen(summary, self.mode))


class SummaryScreen(Screen):
    BINDINGS = [Binding("enter", "home", "Home"), Binding("escape", "home", "Home")]
    DEFAULT_CSS = "SummaryScreen { padding: 2 4; } #summary { height: auto; }"

    def __init__(self, summary: dict, mode: str):
        super().__init__()
        self.summary = summary
        self.mode = mode

    def compose(self) -> ComposeResult:
        s = self.summary
        acc = f"{s['accuracy']:.0%}" if s["accuracy"] is not None else "–"
        med = f"{s['median_ms'] / 1000:.1f}s" if s["median_ms"] is not None else "–"
        lines = [
            f"[b]Session complete[/b] ({self.mode})",
            "",
            f"Clues attempted:   {s['attempts']}",
            f"Correct:           {s['correct']} of {s['graded']} graded",
            f"Accuracy:          [b]{acc}[/b]",
            f"Passes:            {s['passes']}",
            f"Median time:       {med}",
        ]
        if s["pending_review"]:
            lines.append(f"Awaiting your decision: {s['pending_review']}")
        lines += ["", "[dim]Enter to return home[/dim]"]
        yield Static("\n".join(lines), id="summary")
        yield Footer()

    def action_home(self) -> None:
        self.app.go_home()
