"""Tag review screen (milestone 2): accept, change topic, or flag the taxonomy."""

from __future__ import annotations

from rich.markup import escape
from textual.app import ComposeResult
from textual.binding import Binding
from textual.screen import ModalScreen, Screen
from textual.widgets import Footer, Input, OptionList, Static
from textual.widgets.option_list import Option

from jtrainer.service import StudyService
from jtrainer.tagging.tagger import TagReview
from jtrainer.tagging.taxonomy import stored_topics


class TopicPicker(ModalScreen[str | None]):
    BINDINGS = [Binding("escape", "dismiss_none", "Cancel")]
    DEFAULT_CSS = "TopicPicker { align: center middle; } #picker { width: 60; height: 20; }"

    def __init__(self, topics: list[tuple[str, str]]):
        super().__init__()
        self.topics = topics

    def compose(self) -> ComposeResult:
        yield OptionList(
            *[Option(f"{label} [dim]({tid})[/dim]", id=tid) for tid, label in self.topics],
            id="picker",
        )

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        self.dismiss(event.option.id)

    def action_dismiss_none(self) -> None:
        self.dismiss(None)


class NotePrompt(ModalScreen[str | None]):
    BINDINGS = [Binding("escape", "dismiss_none", "Cancel")]
    DEFAULT_CSS = "NotePrompt { align: center middle; } #note { width: 70; }"

    def compose(self) -> ComposeResult:
        yield Input(placeholder="What topic is missing? (Enter to save)", id="note")

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self.dismiss(event.value.strip() or None)

    def action_dismiss_none(self) -> None:
        self.dismiss(None)


class TagReviewScreen(Screen):
    BINDINGS = [
        Binding("a", "accept", "Accept"),
        Binding("c", "change", "Change topic"),
        Binding("f", "flag", "Flag missing topic"),
        Binding("s", "skip", "Skip"),
        Binding("escape", "app.go_home", "Back"),
    ]
    DEFAULT_CSS = "TagReviewScreen { padding: 1 2; } #item { height: auto; } #tally { height: 2; }"

    def __init__(self, service: StudyService):
        super().__init__()
        self.service = service
        version = service.config.taxonomy.version
        self.review = TagReview(service.conn, version, service.clock)
        self.topics = stored_topics(service.conn, version)
        self.labels = dict(self.topics)
        self.queue: list[dict] = []
        self.current: dict | None = None
        self.skipped: set[int] = set()

    def compose(self) -> ComposeResult:
        yield Static("", id="tally")
        yield Static("", id="item")
        yield Footer()

    def on_mount(self) -> None:
        self.advance()

    def advance(self) -> None:
        self.queue = [
            q
            for q in self.review.queue(limit=len(self.skipped) + 50)
            if q["clue_id"] not in self.skipped
        ]
        self.current = self.queue[0] if self.queue else None
        s = self.review.summary()
        acc = "–" if s["accuracy"] is None else f"{s['accuracy']:.0%}"
        self.query_one("#tally", Static).update(
            f"Reviewed {s['total']} · accepted {s.get('accept', 0)} · changed "
            f"{s.get('change', 0)} · flagged {s.get('flag', 0)} · accuracy {acc}"
        )
        item = self.query_one("#item", Static)
        if self.current is None:
            item.update(
                "[b]Nothing to review.[/b]\n\nTag a sample first: `jtrain tag sample --n 500`."
            )
            return
        c = self.current
        secondary = (
            f" · secondary: {escape(self.labels.get(c['secondary_topic'], ''))}"
            if c["secondary_topic"]
            else ""
        )
        item.update(
            f"[b]{escape(c['category'] or '')}[/b]\n\n{escape(c['clue_text'])}\n\n"
            f"Response: [b]{escape(c['response'])}[/b]   Entity: {escape(c['entity'] or '–')}\n\n"
            f"Proposed topic: [b]{escape(self.labels.get(c['topic_id'], c['topic_id']))}[/b]"
            f"{secondary}"
        )

    def action_accept(self) -> None:
        if self.current:
            self.review.accept(self.current["clue_id"])
            self.advance()

    def action_change(self) -> None:
        if not self.current:
            return
        clue_id = self.current["clue_id"]

        def done(topic: str | None) -> None:
            if topic:
                self.review.change(clue_id, topic)
                self.advance()

        self.app.push_screen(TopicPicker(self.topics), done)

    def action_flag(self) -> None:
        if not self.current:
            return
        clue_id = self.current["clue_id"]

        def done(note: str | None) -> None:
            if note:
                self.review.flag(clue_id, note)
                self.advance()

        self.app.push_screen(NotePrompt(), done)

    def action_skip(self) -> None:
        if self.current:
            self.skipped.add(self.current["clue_id"])
            self.advance()
