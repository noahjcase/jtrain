"""Stats screen: overall and per-topic accuracy (naive and IPW-corrected), ability with
uncertainty, priority ranking, response-time distribution, confidence calibration."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.screen import Screen
from textual.widgets import DataTable, Footer, Static

from jtrainer.model.stats import Stats, compute_stats
from jtrainer.service import StudyService


def _pct(x: float | None) -> str:
    return "–" if x is None else f"{x:.0%}"


def summary_text(st: Stats) -> str:
    lines = [
        "[b]Overall[/b]",
        f"  attempts {st.overall.n} · accuracy {_pct(st.overall.naive)} "
        f"(IPW {_pct(st.overall.ipw)}) · ability {st.overall_mu:+.2f} ± {st.overall_sd:.2f}",
    ]
    if st.by_round:
        names = {1: "J!", 2: "DJ!", 3: "FJ!"}
        parts = [f"{names[r]} {_pct(a.naive)} (n={a.n})" for r, a in st.by_round.items()]
        lines.append("  by round: " + " · ".join(parts))
    if st.pending_review:
        lines.append(f"  [yellow]{st.pending_review} attempts await your decision[/yellow]")
    if st.untagged_attempts:
        lines.append(
            f"  [dim]{st.untagged_attempts} attempts on untagged clues count toward "
            "overall figures only[/dim]"
        )
    lines.append("")
    lines.append("[b]Response time[/b] (submit)")
    peak = max((c for _, c in st.time_hist), default=0) or 1
    for label, count in st.time_hist:
        bar = "█" * round(24 * count / peak)
        lines.append(f"  {label:>8} {bar} {count}")
    lines.append("")
    lines.append("[b]Confidence calibration[/b]")
    for b in st.calibration:
        lines.append(
            f"  {b.confidence} (implies {b.implied:.0%}): actual {_pct(b.accuracy)} (n={b.n})"
        )
    brier = "–" if st.brier is None else f"{st.brier:.3f}"
    lines.append(f"  Brier score {brier}  [dim](lower is better; 0.25 = coin flip)[/dim]")
    return "\n".join(lines)


class StatsScreen(Screen):
    BINDINGS = [Binding("escape", "app.go_home", "Back"), Binding("q", "app.go_home", "Back")]
    DEFAULT_CSS = """
    StatsScreen { padding: 1 2; }
    #summary { height: auto; }
    #topics { height: auto; max-height: 24; margin-top: 1; }
    #topics-title { margin-top: 1; height: 1; }
    """

    def __init__(self, service: StudyService):
        super().__init__()
        self.service = service
        self.stats: Stats | None = None

    def compose(self) -> ComposeResult:
        with VerticalScroll():
            yield Static("", id="summary")
            yield Static("[b]Topics[/b] by study priority", id="topics-title")
            yield DataTable(id="topics", zebra_stripes=True, cursor_type="row")
        yield Footer()

    def on_mount(self) -> None:
        st = compute_stats(self.service.conn, self.service.config, self.service.ability)
        self.stats = st
        self.query_one("#summary", Static).update(summary_text(st))
        table = self.query_one("#topics", DataTable)
        table.add_columns("#", "topic", "n", "acc", "IPW acc", "ability", "freq", "p̂", "priority")
        if not st.topics:
            self.query_one("#topics-title", Static).update(
                "[b]Topics[/b] [dim]- no taxonomy loaded yet (see `jtrain taxonomy load`)[/dim]"
            )
        for t in st.topics:
            table.add_row(
                str(t.rank),
                t.label,
                str(t.accuracy.n),
                _pct(t.accuracy.naive),
                _pct(t.accuracy.ipw),
                f"{t.mu:+.2f}±{t.sd:.2f}",
                f"{t.frequency:.1%}",
                f"{t.p_hat:.0%}",
                f"{t.priority:.3f}",
            )
