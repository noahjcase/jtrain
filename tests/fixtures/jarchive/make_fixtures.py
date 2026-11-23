"""Generate hand-built J! Archive-style fixture pages.

These mimic J! Archive's markup (game pages in both the current `_r` response-cell layout
and the older `onmouseover` layout, plus a season page). They were built by hand rather
than saved from the live site; once you have scraped real pages (cached under
data/html_cache), copy two or three of them here to test against the real markup.

Run: uv run python tests/fixtures/jarchive/make_fixtures.py
"""

from __future__ import annotations

import html
from pathlib import Path

HERE = Path(__file__).parent


def outcome_table(right: list[str], wrong: list[str], stumper: bool) -> str:
    cells = "".join(f'<td class="wrong">{w}</td>' for w in wrong)
    if stumper:
        cells += '<td class="wrong">Triple Stumper</td>'
    cells += "".join(f'<td class="right">{r}</td>' for r in right)
    return f'<table width="100%"><tr>{cells}</tr></table>'


def clue_outcome(col: int, row: int, rnd: str) -> tuple[list[str], list[str], bool]:
    """Deterministic variety: some right, some wrong-then-right, some triple stumpers."""
    k = (col * 7 + row * 3 + (5 if rnd == "DJ" else 0)) % 6
    if k == 0:
        return [], ["Amy", "Ben"], True  # two wrong, nobody right
    if k == 1:
        return [], [], True  # nobody buzzed
    if k == 2:
        return ["Cara"], ["Amy"], False  # one wrong, then right
    return [["Amy", "Ben", "Cara"][k % 3]], [], False


def round_html(
    rnd: str,
    values: list[int],
    dd_cells: dict[tuple[int, int], int],
    unrevealed: set[tuple[int, int]],
    old_layout: bool,
) -> str:
    div_id = "jeopardy_round" if rnd == "J" else "double_jeopardy_round"
    title = "Jeopardy! Round" if rnd == "J" else "Double Jeopardy! Round"
    cats = "".join(
        f'<td class="category"><table><tr><td class="category_name">{rnd} CATEGORY {c}</td>'
        f'</tr><tr><td class="category_comments">{"(Ken: A comment.)" if c == 2 else ""}'
        "</td></tr></table></td>"
        for c in range(1, 7)
    )
    rows = []
    order = 0
    for row in range(1, 6):
        tds = []
        for col in range(1, 7):
            if (col, row) in unrevealed:
                tds.append('<td class="clue"></td>')
                continue
            order += 1
            dom = f"clue_{rnd}_{col}_{row}"
            if (col, row) in dd_cells:
                value_td = f'<td class="clue_value_daily_double">DD: ${dd_cells[(col, row)]:,}</td>'
            else:
                value_td = f'<td class="clue_value">${values[row - 1]:,}</td>'
            right, wrong, stumper = clue_outcome(col, row, rnd)
            resp = (
                f'<em class="correct_response">{rnd} response {col}-{row}</em><br /><br />'
                + outcome_table(right, wrong, stumper)
            )
            text = f"{rnd} clue {col}-{row} about &quot;things&quot;"
            if old_layout:
                attr = html.escape(f"toggle('{dom}', '{dom}_stuck', '{resp}')", quote=True)
                header = (
                    f'<div onmouseover="{attr}"><table class="clue_header"><tr>'
                    f'{value_td}<td class="clue_order_number"><a>{order}</a></td>'
                    "</tr></table></div>"
                )
                body = f'<tr><td id="{dom}" class="clue_text">{text}</td></tr>'
            else:
                header = (
                    f'<table class="clue_header"><tr>{value_td}'
                    f'<td class="clue_order_number"><a>{order}</a></td></tr></table>'
                )
                body = (
                    f'<tr><td id="{dom}" class="clue_text">{text}</td>'
                    f'<td id="{dom}_r" class="clue_text" style="display:none;">'
                    f"{resp}</td></tr>"
                )
            tds.append(f'<td class="clue"><table><tr><td>{header}</td></tr>{body}</table></td>')
        rows.append("<tr>" + "".join(tds) + "</tr>")
    return (
        f'<div id="{div_id}"><h2>{title}</h2><table class="round"><tr>{cats}</tr>'
        + "".join(rows)
        + "</table></div>"
    )


def final_table(
    dom: str, category: str, clue: str, response: str, right: list[str], wrong: list[str]
) -> str:
    resp = f'<em class="correct_response">{response}</em>' + outcome_table(right, wrong, not right)
    return (
        '<table class="final_round"><tr><td class="category"><table><tr>'
        f'<td class="category_name">{category}</td></tr></table></td></tr>'
        f'<tr><td id="{dom}" class="clue_text">{clue}</td></tr>'
        f'<tr><td id="{dom}_r" class="clue_text" style="display:none;">{resp}</td></tr></table>'
    )


def game_page(
    show: int,
    air_date: str,
    values_j: list[int],
    values_dj: list[int],
    *,
    unrevealed: set[tuple[str, int, int]] = frozenset(),
    old_layout: bool = False,
    tiebreaker: bool = False,
) -> str:
    j = round_html(
        "J",
        values_j,
        {(3, 4): 1500},
        {(c, r) for rnd, c, r in unrevealed if rnd == "J"},
        old_layout,
    )
    dj = round_html(
        "DJ",
        values_dj,
        {(2, 3): 3000, (5, 5): 2200},
        {(c, r) for rnd, c, r in unrevealed if rnd == "DJ"},
        old_layout,
    )
    final = final_table(
        "clue_FJ",
        "FINAL CATEGORY",
        "The final clue",
        "the final response",
        ["Amy"],
        ["Ben", "Cara"],
    )
    if tiebreaker:
        final += final_table(
            "clue_TB", "TIEBREAKER", "The tiebreaker clue", "the tiebreaker response", ["Cara"], []
        )
    return (
        f"<html><head><title>J! Archive - Show #{show}, aired {air_date}</title></head><body>"
        f'<div id="game_title"><h1>Show #{show} - aired {air_date}</h1></div>'
        f'{j}{dj}<div id="final_jeopardy_round"><h2>Final Jeopardy! Round</h2>{final}</div>'
        "</body></html>"
    )


def main() -> None:
    post_j, post_dj = [200, 400, 600, 800, 1000], [400, 800, 1200, 1600, 2000]
    (HERE / "game_complete.html").write_text(game_page(9201, "2024-09-09", post_j, post_dj))
    (HERE / "game_unrevealed_old_layout.html").write_text(
        game_page(
            4250,
            "2003-03-03",
            post_j,
            post_dj,
            old_layout=True,
            unrevealed={("J", 6, 5), ("DJ", 1, 5), ("DJ", 6, 5)},
        )
    )
    (HERE / "game_tiebreaker.html").write_text(
        game_page(9202, "2024-09-10", post_j, post_dj, tiebreaker=True)
    )
    season = "".join(
        f'<tr><td><a href="showgame.php?game_id={gid}">#{show}, aired&#160;{date}</a></td></tr>'
        for gid, show, date in [
            (8915, 9201, "2024-09-09"),
            (8916, 9202, "2024-09-10"),
            (8917, 9203, "2024-09-11"),
        ]
    )
    (HERE / "season_41.html").write_text(
        f"<html><head><title>J! Archive - Season 41</title></head><body><table>{season}"
        '</table><a href="showseason.php?season=40">previous season</a></body></html>'
    )
    (HERE / "robots.txt").write_text(
        "User-agent: *\nCrawl-delay: 5\nDisallow: /private/\n\nUser-agent: BadBot\nDisallow: /\n"
    )


if __name__ == "__main__":
    main()
