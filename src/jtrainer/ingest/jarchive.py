"""Offline parser for J! Archive game and season pages (milestone 5).

Pages are fetched by `scrape.PoliteClient`, cached to disk, and parsed here offline.
Handles both page layouts: the response in a separate `#clue_X_r` cell (current), and the
older layout with the response inside an `onmouseover` attribute.
"""

from __future__ import annotations

import datetime as dt
import html
import re
from dataclasses import dataclass, field

from selectolax.lexbor import LexborHTMLParser as HTMLParser
from selectolax.lexbor import LexborNode as Node

from jtrainer.ingest.board import board_values
from jtrainer.text import clean_text

_ROUND_IDS = {"J": 1, "DJ": 2}
_CLUE_ID_RE = re.compile(r"^clue_(J|DJ)_(\d)_(\d)$")
_AIRED_RE = re.compile(r"aired\s*(\d{4}-\d{2}-\d{2})")
_SHOW_RE = re.compile(r"Show #(\d+)")
_GAME_LINK_RE = re.compile(r"showgame\.php\?game_id=(\d+)")
_MONEY_RE = re.compile(r"\$?([\d,]+)")


@dataclass
class ParsedClue:
    round: int
    col: int | None
    row: int | None
    value: int | None  # board value
    is_daily_double: bool
    dd_wager: int | None
    category: str
    category_comment: str | None
    clue_text: str
    response: str
    n_correct: int
    n_incorrect: int
    triple_stumper: bool
    order_number: int | None = None
    is_tiebreaker: bool = False


@dataclass
class ParsedGame:
    game_id: int | None
    show_number: int | None
    air_date: dt.date | None
    clues: list[ParsedClue] = field(default_factory=list)

    @property
    def regular_clues(self) -> list[ParsedClue]:
        return [c for c in self.clues if not c.is_tiebreaker]


@dataclass
class SeasonGame:
    game_id: int
    show_number: int | None
    air_date: dt.date | None


def _text(node: Node | None) -> str:
    if node is None:
        return ""
    return clean_text(node.text(separator=" ", strip=False))


def _money(s: str) -> int | None:
    m = _MONEY_RE.search(s or "")
    return int(m.group(1).replace(",", "")) if m else None


def _response_fragment(tree: HTMLParser, clue_dom_id: str) -> HTMLParser | None:
    """The HTML holding the correct response and the right/wrong table for one clue."""
    node = tree.css_first(f"#{clue_dom_id}_r")
    if node is not None:
        return HTMLParser(node.html or "")
    # Older layout: <div onmouseover="toggle('clue_J_1_1', 'clue_J_1_1_stuck', '...')">
    for div in tree.css("div[onmouseover]"):
        attr = div.attributes.get("onmouseover") or ""
        if f"'{clue_dom_id}'" in attr:
            parts = attr.split("', '")
            if len(parts) >= 3:
                frag = parts[-1].rsplit("')", 1)[0]
                frag = html.unescape(frag.replace("\\'", "'").replace('\\"', '"'))
                return HTMLParser(frag)
    return None


def _outcomes(frag: HTMLParser) -> tuple[str, int, int, bool]:
    response = _text(frag.css_first("em.correct_response"))
    rights = [_text(n) for n in frag.css("td.right")]
    wrongs = [_text(n) for n in frag.css("td.wrong")]
    stumper = any(w.lower() == "triple stumper" for w in wrongs)
    n_incorrect = sum(1 for w in wrongs if w.lower() != "triple stumper")
    n_correct = len(rights)
    return response, n_correct, n_incorrect, stumper or n_correct == 0


def _categories(round_div: Node) -> list[tuple[str, str | None]]:
    out = []
    for cat in round_div.css("td.category"):
        name = _text(cat.css_first("td.category_name"))
        comment = _text(cat.css_first("td.category_comments")) or None
        out.append((name, comment))
    return out


def parse_game(page: str, game_id: int | None = None) -> ParsedGame:
    tree = HTMLParser(page)
    title = _text(tree.css_first("title"))
    heading = _text(tree.css_first("#game_title"))
    aired = _AIRED_RE.search(title) or _AIRED_RE.search(heading)
    air_date = dt.date.fromisoformat(aired.group(1)) if aired else None
    show = _SHOW_RE.search(title) or _SHOW_RE.search(heading)
    game = ParsedGame(game_id, int(show.group(1)) if show else None, air_date)

    for div_id, code in (("jeopardy_round", "J"), ("double_jeopardy_round", "DJ")):
        div = tree.css_first(f"#{div_id}")
        if div is None:
            continue
        cats = _categories(div)
        round_ = _ROUND_IDS[code]
        for text_node in div.css("td.clue_text"):
            dom_id = text_node.attributes.get("id") or ""
            m = _CLUE_ID_RE.match(dom_id)
            if not m:
                continue  # the hidden "_r" response cells
            col, row = int(m.group(2)), int(m.group(3))
            clue_text = _text(text_node)
            frag = _response_fragment(tree, dom_id)
            if not clue_text or frag is None:
                continue  # unrevealed clue
            response, n_right, n_wrong, stumper = _outcomes(frag)
            clue_td = text_node.parent
            while (
                clue_td is not None
                and "clue" not in (clue_td.attributes.get("class") or "").split()
            ):
                clue_td = clue_td.parent
            dd_node = clue_td.css_first("td.clue_value_daily_double") if clue_td else None
            order = _text(clue_td.css_first("td.clue_order_number")) if clue_td else ""
            value = board_values(round_, air_date)[row - 1] if air_date and 1 <= row <= 5 else None
            if dd_node is None and clue_td is not None and value is None:
                value = _money(_text(clue_td.css_first("td.clue_value")))
            name, comment = cats[col - 1] if 0 < col <= len(cats) else ("", None)
            game.clues.append(
                ParsedClue(
                    round_,
                    col,
                    row,
                    value,
                    dd_node is not None,
                    _money(_text(dd_node)) if dd_node is not None else None,
                    name,
                    comment,
                    clue_text,
                    response,
                    n_right,
                    n_wrong,
                    stumper,
                    int(order) if order.isdigit() else None,
                )
            )

    final_div = tree.css_first("#final_jeopardy_round")
    if final_div is not None:
        tables = final_div.css("table.final_round") or [final_div]
        for table in tables:
            text_node = table.css_first("td#clue_FJ") or table.css_first("td#clue_TB")
            if text_node is None:
                continue
            dom_id = text_node.attributes.get("id") or ""
            frag = _response_fragment(tree, dom_id)
            if frag is None:
                continue
            response, n_right, n_wrong, stumper = _outcomes(frag)
            cats = _categories(table)
            name, comment = cats[0] if cats else ("", None)
            game.clues.append(
                ParsedClue(
                    3,
                    None,
                    None,
                    None,
                    False,
                    None,
                    name,
                    comment,
                    _text(text_node),
                    response,
                    n_right,
                    n_wrong,
                    stumper,
                    None,
                    is_tiebreaker=dom_id == "clue_TB",
                )
            )
    return game


def parse_season(page: str) -> list[SeasonGame]:
    tree = HTMLParser(page)
    out = []
    for a in tree.css("a"):
        href = a.attributes.get("href") or ""
        m = _GAME_LINK_RE.search(href)
        if not m:
            continue
        text = _text(a)
        aired = _AIRED_RE.search(text)
        show = re.search(r"#(\d+)", text)
        out.append(
            SeasonGame(
                int(m.group(1)),
                int(show.group(1)) if show else None,
                dt.date.fromisoformat(aired.group(1)) if aired else None,
            )
        )
    return out
