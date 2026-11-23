"""Text cleaning and normalization shared by ingest and grading.

Decisions pinned by tests:
- Accents are folded for *matching* ("Dali" matches "Dalí"); stored text keeps them.
- Apostrophes and periods are deleted ("U.S." -> "us", "what's" -> "whats");
  other punctuation becomes whitespace; "&" becomes "and".
- Leading question prefixes ("what is", "who are", "what's", ...) and leading
  articles are stripped repeatedly until none remain.
"""

from __future__ import annotations

import html
import itertools
import re
import unicodedata

_PREFIX_FIRST = ("what", "who", "where", "when", "whom")
_PREFIX_VERBS = ("is", "are", "was", "were")
_CONTRACTIONS = ("whats", "whos", "wheres", "whens")
_ARTICLES = ("the", "a", "an")
_DELETE = str.maketrans("", "", "'’‘`.ʻ")


def clean_text(raw: str | None) -> str:
    """Clean a stored field: decode HTML entities, unescape quotes, trim. Keeps Unicode."""
    if raw is None:
        return ""
    s = raw.replace('\\"', '"').replace("\\'", "'")
    s = html.unescape(s)
    s = s.replace(" ", " ")
    return re.sub(r"\s+", " ", s).strip()


def fold(s: str) -> str:
    """Case-fold, strip accents, delete apostrophes/periods, punctuation -> spaces."""
    s = unicodedata.normalize("NFKD", s)
    s = s.casefold()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.replace("&", " and ")
    s = s.translate(_DELETE)
    out = []
    for ch in s:
        out.append(ch if ch.isalnum() else " ")
    return " ".join("".join(out).split())


def _strip_leading(tokens: list[str]) -> list[str]:
    changed = True
    while changed and tokens:
        changed = False
        if len(tokens) >= 2 and tokens[0] in _PREFIX_FIRST and tokens[1] in _PREFIX_VERBS:
            tokens = tokens[2:]
            changed = True
        elif (tokens and tokens[0] in _CONTRACTIONS) or (tokens and tokens[0] in _ARTICLES):
            tokens = tokens[1:]
            changed = True
    return tokens


def normalize(s: str | None) -> str:
    """Normalize a response for comparison. Idempotent; never raises."""
    if not s:
        return ""
    tokens = fold(s).split()
    # fold() can expose new combining sequences only once; a second pass is a no-op.
    return " ".join(_strip_leading(tokens))


# ---------------------------------------------------------------------------
# Number words

_ONES = (
    "zero one two three four five six seven eight nine ten eleven twelve thirteen "
    "fourteen fifteen sixteen seventeen eighteen nineteen"
).split()
_TENS = ["_", "_", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
_ORD_IRREGULAR = {
    "one": "first",
    "two": "second",
    "three": "third",
    "five": "fifth",
    "eight": "eighth",
    "nine": "ninth",
    "twelve": "twelfth",
}


def _under_1000(n: int) -> list[str]:
    words: list[str] = []
    if n >= 100:
        words += [_ONES[n // 100], "hundred"]
        n %= 100
    if n >= 20:
        words.append(_TENS[n // 10])
        if n % 10:
            words.append(_ONES[n % 10])
    elif n > 0 or not words:
        words.append(_ONES[n])
    return words


def cardinal_words(n: int) -> str:
    if n == 0:
        return "zero"
    parts: list[str] = []
    for scale, name in ((10**9, "billion"), (10**6, "million"), (1000, "thousand"), (1, "")):
        if n >= scale:
            chunk, n = divmod(n, scale)
            parts += _under_1000(chunk)
            if name:
                parts.append(name)
    return " ".join(parts)


def year_words(n: int) -> str | None:
    """'1984' -> 'nineteen eighty four'; None if not a year-style number."""
    if not 1100 <= n <= 2099 or (n % 1000 < 10 and n >= 2000):
        return None
    hi, lo = divmod(n, 100)
    if lo == 0:
        return f"{cardinal_words(hi)} hundred"
    lo_words = cardinal_words(lo) if lo >= 10 else f"oh {cardinal_words(lo)}"
    return f"{cardinal_words(hi)} {lo_words}"


def ordinal_words(n: int) -> str:
    words = cardinal_words(n).split()
    last = words[-1]
    if last in _ORD_IRREGULAR:
        words[-1] = _ORD_IRREGULAR[last]
    elif last.endswith("y"):
        words[-1] = last[:-1] + "ieth"
    else:
        words[-1] = last + "th"
    return " ".join(words)


_ORD_RE = re.compile(r"^(\d+)(st|nd|rd|th)$")


def _token_alternatives(tok: str) -> list[str]:
    if tok.isdigit() and len(tok) <= 10:
        n = int(tok)
        alts = [tok, cardinal_words(n)]
        yw = year_words(n) if len(tok) == 4 else None
        if yw:
            alts.append(yw)
        return alts
    m = _ORD_RE.match(tok)
    if m and len(m.group(1)) <= 6:
        return [tok, ordinal_words(int(m.group(1)))]
    return [tok]


IRREGULAR_SINGULAR = {
    "mice": "mouse",
    "lice": "louse",
    "geese": "goose",
    "men": "man",
    "women": "woman",
    "children": "child",
    "teeth": "tooth",
    "feet": "foot",
    "people": "person",
    "oxen": "ox",
    "cacti": "cactus",
    "fungi": "fungus",
    "nuclei": "nucleus",
    "criteria": "criterion",
    "phenomena": "phenomenon",
    "alumni": "alumnus",
    "larvae": "larva",
    "dice": "die",
}


def number_variants(normalized: str, limit: int = 8) -> set[str]:
    """Spell out digit tokens so '1984' and 'nineteen eighty four' can meet, and fold
    irregular plurals ('mice' -> 'mouse') so both sides meet on the singular."""
    tokens = [IRREGULAR_SINGULAR.get(t, t) for t in normalized.split()]
    options = [_token_alternatives(t) for t in tokens]
    out: set[str] = set()
    for combo in itertools.product(*options):
        out.add(" ".join(combo))
        if len(out) >= limit:
            break
    return out


# ---------------------------------------------------------------------------
# Recorded-response variants (parentheticals, "or" alternatives, slashes)

_GROUP_RE = re.compile(r"\(([^()]*)\)|\[([^\[\]]*)\]")


def _collapse(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def response_surface_variants(response: str) -> list[str]:
    """Surface forms acceptable for a recorded response, before normalization.

    `(Abraham) Lincoln` -> ["Lincoln", "Abraham Lincoln"]
    `Mars (or the Red Planet)` -> ["Mars", "the Red Planet"]
    `Falkland (or Malvinas) Islands` -> ["Falkland Islands", "Malvinas Islands", "Malvinas"]
    """
    response = clean_text(response)
    groups = list(_GROUP_RE.finditer(response))
    without = _collapse(_GROUP_RE.sub(" ", response))
    variants = [without]

    variants.append(canonical_display(response))

    for m in groups:
        content = (m.group(1) if m.group(1) is not None else m.group(2)).strip()
        low = content.lower()
        alt = None
        if low.startswith("or "):
            alt = content[3:].strip()
        elif low.endswith(" accepted"):
            alt = content[: -len(" accepted")].strip()
        if not alt:
            continue
        variants.append(alt)
        before = _GROUP_RE.sub(" ", response[: m.start()]).split()
        after = _GROUP_RE.sub(" ", response[m.end() :]).split()
        if before and after:
            variants.append(_collapse(" ".join([*before[:-1], alt, *after])))

    expanded: list[str] = []
    for v in variants:
        expanded.append(v)
        for tok in v.split():
            if "/" in tok:
                parts = tok.split("/")
                if all(p and p[0].isalpha() for p in parts):
                    for p in parts:
                        expanded.append(_collapse(v.replace(tok, p, 1)))
    seen: list[str] = []
    for v in expanded:
        if v and v not in seen:
            seen.append(v)
    return seen


def response_variants(response: str, aliases: list[str] | tuple[str, ...] = ()) -> set[str]:
    """Normalized, number-expanded strings that count as an exact match."""
    out: set[str] = set()
    for surface in [*response_surface_variants(response), *aliases]:
        n = normalize(surface)
        if n:
            out |= number_variants(n)
    return out


def canonical_display(response: str) -> str:
    """Response with optional parentheticals kept and alternatives dropped:
    "(Abraham) Lincoln" -> "Abraham Lincoln"; "Mars (or the Red Planet)" -> "Mars"."""

    def keep(m: re.Match) -> str:
        content = (m.group(1) if m.group(1) is not None else m.group(2)).strip()
        low = content.lower()
        return " " if low.startswith("or ") or low.endswith(" accepted") else f" {content} "

    return _collapse(_GROUP_RE.sub(keep, clean_text(response)))


def response_key(response: str) -> str:
    """Heuristic entity key: "(Abraham) Lincoln" and "Abraham Lincoln" share one."""
    return normalize(canonical_display(response))
