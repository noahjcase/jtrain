"""Ingest jwolle1's jeopardy_clue_dataset TSV files into `clues`.

The TSV's `answer` column is the clue text the host reads and `question` is the
correct response; they are mapped to `clue_text` and `response` here.
"""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import duckdb

from jtrainer.config import Config
from jtrainer.ingest.board import board_row, season_for
from jtrainer.ingest.flags import Flagger
from jtrainer.text import clean_text, response_key

DATASET_BASE = "https://raw.githubusercontent.com/jwolle1/jeopardy_clue_dataset/main"
MAIN_FILE = "combined_season1-42.tsv"
SPECIAL_FILE = "extra_matches.tsv"

EXPECTED_COLUMNS = [
    "round",
    "clue_value",
    "daily_double_value",
    "category",
    "comments",
    "answer",
    "question",
    "air_date",
    "notes",
]
_ROUND_MAP = {"1": 1, "2": 2, "3": 3, "single": 1, "double": 2, "final": 3}
_INSERT_COLUMNS = [
    "source",
    "source_ref",
    "source_line",
    "air_date",
    "season",
    "round",
    "board_row",
    "clue_value",
    "is_daily_double",
    "dd_wager",
    "category",
    "category_comment",
    "clue_text",
    "response",
    "response_key",
    "notes",
    "is_tournament",
    "is_special",
    "needs_media",
]


@dataclass
class IngestReport:
    path: str
    rows_read: int = 0
    inserted: int = 0
    skipped: Counter = field(default_factory=Counter)
    board_row_missing: int = 0  # J/DJ clues whose value is missing or off-grid
    tournament: int = 0
    needs_media: int = 0

    def check(self) -> None:
        total = self.inserted + sum(self.skipped.values())
        if total != self.rows_read:
            raise AssertionError(f"report does not balance: {total} != {self.rows_read}")

    def summary(self) -> str:
        skipped = ", ".join(f"{k}={v}" for k, v in sorted(self.skipped.items())) or "none"
        return (
            f"{self.path}: read {self.rows_read}, inserted {self.inserted}, "
            f"skipped [{skipped}]; board_row NULL for {self.board_row_missing} J/DJ clues; "
            f"flagged tournament={self.tournament}, needs_media={self.needs_media}"
        )


def _int_or_none(s: str | None) -> int | None:
    s = (s or "").strip().replace(",", "").replace("$", "")
    if not s:
        return None
    try:
        return int(float(s))
    except ValueError:
        return None


def make_source_ref(air_date: str, round_: int, category: str, value, clue: str, resp: str) -> str:
    key = "|".join([air_date, str(round_), category, str(value), clue, resp])
    return "jw:" + hashlib.sha1(key.encode("utf-8")).hexdigest()


def ingest_tsv(
    conn: duckdb.DuckDBPyConnection,
    path: str | Path,
    config: Config,
    *,
    special: bool = False,
    progress: bool = False,
) -> IngestReport:
    path = Path(path)
    flagger = Flagger(config.ingest.tournament_patterns, config.ingest.media_patterns)
    report = IngestReport(path=str(path))
    existing = {r[0] for r in conn.execute("SELECT source_ref FROM clues").fetchall()}
    seen: set[str] = set()
    out_rows: list[list] = []

    with path.open(encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh, delimiter="\t", quoting=csv.QUOTE_NONE)
        header = next(reader)
        if header[: len(EXPECTED_COLUMNS)] != EXPECTED_COLUMNS:
            raise ValueError(f"{path}: unexpected header {header}")
        for line_no, raw in enumerate(reader, start=1):
            if not raw or (len(raw) == 1 and not raw[0].strip()):
                continue
            if raw[0].startswith("#"):  # fixture comment lines
                continue
            report.rows_read += 1
            raw = (raw + [""] * len(EXPECTED_COLUMNS))[: len(EXPECTED_COLUMNS)]
            rec = dict(zip(EXPECTED_COLUMNS, raw, strict=True))
            round_ = _ROUND_MAP.get(rec["round"].strip().lower())
            if round_ is None:
                report.skipped["unsupported_round"] += 1
                continue
            try:
                air_date = dt.date.fromisoformat(rec["air_date"].strip())
            except ValueError:
                report.skipped["bad_date"] += 1
                continue
            clue_text = clean_text(rec["answer"])
            response = clean_text(rec["question"])
            if not clue_text or not response:
                report.skipped["missing_text"] += 1
                continue
            category = clean_text(rec["category"])
            value = _int_or_none(rec["clue_value"]) if round_ != 3 else None
            if value == 0:
                value = None
            dd_value = _int_or_none(rec["daily_double_value"]) or 0
            ref = make_source_ref(
                rec["air_date"].strip(), round_, category, value, clue_text, response
            )
            if ref in seen:
                report.skipped["duplicate_in_file"] += 1
                continue
            seen.add(ref)
            if ref in existing:
                report.skipped["already_present"] += 1
                continue
            notes = clean_text(rec["notes"]) or None
            row = board_row(round_, value, air_date)
            if round_ != 3 and row is None:
                report.board_row_missing += 1
            tournament = flagger.is_tournament(notes)
            media = flagger.needs_media(clue_text, response)
            report.tournament += tournament
            report.needs_media += media
            out_rows.append(
                [
                    "jwolle1",
                    ref,
                    line_no,
                    air_date.isoformat(),
                    season_for(air_date),
                    round_,
                    row,
                    value,
                    dd_value > 0,
                    dd_value if dd_value > 0 else None,
                    category,
                    clean_text(rec["comments"]) or None,
                    clue_text,
                    response,
                    response_key(response),
                    notes,
                    tournament,
                    special,
                    media,
                ]
            )
            if progress and report.rows_read % 50000 == 0:
                print(f"  ...{report.rows_read} rows", file=sys.stderr)

    report.inserted = _bulk_insert(conn, out_rows)
    report.check()
    return report


def _bulk_insert(conn: duckdb.DuckDBPyConnection, rows: list[list]) -> int:
    if not rows:
        return 0
    with tempfile.TemporaryDirectory() as tmp:
        csv_path = Path(tmp) / "clues.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh, quoting=csv.QUOTE_ALL)
            writer.writerow(_INSERT_COLUMNS)
            writer.writerows(["" if v is None else v for v in r] for r in rows)
        types = {
            "source_line": "INTEGER",
            "air_date": "DATE",
            "season": "INTEGER",
            "round": "TINYINT",
            "board_row": "TINYINT",
            "clue_value": "INTEGER",
            "is_daily_double": "BOOLEAN",
            "dd_wager": "INTEGER",
            "is_tournament": "BOOLEAN",
            "is_special": "BOOLEAN",
            "needs_media": "BOOLEAN",
        }
        col_types = {c: types.get(c, "VARCHAR") for c in _INSERT_COLUMNS}
        cols = ", ".join(_INSERT_COLUMNS)
        select = ", ".join(
            f"NULLIF({c}, '')" if col_types[c] == "VARCHAR" else c for c in _INSERT_COLUMNS
        )
        before = conn.execute("SELECT count(*) FROM clues").fetchone()[0]
        conn.execute(
            f"INSERT INTO clues ({cols}) SELECT {select} FROM read_csv(?, header=true, "
            f"quote='\"', escape='\"', columns={col_types!r}, nullstr='')",
            [str(csv_path)],
        )
        after = conn.execute("SELECT count(*) FROM clues").fetchone()[0]
    return after - before


def download(dest_dir: str | Path, files: list[str] | None = None, transport=None) -> list[Path]:
    """Download dataset files from GitHub into dest_dir (skips files already present).
    Writes to a .part file first so an interrupted download never looks complete."""
    import httpx

    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    paths = []
    with httpx.Client(transport=transport, follow_redirects=True, timeout=120) as client:
        for name in files or [MAIN_FILE, SPECIAL_FILE]:
            target = dest / name
            if not target.exists():
                part = target.with_suffix(target.suffix + ".part")
                with client.stream("GET", f"{DATASET_BASE}/{name}") as resp:
                    resp.raise_for_status()
                    with part.open("wb") as fh:
                        for chunk in resp.iter_bytes():
                            fh.write(chunk)
                part.rename(target)
            paths.append(target)
    return paths
