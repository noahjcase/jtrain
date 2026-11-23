"""Polite J! Archive client (milestone 5).

Rules (DESIGN §4.2): read robots.txt first and obey it (including Crawl-delay); at most one
request per `scrape.min_interval_s` (never less than the Crawl-delay); cache raw HTML to
disk and never re-fetch a cached page; back off on 429/5xx with a bounded retry count;
record 404s and do not retry them.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import re
from pathlib import Path
from urllib import robotparser

import duckdb
import httpx

from jtrainer.clock import Clock, SystemClock
from jtrainer.config import Scrape
from jtrainer.db import to_db
from jtrainer.ingest.flags import Flagger
from jtrainer.ingest.jarchive import ParsedGame
from jtrainer.text import clean_text, normalize, response_key


class ScrapeError(RuntimeError):
    pass


class Disallowed(ScrapeError):
    pass


class PoliteClient:
    def __init__(
        self,
        cfg: Scrape,
        cache_dir: str | Path,
        clock: Clock | None = None,
        transport: httpx.BaseTransport | None = None,
        conn: duckdb.DuckDBPyConnection | None = None,
    ):
        self.cfg = cfg
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.clock = clock or SystemClock()
        self.conn = conn
        self.http = httpx.Client(
            base_url=cfg.base_url,
            transport=transport,
            timeout=60,
            headers={"User-Agent": cfg.user_agent},
            follow_redirects=True,
        )
        self._last_request_ms: int | None = None
        self._robots: robotparser.RobotFileParser | None = None
        self.interval_s = cfg.min_interval_s
        self.requests_made = 0

    # -- cache ------------------------------------------------------------------

    def cache_path(self, path: str) -> Path:
        safe = re.sub(r"[^A-Za-z0-9._=-]+", "_", path.strip("/")) or "index"
        if len(safe) > 120:
            safe = safe[:80] + "_" + hashlib.sha1(path.encode()).hexdigest()[:16]
        return self.cache_dir / f"{safe}.html"

    def cached(self, path: str) -> str | None:
        p = self.cache_path(path)
        return p.read_text(encoding="utf-8") if p.exists() else None

    # -- robots -------------------------------------------------------------------

    @property
    def robots(self) -> robotparser.RobotFileParser:
        if self._robots is None:
            text = self.cached("robots.txt")
            if text is None:
                resp = self._request("/robots.txt")
                text = resp.text if resp.status_code == 200 else ""
                self.cache_path("robots.txt").write_text(text, encoding="utf-8")
            rp = robotparser.RobotFileParser()
            rp.parse(text.splitlines())
            self._robots = rp
            delay = rp.crawl_delay(self.cfg.user_agent)
            if delay:
                self.interval_s = max(self.interval_s, float(delay))
        return self._robots

    def allowed(self, path: str) -> bool:
        return self.robots.can_fetch(self.cfg.user_agent, self.cfg.base_url + path)

    # -- requests -----------------------------------------------------------------

    def _wait(self) -> None:
        if self._last_request_ms is not None:
            elapsed = self.clock.monotonic_ms() - self._last_request_ms
            remaining = self.interval_s * 1000 - elapsed
            if remaining > 0:
                self.clock.sleep(remaining / 1000)

    def _request(self, path: str) -> httpx.Response:
        self._wait()
        self._last_request_ms = self.clock.monotonic_ms()
        self.requests_made += 1
        return self.http.get(path)

    def _log(self, path: str, status: int, cache_path: Path | None) -> None:
        if self.conn is not None:
            self.conn.execute(
                "INSERT INTO scrape_log VALUES (?, ?, ?, ?) ON CONFLICT (url) DO UPDATE SET "
                "status = excluded.status, fetched_at = excluded.fetched_at, "
                "cache_path = excluded.cache_path",
                [path, status, to_db(self.clock.now()), str(cache_path) if cache_path else None],
            )

    def get(self, path: str) -> str | None:
        """Page text (from cache if present), or None for a 404."""
        cached = self.cached(path)
        if cached is not None:
            return cached
        if not self.allowed(path):
            raise Disallowed(f"robots.txt disallows {path}")
        for attempt in range(self.cfg.max_retries + 1):
            resp = self._request(path)
            if resp.status_code == 200:
                target = self.cache_path(path)
                target.write_text(resp.text, encoding="utf-8")
                self._log(path, 200, target)
                return resp.text
            if resp.status_code == 404:
                self._log(path, 404, None)
                return None
            if resp.status_code == 429 or resp.status_code >= 500:
                if attempt < self.cfg.max_retries:
                    self.clock.sleep(self.cfg.backoff_s * (2**attempt))
                continue
            self._log(path, resp.status_code, None)
            raise ScrapeError(f"GET {path} -> {resp.status_code}")
        self._log(path, resp.status_code, None)
        raise ScrapeError(
            f"GET {path} failed after {self.cfg.max_retries} retries "
            f"(last status {resp.status_code})"
        )


# ---------------------------------------------------------------------------
# Merge scraped games into the clue table


def _match_key(round_: int, category: str, clue_text: str) -> tuple[int, str, str]:
    return round_, normalize(category), normalize(clue_text)


def merge_game(
    conn: duckdb.DuckDBPyConnection, game: ParsedGame, flagger: Flagger
) -> dict[str, int]:
    """Attach contestant outcomes to existing clues (dedup on air date, round, category and
    clue text) and insert clues the base dataset lacks. Returns counts."""
    counts = {"matched": 0, "inserted": 0, "outcomes": 0}
    if game.air_date is None:
        return counts
    existing = {}
    for cid, rnd, cat, text in conn.execute(
        "SELECT clue_id, round, category, clue_text FROM clues WHERE air_date = ?", [game.air_date]
    ).fetchall():
        existing.setdefault(_match_key(rnd, cat or "", text), cid)
    for i, clue in enumerate(game.clues):
        if clue.is_tiebreaker:
            continue
        key = _match_key(clue.round, clue.category, clue.clue_text)
        clue_id = existing.get(key)
        if clue_id is None:
            pos = f"{clue.round}:{clue.col}:{clue.row}" if clue.round != 3 else "3"
            ref = f"ja:{game.game_id}:{pos}"
            row = conn.execute("SELECT clue_id FROM clues WHERE source_ref = ?", [ref]).fetchone()
            if row:
                clue_id = row[0]
            else:
                response = clean_text(clue.response)
                clue_id = conn.execute(
                    """
                    INSERT INTO clues (source, source_ref, source_line, air_date, season, round,
                        board_row, clue_value, is_daily_double, dd_wager, category,
                        category_comment, clue_text, response, response_key, notes,
                        is_tournament, is_special, needs_media)
                    VALUES ('jarchive', ?, ?, ?, NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL,
                        FALSE, FALSE, ?) RETURNING clue_id
                    """,
                    [
                        ref,
                        i,
                        game.air_date,
                        clue.round,
                        clue.row if clue.round != 3 else None,
                        clue.value,
                        clue.is_daily_double,
                        clue.dd_wager,
                        clue.category,
                        clue.category_comment,
                        clue.clue_text,
                        response,
                        response_key(response),
                        flagger.needs_media(clue.clue_text, response),
                    ],
                ).fetchone()[0]
                counts["inserted"] += 1
            existing[key] = clue_id
        else:
            counts["matched"] += 1
        conn.execute(
            "INSERT INTO clue_outcomes VALUES (?, ?, ?, ?) ON CONFLICT (clue_id) DO UPDATE SET "
            "n_correct = excluded.n_correct, n_incorrect = excluded.n_incorrect, "
            "triple_stumper = excluded.triple_stumper",
            [clue_id, min(clue.n_correct, 3), min(clue.n_incorrect, 3), clue.triple_stumper],
        )
        counts["outcomes"] += 1
    return counts


def season_path(season: int) -> str:
    return f"/showseason.php?season={season}"


def game_path(game_id: int) -> str:
    return f"/showgame.php?game_id={game_id}"


def utcnow() -> dt.datetime:
    return dt.datetime.now(dt.UTC)
