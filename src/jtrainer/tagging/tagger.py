"""Clue tagging: prompt construction, response validation, storage, batch runner, review.

Tags are versioned rows. Re-tagging under a new taxonomy version inserts new rows and
never touches old ones. Every LLM result is cached; tagging the same clue with the same
prompt and taxonomy makes no second API call.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field

import duckdb
import numpy as np

from jtrainer.clock import Clock, SystemClock
from jtrainer.db import to_db
from jtrainer.llm import (
    BatchClient,
    CachedLLM,
    LLMRequest,
    MalformedResponse,
    load_prompt,
    parse_json_object,
)
from jtrainer.tagging.entities import EntityResolver
from jtrainer.tagging.taxonomy import Taxonomy, current_tags_sql


class InvalidTopic(MalformedResponse):
    pass


@dataclass
class TagResult:
    topic_id: str
    secondary_topic: str | None
    canonical_name: str
    aliases: list[str] = field(default_factory=list)


@dataclass
class TagRunReport:
    tagged: int = 0
    from_cache: int = 0
    submitted: int = 0
    already_tagged: int = 0
    failed: Counter = field(default_factory=Counter)
    retried: int = 0


def _schema(topic_ids: list[str]) -> dict:
    return {
        "type": "object",
        "properties": {
            "topic_id": {"type": "string", "enum": topic_ids},
            "secondary_topic": {"anyOf": [{"type": "string", "enum": topic_ids}, {"type": "null"}]},
            "entity": {
                "type": "object",
                "properties": {
                    "canonical_name": {"type": "string"},
                    "aliases": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["canonical_name", "aliases"],
                "additionalProperties": False,
            },
        },
        "required": ["topic_id", "secondary_topic", "entity"],
        "additionalProperties": False,
    }


class Tagger:
    def __init__(
        self,
        conn: duckdb.DuckDBPyConnection,
        taxonomy: Taxonomy,
        llm: CachedLLM,
        model: str,
        clock: Clock | None = None,
        max_tokens: int = 2048,
    ):
        self.conn = conn
        self.taxonomy = taxonomy
        self.llm = llm
        self.model = model
        self.clock = clock or SystemClock()
        self.max_tokens = max_tokens
        self.prompt = load_prompt("tag")
        self.resolver = EntityResolver(conn)
        self._schema = _schema(taxonomy.ids)

    # -- requests and parsing ------------------------------------------------

    def _clue(self, clue_id: int) -> dict:
        row = self.conn.execute(
            "SELECT category, category_comment, clue_text, response FROM clues WHERE clue_id = ?",
            [clue_id],
        ).fetchone()
        if row is None:
            raise KeyError(f"no clue {clue_id}")
        return dict(
            zip(["category", "category_comment", "clue_text", "response"], row, strict=True)
        )

    def request_for(self, clue_id: int) -> LLMRequest:
        system, user = self.prompt.render(
            taxonomy=self.taxonomy.prompt_block(), **self._clue(clue_id)
        )
        # The taxonomy version is part of the system text's identity for caching.
        system = f"[taxonomy {self.taxonomy.version}]\n{system}"
        return LLMRequest(
            self.prompt.version, self.model, system, user, self.max_tokens, self._schema
        )

    def parse(self, text: str) -> TagResult:
        obj = parse_json_object(text)
        topic = obj.get("topic_id")
        if topic not in self.taxonomy.ids:
            raise InvalidTopic(
                f"topic_id {topic!r} is not in taxonomy {self.taxonomy.version}", text
            )
        secondary = obj.get("secondary_topic")
        if secondary not in self.taxonomy.ids or secondary == topic:
            secondary = None
        entity = obj.get("entity") or {}
        name = entity.get("canonical_name") if isinstance(entity, dict) else None
        if not isinstance(name, str) or not name.strip():
            raise MalformedResponse("missing entity.canonical_name", text)
        aliases = entity.get("aliases") or []
        if not isinstance(aliases, list):
            raise MalformedResponse("entity.aliases must be a list", text)
        return TagResult(topic, secondary, name.strip(), [str(a) for a in aliases])

    # -- storage ---------------------------------------------------------------

    def store(
        self,
        clue_id: int,
        result: TagResult,
        model: str | None = None,
        prompt_version: str | None = None,
    ) -> None:
        entity_id = self.resolver.resolve(result.canonical_name, result.aliases)
        self.conn.execute(
            "INSERT INTO clue_tags VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [
                clue_id,
                self.taxonomy.version,
                prompt_version or self.prompt.version,
                model or self.model,
                result.topic_id,
                result.secondary_topic,
                entity_id,
                to_db(self.clock.now()),
            ],
        )

    def record_failure(self, clue_id: int, reason: str, raw: str | None) -> None:
        self.conn.execute(
            "INSERT INTO tag_failures VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                clue_id,
                self.taxonomy.version,
                self.prompt.version,
                self.model,
                reason,
                raw,
                to_db(self.clock.now()),
            ],
        )

    def _already_tagged(self, clue_id: int) -> bool:
        return (
            self.conn.execute(
                "SELECT count(*) FROM clue_tags WHERE clue_id = ? AND taxonomy_version = ? "
                "AND prompt_version = ? AND model = ?",
                [clue_id, self.taxonomy.version, self.prompt.version, self.model],
            ).fetchone()[0]
            > 0
        )

    # -- synchronous tagging ---------------------------------------------------

    def tag_clue(self, clue_id: int, report: TagRunReport | None = None) -> bool:
        report = report or TagRunReport()
        if self._already_tagged(clue_id):
            report.already_tagged += 1
            return True
        req = self.request_for(clue_id)
        cached = self.llm.cached_text(req) is not None
        try:
            result = self.llm.complete(req, self.parse, retries=1)
        except InvalidTopic as exc:
            report.failed["invalid_topic"] += 1
            self.record_failure(clue_id, "invalid_topic", exc.raw)
            return False
        except MalformedResponse as exc:
            report.failed["malformed"] += 1
            self.record_failure(clue_id, "malformed", exc.raw)
            return False
        self.store(clue_id, result)
        report.tagged += 1
        report.from_cache += cached
        return True

    def tag_clues(self, clue_ids: list[int]) -> TagRunReport:
        report = TagRunReport()
        for cid in clue_ids:
            self.tag_clue(cid, report)
        return report

    # -- selection -------------------------------------------------------------

    def untagged_clue_ids(
        self,
        limit: int | None = None,
        include_ineligible: bool = False,
        include_tagged: bool = False,
    ) -> list[int]:
        where = (
            ""
            if include_ineligible
            else ("AND NOT c.needs_media AND NOT c.is_tournament AND NOT c.is_special")
        )
        untagged = "TRUE" if include_tagged else "t.clue_id IS NULL"
        sql = (
            f"SELECT c.clue_id FROM clues c LEFT JOIN ({current_tags_sql(self.taxonomy.version)})"
            f" t USING (clue_id) WHERE {untagged} {where} ORDER BY c.clue_id"
        )
        if limit:
            sql += f" LIMIT {int(limit)}"
        return [r[0] for r in self.conn.execute(sql).fetchall()]

    def sample_clue_ids(self, n: int, rng: np.random.Generator) -> list[int]:
        """A sample of eligible clues, tagged or not, so the same seed always names the same
        clues: an interrupted run resumes (tagged clues are skipped) rather than drawing a
        new sample. Use a different seed for a new sample."""
        pool = np.array(self.untagged_clue_ids(include_tagged=True), dtype=np.int64)
        if len(pool) == 0:
            return []
        picks = rng.choice(pool, size=min(n, len(pool)), replace=False)
        return sorted(int(x) for x in picks)

    # -- batch -----------------------------------------------------------------

    def submit_batch(
        self, client: BatchClient, clue_ids: list[int]
    ) -> tuple[str | None, TagRunReport]:
        """Store cached results directly; submit the rest as one Message Batch."""
        report = TagRunReport()
        items: list[tuple[str, LLMRequest]] = []
        for cid in clue_ids:
            if self._already_tagged(cid):
                continue
            req = self.request_for(cid)
            cached = self.llm.cached_text(req)
            if cached is not None:
                try:
                    self.store(cid, self.parse(cached))
                    report.tagged += 1
                    report.from_cache += 1
                    continue
                except MalformedResponse:
                    pass
            items.append((f"clue-{cid}", req))
        if not items:
            return None, report
        batch_id = client.submit_batch(items)
        self.conn.execute(
            "INSERT INTO tag_batches VALUES (?, ?, ?, ?, ?, ?, NULL)",
            [
                batch_id,
                self.taxonomy.version,
                self.prompt.version,
                self.model,
                len(items),
                to_db(self.clock.now()),
            ],
        )
        report.submitted = len(items)
        return batch_id, report

    def collect_batch(self, client: BatchClient, batch_id: str) -> TagRunReport:
        """Map results back to clues by custom_id. Malformed output gets one synchronous
        retry; anything still failing is recorded with its raw text, and the batch goes on."""
        report = TagRunReport()
        for res in client.batch_results(batch_id):
            try:
                clue_id = int(res.custom_id.removeprefix("clue-"))
            except ValueError:
                report.failed["bad_custom_id"] += 1
                continue
            if res.kind != "succeeded" or res.text is None:
                report.failed[res.kind] += 1
                self.record_failure(clue_id, f"batch_{res.kind}", res.error)
                continue
            req = self.request_for(clue_id)
            try:
                result = self.parse(res.text)
                self.llm.store(req, res.text)
            except MalformedResponse:
                report.retried += 1
                try:
                    result = self.llm.complete(req, self.parse, retries=0)
                except MalformedResponse as exc:
                    reason = "invalid_topic" if isinstance(exc, InvalidTopic) else "malformed"
                    report.failed[reason] += 1
                    self.record_failure(clue_id, reason, exc.raw or res.text)
                    continue
            if not self._already_tagged(clue_id):
                self.store(clue_id, result)
            report.tagged += 1
        self.conn.execute(
            "UPDATE tag_batches SET collected_at = ? WHERE batch_id = ?",
            [to_db(self.clock.now()), batch_id],
        )
        return report


# ---------------------------------------------------------------------------
# Hand review (milestone 2 sample review)


class TagReview:
    def __init__(
        self, conn: duckdb.DuckDBPyConnection, taxonomy_version: str, clock: Clock | None = None
    ):
        self.conn = conn
        self.version = taxonomy_version
        self.clock = clock or SystemClock()

    def queue(self, limit: int = 500) -> list[dict]:
        """Model-tagged clues not yet reviewed under this version."""
        rows = self.conn.execute(
            f"""
            SELECT t.clue_id, t.topic_id, t.secondary_topic, c.category, c.clue_text,
                   c.response, e.canonical_name
            FROM ({current_tags_sql(self.version)}) t
            JOIN clues c USING (clue_id)
            LEFT JOIN entities e USING (entity_id)
            WHERE t.clue_id NOT IN (SELECT clue_id FROM tag_reviews WHERE taxonomy_version = ?)
            ORDER BY t.clue_id LIMIT ?
            """,
            [self.version, limit],
        ).fetchall()
        cols = [
            "clue_id",
            "topic_id",
            "secondary_topic",
            "category",
            "clue_text",
            "response",
            "entity",
        ]
        return [dict(zip(cols, r, strict=True)) for r in rows]

    def _current(self, clue_id: int) -> tuple[str | None, int | None]:
        row = self.conn.execute(
            f"SELECT topic_id, entity_id FROM ({current_tags_sql(self.version)}) WHERE clue_id = ?",
            [clue_id],
        ).fetchone()
        return (row[0], row[1]) if row else (None, None)

    def _log(
        self, clue_id: int, action: str, original: str | None, new: str | None, note: str | None
    ) -> None:
        self.conn.execute(
            "INSERT INTO tag_reviews VALUES (?, ?, ?, ?, ?, ?, ?)",
            [clue_id, self.version, action, original, new, note, to_db(self.clock.now())],
        )

    def accept(self, clue_id: int) -> None:
        topic, _ = self._current(clue_id)
        self._log(clue_id, "accept", topic, topic, None)

    def change(self, clue_id: int, new_topic: str) -> None:
        valid = {
            r[0]
            for r in self.conn.execute(
                "SELECT topic_id FROM topics WHERE taxonomy_version = ?", [self.version]
            ).fetchall()
        }
        if new_topic not in valid:
            raise ValueError(f"{new_topic!r} is not a topic in {self.version}")
        topic, entity_id = self._current(clue_id)
        self._log(clue_id, "change", topic, new_topic, None)
        self.conn.execute(
            "INSERT INTO clue_tags VALUES (?, ?, 'human', 'human', ?, NULL, ?, ?)",
            [clue_id, self.version, new_topic, entity_id, to_db(self.clock.now())],
        )

    def flag(self, clue_id: int, note: str) -> None:
        topic, _ = self._current(clue_id)
        self._log(clue_id, "flag", topic, None, note)

    def summary(self) -> dict:
        rows = self.conn.execute(
            "SELECT action, count(*) FROM tag_reviews WHERE taxonomy_version = ? GROUP BY 1",
            [self.version],
        ).fetchall()
        counts = dict(rows)
        total = sum(counts.values())
        accepted = counts.get("accept", 0)
        return {**counts, "total": total, "accuracy": accepted / total if total else None}


def dump_failures(conn: duckdb.DuckDBPyConnection, version: str) -> str:
    rows = conn.execute(
        "SELECT clue_id, reason, raw_text FROM tag_failures "
        "WHERE taxonomy_version = ? ORDER BY failed_at",
        [version],
    ).fetchall()
    return json.dumps([{"clue_id": r[0], "reason": r[1], "raw": r[2]} for r in rows], indent=1)
