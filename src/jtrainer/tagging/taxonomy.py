"""Topic taxonomy: YAML loading, validation, storage, and current-tag queries."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import duckdb
import yaml

from jtrainer.clock import Clock, SystemClock
from jtrainer.db import to_db

_ID_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_VERSION_RE = re.compile(r"^[A-Za-z0-9._-]+$")


class TaxonomyError(ValueError):
    pass


@dataclass
class Topic:
    topic_id: str
    label: str
    description: str
    examples: list[str] = field(default_factory=list)


@dataclass
class Taxonomy:
    version: str
    topics: list[Topic]
    notes: str = ""

    @property
    def ids(self) -> list[str]:
        return [t.topic_id for t in self.topics]

    def get(self, topic_id: str) -> Topic | None:
        return next((t for t in self.topics if t.topic_id == topic_id), None)

    def prompt_block(self) -> str:
        return "\n".join(f"{t.topic_id}: {t.label} - {t.description}" for t in self.topics)


def parse_taxonomy(data: dict, source: str = "<taxonomy>") -> Taxonomy:
    if not isinstance(data, dict):
        raise TaxonomyError(f"{source}: top level must be a mapping")
    version = str(data.get("version", "")).strip()
    if not version or not _VERSION_RE.match(version):
        raise TaxonomyError(f"{source}: missing or invalid 'version'")
    raw_topics = data.get("topics")
    if not isinstance(raw_topics, list) or not raw_topics:
        raise TaxonomyError(f"{source}: 'topics' must be a non-empty list")
    topics: list[Topic] = []
    seen: set[str] = set()
    for i, t in enumerate(raw_topics):
        if not isinstance(t, dict):
            raise TaxonomyError(f"{source}: topic #{i + 1} must be a mapping")
        tid = str(t.get("id", "")).strip()
        if not tid or not _ID_RE.match(tid):
            raise TaxonomyError(f"{source}: topic #{i + 1} has missing or invalid id {tid!r}")
        if tid in seen:
            raise TaxonomyError(f"{source}: duplicate topic_id {tid!r}")
        seen.add(tid)
        desc = str(t.get("description") or "").strip()
        if not desc:
            raise TaxonomyError(f"{source}: topic {tid!r} is missing a description")
        topics.append(
            Topic(tid, str(t.get("label") or tid), desc, [str(e) for e in t.get("examples") or []])
        )
    return Taxonomy(version, topics, str(data.get("notes") or ""))


def load_taxonomy(path: str | Path) -> Taxonomy:
    path = Path(path)
    return parse_taxonomy(yaml.safe_load(path.read_text()), str(path))


def store_taxonomy(
    conn: duckdb.DuckDBPyConnection, tax: Taxonomy, clock: Clock | None = None
) -> bool:
    """Insert a taxonomy version. Returns False if it was already stored identically.

    Changing the topics of an existing version is refused: bump the version instead.
    """
    clock = clock or SystemClock()
    existing = conn.execute(
        "SELECT topic_id, label, description FROM topics WHERE taxonomy_version = ? "
        "ORDER BY topic_id",
        [tax.version],
    ).fetchall()
    if existing:
        wanted = sorted((t.topic_id, t.label, t.description) for t in tax.topics)
        if [tuple(r) for r in existing] != wanted:
            raise TaxonomyError(
                f"taxonomy {tax.version!r} is already stored with different topics; "
                "bump the version instead of editing it"
            )
        return False
    conn.execute(
        "INSERT INTO taxonomies VALUES (?, ?, ?)", [tax.version, to_db(clock.now()), tax.notes]
    )
    conn.executemany(
        "INSERT INTO topics VALUES (?, ?, ?, ?)",
        [[tax.version, t.topic_id, t.label, t.description] for t in tax.topics],
    )
    return True


def stored_topics(conn: duckdb.DuckDBPyConnection, version: str) -> list[tuple[str, str]]:
    return [
        tuple(r)
        for r in conn.execute(
            "SELECT topic_id, label FROM topics WHERE taxonomy_version = ? ORDER BY topic_id",
            [version],
        ).fetchall()
    ]


def current_tags_sql(version: str) -> str:
    """SQL for the current tag per clue under `version`: a human correction if one exists,
    else the most recent model tag. Columns: clue_id, topic_id, secondary_topic, entity_id."""
    if not _VERSION_RE.match(version):
        raise TaxonomyError(f"invalid taxonomy version {version!r}")
    return f"""
        SELECT clue_id, topic_id, secondary_topic, entity_id FROM (
            SELECT clue_id, topic_id, secondary_topic, entity_id,
                   row_number() OVER (PARTITION BY clue_id
                                      ORDER BY (model = 'human') DESC, tagged_at DESC) AS rn
            FROM clue_tags WHERE taxonomy_version = '{version}'
        ) WHERE rn = 1
    """
