"""Answer-entity normalization and deduplication.

Entities are keyed by the normalized canonical name. J! Archive parentheticals are
resolved first, so "(Abraham) Lincoln" and "Abraham Lincoln" share one entity. Aliases
are *not* used as match keys (a "Lincoln" alias must not merge Lincoln, Nebraska into the
president); they are stored for grading. Distinct names stay distinct ("John Adams" /
"John Quincy Adams").
"""

from __future__ import annotations

import duckdb

from jtrainer.text import canonical_display, normalize, response_key


class EntityResolver:
    def __init__(self, conn: duckdb.DuckDBPyConnection):
        self.conn = conn
        self._cache: dict[str, tuple[int, list[str]]] = {}

    def _lookup(self, key: str) -> tuple[int, list[str]] | None:
        if key in self._cache:
            return self._cache[key]
        row = self.conn.execute(
            "SELECT entity_id, aliases FROM entities WHERE canonical_key = ?", [key]
        ).fetchone()
        if row:
            self._cache[key] = (row[0], list(row[1] or []))
            return self._cache[key]
        return None

    def resolve(
        self, canonical_name: str, aliases: list[str] | tuple[str, ...] = (), source: str = "llm"
    ) -> int:
        name = canonical_display(canonical_name)
        key = normalize(name)
        if not key:
            raise ValueError(f"cannot resolve an empty entity name: {canonical_name!r}")
        found = self._lookup(key)
        if found is None:
            clean_aliases = self._new_aliases(key, [], aliases)
            entity_id = self.conn.execute(
                "INSERT INTO entities (canonical_name, canonical_key, aliases, source) "
                "VALUES (?, ?, ?, ?) RETURNING entity_id",
                [name, key, clean_aliases, source],
            ).fetchone()[0]
            self._cache[key] = (entity_id, clean_aliases)
            return entity_id
        entity_id, existing = found
        additions = self._new_aliases(key, existing, aliases)
        if additions:
            merged = existing + additions
            self.conn.execute(
                "UPDATE entities SET aliases = ? WHERE entity_id = ?", [merged, entity_id]
            )
            self._cache[key] = (entity_id, merged)
        if source == "llm":
            self.conn.execute(
                "UPDATE entities SET source = 'llm', canonical_name = ? "
                "WHERE entity_id = ? AND source = 'heuristic'",
                [name, entity_id],
            )
        return entity_id

    @staticmethod
    def _new_aliases(key: str, existing: list[str], candidates) -> list[str]:
        have = {normalize(a) for a in existing} | {key}
        out = []
        for a in candidates or ():
            a = canonical_display(str(a))
            na = normalize(a)
            if na and na not in have:
                have.add(na)
                out.append(a)
        return out

    def heuristic_for_response(self, response: str) -> int | None:
        if not response_key(response):
            return None
        return self.resolve(response, (), source="heuristic")


def entity_for_clue(
    conn: duckdb.DuckDBPyConnection,
    clue_id: int,
    taxonomy_version: str,
    resolver: EntityResolver | None = None,
) -> int | None:
    """The clue's entity: from its current tag if tagged, else a heuristic entity built from
    the normalized response (so spaced review works before any LLM tagging)."""
    from jtrainer.tagging.taxonomy import current_tags_sql

    row = conn.execute(
        f"SELECT entity_id FROM ({current_tags_sql(taxonomy_version)}) WHERE clue_id = ?",
        [clue_id],
    ).fetchone()
    if row and row[0] is not None:
        return row[0]
    resp = conn.execute("SELECT response FROM clues WHERE clue_id = ?", [clue_id]).fetchone()
    if not resp:
        return None
    return (resolver or EntityResolver(conn)).heuristic_for_response(resp[0])


def aliases_for_clue(
    conn: duckdb.DuckDBPyConnection, clue_id: int, taxonomy_version: str
) -> list[str]:
    from jtrainer.tagging.taxonomy import current_tags_sql

    row = conn.execute(
        f"SELECT e.canonical_name, e.aliases FROM ({current_tags_sql(taxonomy_version)}) t "
        "JOIN entities e USING (entity_id) WHERE t.clue_id = ?",
        [clue_id],
    ).fetchone()
    if not row:
        return []
    return [row[0], *(row[1] or [])]


def clues_for_entity(
    conn: duckdb.DuckDBPyConnection, entity_id: int, taxonomy_version: str
) -> list[int]:
    """All clues whose response is this entity: tagged with it, or sharing its response key."""
    from jtrainer.tagging.taxonomy import current_tags_sql

    row = conn.execute(
        "SELECT canonical_key, aliases FROM entities WHERE entity_id = ?", [entity_id]
    ).fetchone()
    if not row:
        return []
    key = row[0]
    rows = conn.execute(
        f"""
        SELECT clue_id FROM ({current_tags_sql(taxonomy_version)}) WHERE entity_id = ?
        UNION
        SELECT c.clue_id FROM clues c
        LEFT JOIN ({current_tags_sql(taxonomy_version)}) t USING (clue_id)
        WHERE c.response_key = ? AND (t.entity_id IS NULL OR t.entity_id = ?)
        ORDER BY 1
        """,
        [entity_id, key, entity_id],
    ).fetchall()
    return [r[0] for r in rows]
