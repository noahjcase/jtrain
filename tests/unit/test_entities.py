"""Entity normalization and deduplication (TESTING §7)."""

from jtrainer.tagging.entities import EntityResolver, entity_aliases


def test_parenthetical_resolves_to_one_entity(db):
    r = EntityResolver(db)
    a = r.resolve("(Abraham) Lincoln")
    b = r.resolve("Abraham Lincoln")
    assert a == b
    assert db.execute("SELECT canonical_name FROM entities").fetchall() == [("Abraham Lincoln",)]


def test_new_alias_added_not_duplicated(db):
    r = EntityResolver(db)
    eid = r.resolve("Abraham Lincoln", ["Honest Abe"])
    r.resolve("Abraham Lincoln", ["Honest Abe", "Lincoln", "honest abe"])
    r2 = EntityResolver(db)  # fresh cache reads from the database
    r2.resolve("Abraham Lincoln", ["Lincoln", "the Great Emancipator"])
    assert entity_aliases(db, eid) == ["Honest Abe", "Lincoln", "the Great Emancipator"]
    assert db.execute("SELECT count(*) FROM entities").fetchone()[0] == 1


def test_similar_names_stay_distinct(db):
    r = EntityResolver(db)
    assert r.resolve("John Adams") != r.resolve("John Quincy Adams")


def test_alias_does_not_merge_entities(db):
    r = EntityResolver(db)
    president = r.resolve("Abraham Lincoln", ["Lincoln"])
    city = r.resolve("Lincoln, Nebraska", ["Lincoln"])
    assert president != city


def test_heuristic_then_llm_upgrades(db):
    r = EntityResolver(db)
    h = r.heuristic_for_response("(Abraham) Lincoln")
    assert db.execute("SELECT source FROM entities").fetchone()[0] == "heuristic"
    assert r.resolve("Abraham Lincoln", ["Lincoln"]) == h
    assert db.execute("SELECT source FROM entities").fetchone()[0] == "llm"


def test_new_alias_on_entity_already_referenced(db):
    """Regression: DuckDB runs an UPDATE of a list column as delete+insert, which a foreign
    key from clue_tags/review_state rejects. Adding aliases must not touch `entities`."""
    r = EntityResolver(db)
    eid = r.resolve("vampires", ["vampire"])
    db.execute("INSERT INTO taxonomies VALUES ('v1', '2026-01-01', '')")
    db.execute(
        "INSERT INTO clues (source, source_ref, round, clue_text, response) "
        "VALUES ('t', 'r1', 1, 'Nosferatu, Dracula & Barnabas', 'vampires')"
    )
    db.execute(
        "INSERT INTO clue_tags VALUES (1, 'v1', 'p', 'm', 't', NULL, ?, '2026-01-01')", [eid]
    )
    db.execute("INSERT INTO review_state VALUES (?, '{}', '2026-01-01', NULL, NULL)", [eid])
    assert r.resolve("vampires", ["the undead"]) == eid
    assert EntityResolver(db).resolve("vampires", ["vampyr"]) == eid  # cold cache too
    assert entity_aliases(db, eid) == ["vampire", "the undead", "vampyr"]
