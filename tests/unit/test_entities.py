"""Entity normalization and deduplication (TESTING §7)."""

from jtrainer.tagging.entities import EntityResolver


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
    aliases = db.execute("SELECT aliases FROM entities WHERE entity_id = ?", [eid]).fetchone()[0]
    assert aliases == ["Honest Abe", "Lincoln", "the Great Emancipator"]
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
