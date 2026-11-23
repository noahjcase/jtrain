"""Tagging (TESTING §7)."""

import datetime as dt
from pathlib import Path

import pytest
from conftest import FIXTURES, SAMPLE_TSV, TEST_TAXONOMY, clue_id_by_response, make_config

from jtrainer.ingest.jwolle1 import ingest_tsv
from jtrainer.llm import BatchResult, CachedLLM, parse_prompt
from jtrainer.tagging.tagger import Tagger
from jtrainer.tagging.taxonomy import (
    TaxonomyError,
    current_tags_sql,
    load_taxonomy,
    parse_taxonomy,
    store_taxonomy,
)

RESP = FIXTURES / "tag_responses"


def read(name: str) -> str:
    return (RESP / name).read_text()


@pytest.fixture
def setup(db, fake_llm, clock):
    ingest_tsv(db, SAMPLE_TSV, make_config())
    tax = load_taxonomy(TEST_TAXONOMY)
    store_taxonomy(db, tax, clock)
    tagger = Tagger(db, tax, CachedLLM(fake_llm, db, clock), "test-tagger", clock)
    return db, tagger, fake_llm


def test_taxonomy_loads():
    tax = load_taxonomy(TEST_TAXONOMY)
    assert tax.version == "test-v1" and "us_presidents" in tax.ids
    real = load_taxonomy(Path(__file__).parents[2] / "taxonomy" / "v1.yaml")
    assert 40 <= len(real.topics) <= 60


def test_taxonomy_rejects_duplicates_and_missing_descriptions():
    with pytest.raises(TaxonomyError, match="duplicate topic_id 'a'"):
        parse_taxonomy(
            {
                "version": "x",
                "topics": [{"id": "a", "description": "one"}, {"id": "a", "description": "two"}],
            }
        )
    with pytest.raises(TaxonomyError, match="'b' is missing a description"):
        parse_taxonomy({"version": "x", "topics": [{"id": "b", "label": "B"}]})
    with pytest.raises(TaxonomyError, match="version"):
        parse_taxonomy({"topics": [{"id": "a", "description": "d"}]})


def test_taxonomy_version_is_immutable(db, clock):
    tax = load_taxonomy(TEST_TAXONOMY)
    assert store_taxonomy(db, tax, clock) is True
    assert store_taxonomy(db, tax, clock) is False
    tax.topics[0].description = "changed"
    with pytest.raises(TaxonomyError, match="bump the version"):
        store_taxonomy(db, tax, clock)


def test_prompt_requires_version():
    with pytest.raises(ValueError, match="no version"):
        parse_prompt("x", "---\nname: x\n---\nsystem\n===USER===\nuser")
    with pytest.raises(ValueError, match="no version"):
        parse_prompt("x", "system\n===USER===\nuser")
    p = parse_prompt("x", "---\nversion: x-v9\n---\nsys $a\n===USER===\nuser $b")
    assert p.version == "x-v9" and p.render(a=1, b=2) == ("sys 1", "user 2")


def test_tag_inserts_row_with_versions(setup):
    db, tagger, llm = setup
    cid = clue_id_by_response(db, "(Abraham) Lincoln")
    llm.register(tagger.request_for(cid), read("valid_lincoln.json"))
    assert tagger.tag_clue(cid)
    row = db.execute(
        "SELECT taxonomy_version, prompt_version, model, topic_id, entity_id "
        "FROM clue_tags WHERE clue_id = ?",
        [cid],
    ).fetchone()
    assert row[:4] == ("test-v1", "tag-v1", "test-tagger", "us_presidents")
    assert db.execute(
        "SELECT canonical_name, aliases FROM entities WHERE entity_id = ?", [row[4]]
    ).fetchone() == ("Abraham Lincoln", ["Lincoln", "Honest Abe"])


def test_retag_new_version_inserts_and_keeps_old(setup, clock):
    db, tagger, llm = setup
    cid = clue_id_by_response(db, "Paris")
    llm.register(tagger.request_for(cid), read("valid_paris.json"))
    tagger.tag_clue(cid)
    old = db.execute("SELECT * FROM clue_tags").fetchall()

    tax2 = parse_taxonomy(
        {
            "version": "test-v2",
            "topics": [
                {"id": "world_geography", "description": "Geography v2"},
                {"id": "arts", "description": "Arts v2"},
            ],
        }
    )
    store_taxonomy(db, tax2, clock)
    tagger2 = Tagger(db, tax2, tagger.llm, "test-tagger", clock)
    llm.register(
        tagger2.request_for(cid),
        '{"topic_id": "arts", "secondary_topic": null, '
        '"entity": {"canonical_name": "Paris", "aliases": []}}',
    )
    tagger2.tag_clue(cid)
    rows = db.execute("SELECT taxonomy_version, topic_id FROM clue_tags ORDER BY 1").fetchall()
    assert rows == [("test-v1", "world_geography"), ("test-v2", "arts")]
    assert (
        db.execute("SELECT * FROM clue_tags WHERE taxonomy_version = 'test-v1'").fetchall() == old
    )


def test_queries_use_configured_version(setup, clock):
    db, tagger, llm = setup
    cid = clue_id_by_response(db, "Paris")
    db.execute("INSERT INTO taxonomies VALUES ('other', ?, '')", [dt.datetime(2025, 1, 1)])
    db.execute(
        "INSERT INTO clue_tags VALUES (?, 'other', 'p', 'm', 'zzz', NULL, NULL, ?)",
        [cid, dt.datetime(2025, 1, 1)],
    )
    llm.register(tagger.request_for(cid), read("valid_paris.json"))
    tagger.tag_clue(cid)
    v1 = db.execute(
        f"SELECT topic_id FROM ({current_tags_sql('test-v1')}) WHERE clue_id = ?", [cid]
    ).fetchall()
    other = db.execute(
        f"SELECT topic_id FROM ({current_tags_sql('other')}) WHERE clue_id = ?", [cid]
    ).fetchall()
    assert v1 == [("world_geography",)] and other == [("zzz",)]


def test_unknown_topic_rejected_and_logged(setup):
    db, tagger, llm = setup
    cid = clue_id_by_response(db, "Paris")
    llm.register(tagger.request_for(cid), read("invalid_topic.json"))
    assert tagger.tag_clue(cid) is False
    assert db.execute("SELECT count(*) FROM clue_tags").fetchone()[0] == 0
    reason, raw = db.execute("SELECT reason, raw_text FROM tag_failures").fetchone()
    assert reason == "invalid_topic" and "astrology" in raw


def test_malformed_retried_once_then_failure(setup):
    db, tagger, llm = setup
    cid = clue_id_by_response(db, "gold")
    llm.register(tagger.request_for(cid), read("malformed.txt"))
    assert tagger.tag_clue(cid) is False
    assert len(llm.calls) == 2
    assert db.execute("SELECT reason, raw_text FROM tag_failures").fetchone() == (
        "malformed",
        read("malformed.txt"),
    )


def test_fenced_json_accepted(setup):
    db, tagger, llm = setup
    cid = clue_id_by_response(db, "gold")
    llm.register(tagger.request_for(cid), read("fenced.txt"))
    assert tagger.tag_clue(cid)


def test_cache_prevents_second_call(setup, clock):
    db, tagger, llm = setup
    cid = clue_id_by_response(db, "gold")
    llm.register(tagger.request_for(cid), read("fenced.txt"))
    tagger.tag_clue(cid)
    db.execute("DELETE FROM clue_tags")  # even if tags are removed, the LLM result is cached
    tagger.tag_clue(cid)
    assert len(llm.calls) == 1
    assert db.execute("SELECT count(*) FROM clue_tags").fetchone()[0] == 1


def test_batch_maps_custom_ids_and_handles_partial(setup):
    db, tagger, llm = setup
    ids = {r: clue_id_by_response(db, r) for r in ["(Abraham) Lincoln", "Paris", "gold", "iron"]}
    llm.register(tagger.request_for(ids["(Abraham) Lincoln"]), read("valid_lincoln.json"))
    llm.register(tagger.request_for(ids["Paris"]), read("valid_paris.json"))
    # gold: malformed in batch, valid on the synchronous retry
    llm.register(tagger.request_for(ids["gold"]), read("malformed.txt"), read("fenced.txt"))
    batch_id, report = tagger.submit_batch(llm, list(ids.values()))
    assert report.submitted == 4
    llm.batch_outcomes[f"clue-{ids['iron']}"] = BatchResult(
        f"clue-{ids['iron']}", "errored", error="overloaded"
    )
    collected = tagger.collect_batch(llm, batch_id)
    assert collected.tagged == 3 and collected.retried == 1
    assert collected.failed == {"errored": 1}
    topics = dict(db.execute("SELECT clue_id, topic_id FROM clue_tags").fetchall())
    assert topics == {
        ids["(Abraham) Lincoln"]: "us_presidents",
        ids["Paris"]: "world_geography",
        ids["gold"]: "science",
    }
    assert db.execute("SELECT clue_id, reason FROM tag_failures").fetchall() == [
        (ids["iron"], "batch_errored")
    ]
    assert db.execute("SELECT collected_at IS NOT NULL FROM tag_batches").fetchone()[0]


def test_batch_submit_skips_cached(setup):
    db, tagger, llm = setup
    cid = clue_id_by_response(db, "Paris")
    llm.register(tagger.request_for(cid), read("valid_paris.json"))
    tagger.tag_clue(cid)
    db.execute("DELETE FROM clue_tags")
    batch_id, report = tagger.submit_batch(llm, [cid])
    assert batch_id is None and report.from_cache == 1 and len(llm.calls) == 1


def test_sample_selection_is_deterministic(setup, rng):
    _, tagger, _ = setup
    import numpy as np

    a = tagger.sample_clue_ids(5, np.random.default_rng(1))
    b = tagger.sample_clue_ids(5, np.random.default_rng(1))
    assert a == b and len(a) == 5
