"""Quizzing precedes tagging: attempts made before tags exist get topics retroactively."""

import datetime as dt

from conftest import SAMPLE_TSV, TEST_TAXONOMY, make_config, tag_all

from jtrainer.ingest.jwolle1 import ingest_tsv
from jtrainer.model.stats import compute_stats
from jtrainer.service import StudyService
from jtrainer.tagging.taxonomy import load_taxonomy, store_taxonomy


def test_attempts_before_tagging_count_after(db, clock, rng):
    cfg = make_config()
    ingest_tsv(db, SAMPLE_TSV, cfg)
    service = StudyService(db, cfg, clock, rng)
    service.start_session("random")
    for _ in range(10):
        p = service.next_clue()
        service.submit(p, p.clue["response"], 3, 500, 1500)
    service.end_session()

    before = compute_stats(db, cfg)
    assert before.overall.n == 10 and before.untagged_attempts == 10
    assert not any(t.accuracy.n for t in before.topics)

    store_taxonomy(db, load_taxonomy(TEST_TAXONOMY))
    tag_all(db, when=dt.datetime(2026, 6, 1))
    after = compute_stats(db, cfg)
    assert after.untagged_attempts == 0
    assert sum(t.accuracy.n for t in after.topics) == 10
    assert sum(t.accuracy.n for t in after.topics if t.mu != 0.0) == 10
