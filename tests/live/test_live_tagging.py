"""Live tagging against the real API (TESTING §14). Run manually: uv run pytest -m live"""

from pathlib import Path

import pytest

from jtrainer import db as dbmod
from jtrainer.clock import SystemClock
from jtrainer.config import Config
from jtrainer.llm import AnthropicClient, CachedLLM
from jtrainer.tagging.tagger import Tagger
from jtrainer.tagging.taxonomy import load_taxonomy, store_taxonomy

pytestmark = pytest.mark.live

ROOT = Path(__file__).parents[2]

# (category, clue, response, hand label)
CASES = [
    (
        "PRESIDENTS",
        'He held office "4 score & 7 years" after the founding',
        "Abraham Lincoln",
        "us_presidents",
    ),
    ("CAPITAL CITIES", "This capital of Peru was founded by Pizarro", "Lima", "world_capitals"),
    ("THE BARD", '"Out, damned spot!" is spoken in this play', "Macbeth", "shakespeare"),
    ("OPERA", "Verdi wrote this opera about an Ethiopian princess", "Aida", "opera_ballet"),
    ("THE BIBLE", "He was swallowed by a great fish", "Jonah", "bible"),
    ("ELEMENTARY", "Element with the symbol Au", "gold", "elements_chemistry"),
    ("RHYME TIME", "A pudgy feline", "a fat cat", "wordplay_rhyme"),
    (
        "BEFORE & AFTER",
        "Honest Abe's Nebraska capital namesake meets a car brand",
        "Lincoln Continental",
        "wordplay_before_after",
    ),
    ("THE OSCARS", "1997's Best Picture about a doomed ocean liner", "Titanic", "film"),
    ("ASTRONOMY", "The Red Planet", "Mars", "astronomy"),
]


def test_live_tagging(tmp_path):
    cfg = Config.load(ROOT / "config.toml")
    conn = dbmod.connect(tmp_path / "live.duckdb")
    tax = load_taxonomy(ROOT / "taxonomy" / f"{cfg.taxonomy.version}.yaml")
    store_taxonomy(conn, tax)
    ids = []
    for i, (cat, clue, resp, _) in enumerate(CASES):
        ids.append(
            conn.execute(
                "INSERT INTO clues (source, source_ref, round, category, clue_text, response) "
                "VALUES ('live', ?, 1, ?, ?, ?) RETURNING clue_id",
                [f"l{i}", cat, clue, resp],
            ).fetchone()[0]
        )
    tagger = Tagger(
        conn,
        tax,
        CachedLLM(AnthropicClient(cfg.llm.effort), conn, SystemClock()),
        cfg.llm.tag_model,
    )
    report = tagger.tag_clues(ids)
    assert not report.failed, report.failed
    got = dict(conn.execute("SELECT clue_id, topic_id FROM clue_tags").fetchall())
    assert all(t in tax.ids for t in got.values())
    agree = sum(got[cid] == case[3] for cid, case in zip(ids, CASES, strict=True))
    assert agree >= 8, {case[2]: got[cid] for cid, case in zip(ids, CASES, strict=True)}
