"""CLI smoke tests against a temporary config (no network)."""

import json
import shutil

import httpx
import pytest
from conftest import SAMPLE_TSV, TEST_TAXONOMY

from jtrainer.cli import main
from jtrainer.db import migrations
from jtrainer.ingest import jwolle1


@pytest.fixture
def cfg_path(tmp_path):
    (tmp_path / "taxonomy").mkdir()
    shutil.copy(TEST_TAXONOMY, tmp_path / "taxonomy" / "test-v1.yaml")
    p = tmp_path / "config.toml"
    p.write_text(
        '[paths]\ndb = "data/t.duckdb"\nraw_dir = "data/raw"\n'
        'html_cache = "data/html"\n[taxonomy]\nversion = "test-v1"\n'
    )
    return p


def test_ingest_taxonomy_stats_info(cfg_path, capsys):
    assert main(["--config", str(cfg_path), "ingest", "--file", str(SAMPLE_TSV)]) == 0
    assert "inserted 53" in capsys.readouterr().out
    assert main(["--config", str(cfg_path), "taxonomy", "load"]) == 0
    assert "7 topics" in capsys.readouterr().out
    assert main(["--config", str(cfg_path), "stats"]) == 0
    assert "attempts 0" in capsys.readouterr().out
    assert main(["--config", str(cfg_path), "info"]) == 0
    info = json.loads(capsys.readouterr().out)
    assert info["clues"] == 53 and info["schema_version"] == len(migrations())
    assert main(["--config", str(cfg_path), "tag", "status"]) == 0
    assert "0 clues tagged" in capsys.readouterr().out


def test_ingest_missing_file(cfg_path, capsys):
    assert main(["--config", str(cfg_path), "ingest"]) == 1
    assert "jtrain download" in capsys.readouterr().out


def test_scrape_requires_terms_confirmation(cfg_path, capsys):
    assert main(["--config", str(cfg_path), "scrape", "--season", "41"]) == 1
    assert "robots.txt" in capsys.readouterr().out


def test_fit_without_outcomes(cfg_path, capsys):
    assert main(["--config", str(cfg_path), "fit-irt"]) == 1


def test_tag_requires_credentials(cfg_path, monkeypatch, capsys):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    with pytest.raises(SystemExit):
        main(["--config", str(cfg_path), "tag", "sample", "--n", "3"])
    assert "ANTHROPIC_API_KEY" in capsys.readouterr().out


def test_golden_candidates(cfg_path, tmp_path, capsys):
    main(["--config", str(cfg_path), "ingest", "--file", str(SAMPLE_TSV)])
    from jtrainer import db as dbmod

    conn = dbmod.connect(tmp_path / "data" / "t.duckdb")
    conn.execute(
        "INSERT INTO sessions (started_at, mode, app_version) VALUES ('2026-01-01', 'random', 't')"
    )
    conn.execute(
        "INSERT INTO attempts (session_id, clue_id, shown_at, user_response, correct, "
        "grader, overridden, sampling_mode, sampling_prob) VALUES (1, 1, '2026-01-01', "
        "'Lincon', TRUE, 'user', TRUE, 'random', 0.1)"
    )
    conn.close()
    out = tmp_path / "cands.yaml"
    assert main(["--config", str(cfg_path), "golden-candidates", "--out", str(out)]) == 0
    assert main(["--config", str(cfg_path), "golden-candidates", "--out", str(out)]) == 0
    assert "added 0" in capsys.readouterr().out.splitlines()[-1]
    assert "Lincon" in out.read_text()


def test_download_with_mock_transport(tmp_path):
    served = []

    def handler(request):
        served.append(request.url.path)
        return httpx.Response(200, content=b"round\tclue_value\n")

    paths = jwolle1.download(tmp_path, ["a.tsv"], transport=httpx.MockTransport(handler))
    assert paths[0].read_bytes() == b"round\tclue_value\n"
    jwolle1.download(tmp_path, ["a.tsv"], transport=httpx.MockTransport(handler))
    assert len(served) == 1  # present files are not fetched again
