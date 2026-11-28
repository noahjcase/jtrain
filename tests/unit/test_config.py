"""Config loading."""

import pytest

from jtrainer.config import Config


def test_defaults_match_shipped_config_toml():
    shipped = Config.load()
    default = Config()
    for section in ("grading", "sampling", "review", "priority", "difficulty", "eligibility"):
        assert getattr(shipped, section) == getattr(default, section), section


def test_unknown_key_rejected(tmp_path):
    p = tmp_path / "c.toml"
    p.write_text("[grading]\nfuzy_high = 3\n")
    with pytest.raises(ValueError, match="fuzy_high"):
        Config.load(p)


def test_mix_must_sum_to_one(tmp_path):
    p = tmp_path / "c.toml"
    p.write_text("[sampling.mix]\npriority = 0.5\nreview = 0.2\nrandom = 0.2\n")
    with pytest.raises(ValueError, match="sum to 1"):
        Config.load(p)


def test_negative_recency_half_life_rejected(tmp_path):
    p = tmp_path / "c.toml"
    p.write_text("[sampling]\nrecency_half_life_years = -1\n")
    with pytest.raises(ValueError, match="recency_half_life_years"):
        Config.load(p)


def test_missing_file_gives_defaults(tmp_path):
    assert Config.load(tmp_path / "none.toml").grading == Config().grading


def test_resolve_relative_to_config(tmp_path):
    p = tmp_path / "c.toml"
    p.write_text('[paths]\ndb = "x/y.duckdb"\n')
    cfg = Config.load(p)
    assert cfg.resolve(cfg.paths.db) == tmp_path / "x" / "y.duckdb"
