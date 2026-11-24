"""Configuration: typed sections loaded from config.toml.

Defaults here mirror config.toml so tests can build a Config explicitly without the file.
"""

from __future__ import annotations

import dataclasses
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Paths:
    db: str = "data/jtrainer.duckdb"
    raw_dir: str = "data/raw"
    html_cache: str = "data/html_cache"
    taxonomy_dir: str = "taxonomy"


@dataclass
class Ingest:
    tournament_patterns: list[str] = field(
        default_factory=lambda: [
            r"\bkids?\b",
            r"\bteen\b",
            r"\bcollege\b",
            "back to school",
            "high school",
            r"\bcelebrity\b",
            "power players",
        ]
    )
    media_patterns: list[str] = field(
        default_factory=lambda: [
            "clue crew",
            r"\b(seen|heard|shown|pictured|displayed|played) here\b",
            r"\b(seen|shown) (on|in) (the|this) (monitor|video|map|photo|picture)\b",
            r"\bin this (video|clip|photo|picture)\b",
            r"\bon the monitor\b",
            r"\bas you (see|hear)\b",
            r"\blisten (to|for) (this|these|the following)\b",
        ]
    )


@dataclass
class Eligibility:
    include_media: bool = False
    include_tournament: bool = False
    include_special: bool = False


@dataclass
class Taxonomy:
    version: str = "v1"


@dataclass
class LLM:
    judge_model: str = "claude-opus-5-5"
    tag_model: str = "claude-haiku-5-5"
    effort: str = "low"
    judge_max_tokens: int = 2048
    tag_max_tokens: int = 2048
    batch_size: int = 10000


@dataclass
class Grading:
    fuzzy_high: float = 92.0
    fuzzy_low: float = 60.0
    min_len_fuzzy: int = 5


@dataclass
class Difficulty:
    j_row1: float = -1.6
    row_step: float = 0.45
    dj_offset: float = 0.7
    final: float = 1.4


@dataclass
class Ability:
    prior_mean: float = 0.0
    prior_sd: float = 1.5
    drift_sd_per_day: float = 0.01


@dataclass
class Priority:
    half_life_years: float = 8.0
    target_round: int = 2
    target_row: int = 3


@dataclass
class Review:
    desired_retention: float = 0.9
    slow_ms: int = 12000
    learning_steps_minutes: list[float] = field(default_factory=list)
    relearning_steps_minutes: list[float] = field(default_factory=list)


@dataclass
class Sampling:
    session_length: int = 50
    difficulty_kernel_sd: float = 0.8
    mix: dict[str, float] = field(
        default_factory=lambda: {"priority": 0.6, "review": 0.2, "random": 0.2}
    )


@dataclass
class Confidence:
    probs: list[float] = field(default_factory=lambda: [0.3, 0.6, 0.9])


@dataclass
class Scrape:
    base_url: str = "https://j-archive.com"
    min_interval_s: float = 20.0
    max_retries: int = 3
    backoff_s: float = 30.0
    user_agent: str = "jtrainer/0.1 (personal study tool)"


@dataclass
class Config:
    paths: Paths = field(default_factory=Paths)
    ingest: Ingest = field(default_factory=Ingest)
    eligibility: Eligibility = field(default_factory=Eligibility)
    taxonomy: Taxonomy = field(default_factory=Taxonomy)
    llm: LLM = field(default_factory=LLM)
    grading: Grading = field(default_factory=Grading)
    difficulty: Difficulty = field(default_factory=Difficulty)
    ability: Ability = field(default_factory=Ability)
    priority: Priority = field(default_factory=Priority)
    review: Review = field(default_factory=Review)
    sampling: Sampling = field(default_factory=Sampling)
    confidence: Confidence = field(default_factory=Confidence)
    scrape: Scrape = field(default_factory=Scrape)
    root: Path = PROJECT_ROOT

    @classmethod
    def from_dict(cls, data: dict[str, Any], root: Path = PROJECT_ROOT) -> Config:
        cfg = cls(root=root)
        for f in dataclasses.fields(cls):
            if f.name == "root" or f.name not in data:
                continue
            section = getattr(cfg, f.name)
            known = {sf.name for sf in dataclasses.fields(section)}
            unknown = set(data[f.name]) - known
            if unknown:
                raise ValueError(f"Unknown keys in [{f.name}]: {sorted(unknown)}")
            setattr(cfg, f.name, dataclasses.replace(section, **data[f.name]))
        mix = cfg.sampling.mix
        if abs(sum(mix.values()) - 1.0) > 1e-9:
            raise ValueError(f"[sampling.mix] must sum to 1, got {mix}")
        return cfg

    @classmethod
    def load(cls, path: str | Path | None = None) -> Config:
        path = Path(path) if path else PROJECT_ROOT / "config.toml"
        if not path.exists():
            return cls()
        with path.open("rb") as fh:
            return cls.from_dict(tomllib.load(fh), root=path.resolve().parent)

    def resolve(self, rel: str) -> Path:
        p = Path(rel)
        return p if p.is_absolute() else self.root / p
