# jtrain

A terminal trainer for *Jeopardy!* preparation. It serves real archive clues, grades your
responses, logs every attempt, and steers study toward the topics where improvement is
worth the most. Design: [DESIGN.md](DESIGN.md). Test plan: [TESTING.md](TESTING.md).

## Quick start

```sh
uv sync
uv run jtrain download      # jwolle1 clue dataset (seasons 1-42) into data/raw/
uv run jtrain ingest        # ~10 s: 544k clues into data/jtrainer.duckdb
uv run jtrain taxonomy load # store taxonomy/v1.yaml
uv run jtrain               # open the trainer
```

Set `ANTHROPIC_API_KEY` to enable the LLM judge for borderline answers and topic tagging.
Without it everything still works: borderline answers ask you to decide (`y`/`n`).

## Using the trainer

| Key on home | Mode |
|---|---|
| `r` | Random quiz: uniform over eligible clues (the unbiased baseline) |
| `a` | Adaptive: 60% topic priority, 20% due reviews, 20% random, favouring recent seasons (see `config.toml`) |
| `v` | Spaced review: missed facts, shown through a *different* clue when one exists |
| `g` | Game: a full archived game (`a`) or a synthesized board (`s`), with wagering and Coryat |
| `s` | Stats: accuracy (naive and IPW-corrected), ability ± uncertainty, priorities, timing, calibration |
| `t` | Tag review: accept / change / flag sampled topic tags |

Answering: type, **Enter**, then **1/2/3** for confidence (guess/maybe/sure) before the
reveal. Blank Enter or **Esc** passes. On the verdict: **Enter** next, **o** override,
**q** end. **Ctrl+G** ends a session or game at any time. Sessions are 50 clues by
default.

Until clues are tagged, adaptive mode falls back to random sampling plus reviews, and stats
show overall figures only. Attempts made before tagging get topics retroactively.

## Topic tagging (milestone 2)

```sh
uv run jtrain tag sample --n 500   # synchronous; then review with `t` in the TUI
uv run jtrain tag status
uv run jtrain tag submit           # Message Batches API, up to llm.batch_size per call
uv run jtrain tag collect          # stores results; failures are logged with raw text
```

Every LLM result is cached in the database: nothing is sent twice for the same prompt
version, taxonomy version, model and input.

**Cost check before the full batch.** The models are set in `config.toml` (`llm.tag_model`,
`llm.judge_model`), defaulting to `claude-opus-5-5`. A tagging request is ~1,900 prompt
tokens (mostly the taxonomy, which is prompt-cached) plus a short JSON answer. Rough
estimates for the full eligible corpus (495k clues) at batch pricing:

| Model | Approx. full-corpus cost |
|---|---|
| `claude-opus-5-5` | $1,100 – $2,900 |
| `claude-sonnet-5-5` | $550 – $1,450 |
| `claude-haiku-5-5` | $40 – $75 |

The range depends on cache hit rate and thinking tokens. These are estimates, not quotes.
A sensible path: run the 500-clue sample (a few dollars on any model), hand-review it,
optionally re-run the sample with a cheaper model under a new taxonomy version and compare,
then choose the model for the full batch.

## Contestant outcomes and difficulty (milestone 5)

```sh
uv run jtrain scrape --season 41 --confirm-terms   # read J! Archive's terms first
uv run jtrain fit-irt                              # fit clue difficulties, used everywhere
```

The client reads robots.txt first and obeys it, including its `Crawl-delay` (20 s as of
November 2026, so a season of ~230 games takes about 80 minutes). It caches every page under
`data/html_cache/` and never re-fetches one, backs off on 429/5xx, and records 404s.

## Development

```sh
uv run pytest                    # default suite (~7 s with -n auto)
uv run pytest -m slow            # simulation-recovery tests
uv run pytest -m live            # real API calls; skipped without ANTHROPIC_API_KEY
uv run ruff check . && uv run ruff format --check .
uv run pytest --cov=jtrainer --cov-branch --cov-report=json && uv run python scripts/check_coverage.py
```

`.github/workflows/ci.yml` runs lint, tests and the coverage floors on push, and the slow
suite nightly.

## Notes and decisions

- **Dataset.** The jwolle1 dataset now covers seasons 1-42 (through July 2026), and
  kids/teen matches are also published as a separate file. Ingest reads the combined file;
  special matches (`extra_matches.tsv`) only with `--special`.
- **Eligibility.** Excluded by default: media-dependent clues (Clue Crew, "seen here",
  gesture responses in brackets), and non-standard-difficulty matches per
  `ingest.tournament_patterns`: kids, teen, college, high school, celebrity, Power Players.
  Tournament of Champions and similar adult tournaments stay in.
- **Grading.** Accents are folded for matching ("Dali" matches "Dalí"). Number words,
  ordinals, irregular plurals and initialisms are handled. A fuzzy match only passes at 92+
  and when nothing suggests a different thing (first letter, numerals, roman numerals).
  Partial names, initialisms and number variants go to the judge instead of failing. On
  the 104-case golden set, no case is graded confidently wrong; about half need the judge.
- **Recency in adaptive mode.** `sampling.recency_half_life_years` (default 10) halves a
  clue's weight for every 10 years before the newest clue, in the priority and random
  parts of adaptive sessions. Topic shares are unchanged; within a topic, recent clues come
  up more often. On the full archive, 2021-2026 goes from 16% of draws to 38%, and
  1984-2000 from 38% to 11%. 0 turns it off. Random mode stays uniform, and IPW stats stay
  unbiased because each pick records its exact probability.
- **Review before tagging.** Spaced review works from day one. Until the LLM supplies
  entities, a clue's entity is its normalized response with parentheticals resolved.
- **Difficulty.** Clue difficulty comes from round × board row until `fit-irt` has
  contestant data. Then fitted per-clue difficulties are used by the ability model,
  priorities and samplers.
- **J! Archive fixtures** in `tests/fixtures/jarchive/` are hand-built in the site's
  markup (`make_fixtures.py`). Once you have scraped real pages, copy two or three into the
  fixtures to test against live markup.
- **Judge calls are synchronous.** A borderline answer briefly blocks the UI while the
  model responds.
