# Jeopardy! Trainer — Design

A terminal app for preparing to compete on *Jeopardy!*. It serves real
archive clues, grades responses, logs every attempt, and steers study
toward the topics where improvement is worth the most.

## 1. Principles

These drive every design decision below.

1. **The unit of learning is the fact, not the clue.** The show draws on a
   stable canon (presidents, capitals, Shakespeare, opera, the Bible, Best
   Pictures, elements, novels and authors). The same correct responses
   recur under different wording. Review schedules track the underlying
   answer entity, and each review presents it through a different clue
   when one exists, so the user learns the fact rather than the wording.
2. **Study what pays.** Priority is how often a topic appears times the
   user's chance of missing it, not weakness alone.
3. **Retrieval over reading.** The app is a quiz engine first.
4. **Log everything, decide later.** Every attempt is stored with enough
   context (sampling mode, sampling probability, timing, confidence) that
   later models can be fit to historical data without re-collecting it.
5. **Usable at every milestone.** Each milestone in §10 produces a tool
   the user can study with daily. Avoid building ahead of need.

## 2. Scope

In scope: single user, local, offline-capable TUI; real clues; grading;
attempt logging; topic tagging; adaptive sampling; spaced review; stats;
eventually a full-game simulator with wagering.

Not in scope: buzzer-timing practice, multiplayer, mobile or web UI,
hosting, accounts.

## 3. Stack

- Python 3.12+, managed with `uv`; `pyproject.toml`; `ruff`; `pytest`.
- **DuckDB** as the single database file (`data/jtrainer.duckdb`).
- **Textual** for the TUI.
- **Anthropic API** for tagging, entity normalization, and borderline
  grading. Model name is configuration, not code. Use the Message Batches
  API for bulk jobs. Cache every LLM result in the database; never call
  the API twice for the same input and prompt version.
- `rapidfuzz` for fuzzy matching.
- `fsrs` (py-fsrs) for spaced-repetition scheduling.
- `httpx` + `selectolax` (or BeautifulSoup) for the later scrape.

`data/` is git-ignored. The API key comes from the environment.

## 4. Data sources

### 4.1 Base dataset (milestone 1)

jwolle1's `jeopardy_clue_dataset` on GitHub: about 473,000 clues from
Season 1 through July 2023, as TSV. Columns: `round` (1 = Jeopardy,
2 = Double Jeopardy, 3 = Final), `clue_value`, `daily_double_value`,
`category`, `comments`, `answer`, `question`, `air_date`, `notes`.

Ingestion notes:

- **The column names are inverted relative to app terminology.** In the
  TSV, `answer` is the clue text read by the host and `question` is the
  correct response. Map them to `clue_text` and `response` on ingest.
- The dataset omits some clues (e.g. unrevealed ones).
- Kids, teen, and college tournament clues are included; flag them via
  `notes` and exclude from default sampling.
- Special-match clues live in `extra_matches.tsv`; skip by default.
- **Board values doubled on 2001-11-26.** Do not use dollar value as a
  difficulty feature directly. Derive `board_row` (1–5) from value, round,
  and air date, and use that.
- Flag clues that depend on media the user cannot see (Clue Crew clues,
  "seen here", "heard here", etc.) as `needs_media` and exclude them by
  default. A simple pattern match is fine to start; refine later.

### 4.2 J! Archive scrape (milestone 5)

J! Archive game pages record which contestants responded correctly or
incorrectly to each clue and which clues were triple stumpers. That is
the best available signal of real difficulty and is absent from the base
dataset. The scrape also fills in seasons after July 2023.

Rules: read the site's terms and `robots.txt` first; rate-limit to at
most one request every few seconds; cache raw HTML to disk and parse
offline; never re-fetch a cached page. It is a volunteer-run site.

## 5. Data model

All tables in DuckDB. Sketch, not final DDL.

```
clues
  clue_id          INTEGER PK   -- stable internal id
  source           VARCHAR      -- 'jwolle1' | 'jarchive'
  source_ref       VARCHAR      -- id/url in source, for dedup
  air_date         DATE
  season           INTEGER      -- nullable until known
  round            TINYINT      -- 1, 2, 3
  board_row        TINYINT      -- 1–5; NULL for Final
  clue_value       INTEGER      -- as aired
  is_daily_double  BOOLEAN
  dd_wager         INTEGER
  category         VARCHAR
  category_comment VARCHAR
  clue_text        VARCHAR
  response         VARCHAR      -- as recorded, incl. parentheticals
  is_tournament    BOOLEAN
  needs_media      BOOLEAN

clue_outcomes                    -- milestone 5
  clue_id          FK
  n_correct        TINYINT
  n_incorrect      TINYINT
  triple_stumper   BOOLEAN

taxonomies
  taxonomy_version VARCHAR PK
  created_at       TIMESTAMP
  notes            VARCHAR

topics
  taxonomy_version FK
  topic_id         VARCHAR      -- e.g. 'us_presidents'
  label            VARCHAR
  description      VARCHAR      -- used verbatim in the tagging prompt

clue_tags                        -- many rows per clue over time
  clue_id          FK
  taxonomy_version FK
  prompt_version   VARCHAR
  model            VARCHAR
  topic_id         VARCHAR
  secondary_topic  VARCHAR      -- nullable
  entity_id        FK           -- normalized answer entity
  tagged_at        TIMESTAMP

entities
  entity_id        INTEGER PK
  canonical_name   VARCHAR
  aliases          VARCHAR[]
  wikidata_qid     VARCHAR      -- optional, later

sessions
  session_id       INTEGER PK
  started_at       TIMESTAMP
  ended_at         TIMESTAMP
  mode             VARCHAR      -- 'random' | 'adaptive' | 'review' | 'game'
  app_version      VARCHAR

attempts
  attempt_id       INTEGER PK
  session_id       FK
  clue_id          FK
  shown_at         TIMESTAMP
  first_key_ms     INTEGER      -- clue shown → first keystroke
  submit_ms        INTEGER      -- clue shown → submit
  user_response    VARCHAR
  confidence       TINYINT      -- 1 guess, 2 maybe, 3 sure; before reveal
  correct          BOOLEAN      -- final verdict
  grader           VARCHAR      -- 'exact' | 'fuzzy' | 'llm' | 'user'
  grader_detail    VARCHAR      -- score, LLM rationale, etc.
  overridden       BOOLEAN
  sampling_mode    VARCHAR      -- which sampler chose this clue
  sampling_prob    DOUBLE       -- probability the sampler assigned it
  wager            INTEGER      -- DD/Final in game mode; else NULL

review_state                     -- FSRS state per entity
  entity_id        FK PK
  fsrs_card        JSON
  due_at           TIMESTAMP
  last_attempt_id  FK
```

Key decisions:

- **Tags are versioned rows, never columns on `clues`.** The taxonomy
  will be revised at least once. Re-tagging inserts rows under a new
  version; queries pick a version via config. Attempts join to tags by
  `clue_id`, so attempts made before tagging get topics retroactively.
- **`sampling_prob` is recorded on every attempt.** Once sampling is
  adaptive, the user's attempts are not a random sample of the board.
  Storing the selection probability allows inverse-probability-weighted
  estimates of true per-topic accuracy.
- Never delete attempts. Corrections are overrides, recorded as such.

## 6. Topic taxonomy and tagging

*Jeopardy!* category titles are puns ("2 VERBS IN ONE"), not subjects, so
every clue is mapped to a fixed taxonomy of roughly 40–60 topics (e.g.
`us_presidents`, `world_capitals`, `opera`, `bible`, `wordplay_anagrams`).

Process:

1. Draft the taxonomy by hand as a YAML file (id, label, one-line
   description, a few example clues).
2. Tag a random sample of about 500 clues. Review by hand in a small TUI
   screen: accept, change topic, or flag the taxonomy as missing a topic.
3. Revise the taxonomy; bump `taxonomy_version`.
4. Batch-tag the full corpus with the Message Batches API.

The tagging call returns, as structured JSON: `topic_id`,
optional `secondary_topic`, and a normalized answer entity (canonical
name with J! Archive parentheticals resolved, e.g. "(Abraham) Lincoln"
→ "Abraham Lincoln"). Entities are deduplicated into `entities` by
canonical name plus aliases.

Wordplay categories (rhyme time, before & after, anagrams, "crossword
clues") are tagged by their wordplay type, since the skill is the
mechanic, not the subject.

This work runs in parallel with milestone 1; it does not block quizzing.

## 7. Grading

Pipeline, stopping at the first decisive step:

1. **Normalize** both strings: lowercase, strip punctuation, leading
   articles, "what is"/"who is" prefixes, and the response's
   parenthetical portions (treated as optional).
2. **Exact match** against the normalized response and entity aliases →
   correct.
3. **Fuzzy match** (`rapidfuzz` token ratio). Above a high threshold →
   correct; below a low threshold → incorrect.
4. **LLM judge** for the band in between, given the clue, the recorded
   response, and the user's response, asked to apply *Jeopardy!*
   acceptability standards. Return verdict and one-line rationale.
5. **User override** is always one key away after the verdict is shown.
   Overrides are logged and are the ground truth for tuning thresholds.

Blank responses and explicit passes are incorrect without grading.

## 8. Models

### 8.1 Topic ability

Goal: an estimate, per topic, of the probability the user responds
correctly to a clue of given difficulty, with uncertainty.

- **Version 1: online Elo/Glicko-style update.** Each topic has a rating
  and uncertainty; each clue has a difficulty. Difficulty prior comes from
  `round` × `board_row` (Final Jeopardy treated as its own level). After
  each attempt, update the topic rating. Simple, incremental, adequate to
  drive sampling.
- **Version 2 (after milestone 5): offline Rasch / 1PL IRT** with
  hierarchical pooling, P(correct) = logistic(θ_topic − b_clue), where
  θ_topic ~ Normal(θ_global, σ²). Clue difficulty b is calibrated from
  contestant outcomes (`clue_outcomes`), with `round` × `board_row` as
  the prior. Caveat: contestant data is censored; "no one answered" mixes
  "no one knew" with "no one buzzed". Treat it as a noisy signal.

### 8.2 Topic priority

```
priority_topic = frequency_topic × (1 − p̂_topic)
```

- `frequency_topic` is the topic's share of clues, weighted toward recent
  seasons (e.g. exponential decay by air date), since writers' tastes
  drift. Exact weighting is a config parameter.
- `p̂_topic` is the predicted accuracy at the difficulty the user will
  face on the show (mid-board Double Jeopardy is a reasonable target).

### 8.3 Spaced review

FSRS schedules review per **entity**, not per clue. An entity enters the
review queue the first time the user misses a clue whose response is
that entity, or answers it correctly but slowly or with low confidence.
At review time, present a different clue for the same entity when one
exists.

## 9. Sampling and app modes

Each attempt records which sampler chose the clue and its probability.

- **Random** — uniform over eligible clues. Default in milestone 1. Gives
  an unbiased baseline.
- **Adaptive** — a mix, configurable, initially:
  - 60% weighted by topic priority, then by difficulty near the user's
    ability in that topic;
  - 20% due FSRS reviews;
  - 20% uniform random, to keep estimates honest and detect drift.
- **Review** — due FSRS items only.
- **Game** (milestone 6) — a full archived game or a synthesized board:
  six categories × five clues per round, Daily Doubles, Final Jeopardy,
  wagering, and Coryat score (score ignoring wagers).

Eligibility filters by default: not `needs_media`, not tournament, not
special matches.

## 10. Milestones

Each is usable on its own. Acceptance criteria in italics.

1. **Ingest and random quiz.** Load the TSV into DuckDB with derived
   fields. TUI shows a clue, takes a typed response and a 1–3 confidence
   key, grades with steps 1–3 and 5 of §7, logs the attempt with timings.
   *The user can do a 50-clue random session and see accuracy afterward.*
2. **Taxonomy and tagging** (parallel with 1). Taxonomy YAML, sample
   tagging, review screen, full batch, entity table.
   *Every eligible clue has a tag under the current taxonomy; the
   500-clue hand review shows acceptable accuracy.*
3. **Topic ability and adaptive sampling.** §8.1 v1, §8.2, adaptive mode,
   stats screen (per-topic accuracy, IPW-corrected; ability with
   uncertainty; priority ranking; response-time distribution; confidence
   calibration by bin with Brier score). LLM judge (§7 step 4) added.
   *Adaptive mode visibly shifts toward high-priority topics.*
4. **Spaced review.** Entity-level FSRS, review mode, alternate-clue
   presentation. *Missed entities reappear on schedule via new clues.*
5. **Scrape and difficulty calibration.** Polite J! Archive scrape for
   contestant outcomes and post-2023 games; §8.1 v2.
   *Clue difficulties fitted; post-2023 clues available.*
6. **Game simulator.** Full boards, wagering, Coryat, game history.
   *The user can play a full archived game and get a Coryat score.*

## 11. Layout

```
pyproject.toml
DESIGN.md
TESTING.md                  # test suite spec
config.toml                 # model names, thresholds, sampling mix
taxonomy/
  v1.yaml
src/jtrainer/
  db.py                     # connection, migrations
  ingest/                   # jwolle1 TSV, later jarchive parser
  grading.py
  tagging/                  # prompts (versioned), batch runner
  model/                    # ability, priority, fsrs wrapper
  sampling.py
  tui/                      # Textual app and screens
tests/
data/                       # git-ignored: duckdb file, raw TSV, html cache
```

Schema changes go through numbered migration scripts. Prompts are files
with explicit version strings, not inline literals.

## 12. Open questions

- Final taxonomy size and granularity (expect revision after the sample).
- Confidence scale: 1–3 keys vs. a percentage. Three keys is faster; start
  there.
- Recency weighting for topic frequency: decay rate, or a hard window.
- Whether response time should feed the "correct" signal for review
  scheduling (a slow correct response may count as a lapse).
