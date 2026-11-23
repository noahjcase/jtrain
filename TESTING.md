# Jeopardy! Trainer — Test Suite Spec

Companion to `DESIGN.md`. Section references (§) point there.

## 1. Principles

1. **Deterministic by construction.** All randomness goes through an
   injected `numpy.random.Generator`; all time goes through an injected
   clock. No test depends on wall-clock time, global RNG state, or the
   network.
2. **No live API or network calls in the default run.** The Anthropic
   client and the HTTP transport are behind interfaces with fakes.
   Live calls exist only under the `live` marker and are skipped unless
   explicitly selected.
3. **Statistical code is tested by simulation.** Models and samplers are
   checked by simulating a user with known parameters and asserting the
   estimates recover them within tolerances derived from the simulation's
   own variance, with fixed seeds.
4. **Every bug gets a regression test.** Every grading override the user
   makes in real use is a candidate golden-set case (§6.3).
5. **Test behavior, not implementation.** Assert on database state,
   returned values, and rendered screens; don't assert on private
   helpers unless they carry real logic (normalization, board-row
   derivation, wager limits).

## 2. Tooling

- `pytest`, `pytest-cov`, `hypothesis` (property tests),
  `pytest-textual-snapshot` (TUI snapshots), `pytest-xdist` (parallel).
- `scipy.stats` for the few distributional checks.
- Run: `uv run pytest` (default, fast), `uv run pytest -m slow`,
  `uv run pytest -m live`.

### Markers

| Marker | Meaning | Default run |
|---|---|---|
| (none) | Unit and integration tests; < 1 s each | yes |
| `slow` | Simulation-recovery tests, full-board games, large fixtures | no |
| `live` | Real Anthropic API calls; needs `ANTHROPIC_API_KEY` | no |
| `snapshot` | TUI snapshot comparisons | yes |

`live` tests must also be skipped (not failed) when the key is absent.
No test is ever marked to hit J! Archive.

### Coverage targets

- `ingest`, `grading`, `model`, `sampling`, game rules: ≥ 90% line,
  with branch coverage on.
- `tagging`, `db`: ≥ 80%.
- `tui`: no target; covered by Pilot and snapshot tests in §10.

Coverage is a floor, not a goal; don't write tests to move the number.

## 3. Layout

```
tests/
  conftest.py
  fixtures/
    clues_sample.tsv          # ~60 hand-built rows, see §4.2
    grading_golden.yaml       # see §6.3
    taxonomy_test.yaml        # 6–8 topics for tagging tests
    tag_responses/            # canned LLM JSON, valid and malformed
    jarchive/                 # 2–3 saved game pages + 1 season page
  unit/
    test_ingest.py
    test_board_row.py
    test_normalize.py
    test_grading.py
    test_tagging.py
    test_entities.py
    test_ability.py
    test_priority.py
    test_review.py
    test_sampling.py
    test_game_rules.py
    test_jarchive_parser.py
    test_scrape_client.py
  integration/
    test_migrations.py
    test_session_flow.py
    test_retroactive_tags.py
    test_ipw.py
  sim/                        # all marked slow
    test_ability_recovery.py
    test_irt_recovery.py
    test_adaptive_convergence.py
  tui/
    test_quiz_screen.py
    test_review_screen.py
    test_stats_screen.py
    test_tag_review_screen.py
    __snapshots__/
  live/
    test_live_tagging.py
    test_live_judge.py
```

## 4. Shared fixtures (`conftest.py`)

### 4.1 Infrastructure

- `db` — fresh in-memory DuckDB with all migrations applied. Function
  scope.
- `db_loaded` — `db` plus `clues_sample.tsv` ingested and a test
  taxonomy with tags. Function scope; built from a session-scoped
  template by copying, so it stays fast.
- `rng` — `numpy.random.default_rng(12345)`. Tests needing independent
  streams use `rng.spawn(n)`.
- `clock` — a `FakeClock` with `now()` and `advance(ms)`; starts at
  2026-01-01T00:00:00Z.
- `fake_llm` — implements the same interface as the real client.
  Returns canned responses keyed by `(prompt_version, input_hash)`,
  records every call, and raises if called for an unregistered input.
  Exposes `.calls` for asserting call counts.
- `fake_http` — an `httpx.MockTransport` serving files from
  `fixtures/jarchive/`, recording requests and their timestamps on the
  fake clock.
- `config` — a test config object with thresholds and sampling mix set
  explicitly, so tests don't silently change when `config.toml` does.

### 4.2 `clues_sample.tsv`

Hand-built in the jwolle1 column format. Each row exists to exercise a
case. Required rows include:

- Single, Double, and Final Jeopardy clues before and after 2001-11-26,
  including clues aired **on** 2001-11-26.
- Every board value in both value eras, per round.
- A Daily Double in each round and era (`daily_double_value` > 0).
- A clue with a value outside the valid set for its round and era (data
  error).
- A clue with a missing value.
- Tournament clues (kids, teen, college) identified via `notes`.
- Media-dependent clues: "Clue Crew", "seen here", "heard here",
  "shown here", plus a near-miss that should **not** be flagged (e.g. a
  clue mentioning a "crew" in an unrelated sense).
- Responses with parentheticals: `(Abraham) Lincoln`,
  `the Louvre (Museum)`, `Mars (or the Red Planet)`.
- Responses with HTML entities, escaped quotes, accented characters,
  and trailing whitespace.
- Two clues with the same response entity worded differently (for
  alternate-clue review tests).
- An entity appearing only once (review fallback).
- A duplicated row (dedup test).

Document each row's purpose in a comment line or a sidecar
`clues_sample.md`.

## 5. Ingest (`test_ingest.py`, `test_board_row.py`)

Column mapping:

- TSV `answer` lands in `clue_text`; TSV `question` lands in `response`.
  Assert on a specific known row, not just that columns exist.
- `daily_double_value` > 0 sets `is_daily_double` and `dd_wager`.

Board row (pure function; table-driven and property tests):

- Before 2001-11-26: J values 100–500 → rows 1–5; DJ 200–1000 →
  rows 1–5.
- On and after 2001-11-26: J 200–1000 → 1–5; DJ 400–2000 → 1–5.
- The same value maps to different rows across eras (J $200 is row 2
  before, row 1 after).
- Final Jeopardy → `NULL`.
- Daily Double uses `clue_value` (board value), never `dd_wager`.
- Off-grid or missing value → `NULL`, counted in the ingest report, not
  an exception.
- Property: for every valid (round, era, value), the output is in 1–5.

Flags:

- Tournament rows flagged; special-match file skipped by default.
- `needs_media` flagged for each pattern; the near-miss row is not.

Cleaning:

- HTML entities decoded; escaped quotes unescaped; whitespace stripped;
  Unicode preserved (no ASCII folding in stored text).

Idempotence and integrity:

- Ingesting the same file twice leaves the row count unchanged
  (`source_ref` dedup).
- The duplicated row produces one clue.
- The ingest report's counts (read, inserted, skipped by reason) sum
  to the rows read.

## 6. Grading (`test_normalize.py`, `test_grading.py`)

### 6.1 Normalization (§7 step 1)

Table-driven cases, plus Hypothesis properties:

- Case-folded; punctuation stripped; whitespace collapsed.
- Prefixes stripped: "what is", "who are", "what's", "where is", etc.,
  with and without a trailing "?".
- Leading articles stripped: "the", "a", "an".
- Parentheticals in the **recorded** response become optional: for
  `(Abraham) Lincoln`, both "Lincoln" and "Abraham Lincoln" normalize to
  a match.
- "or" alternatives in parentheticals (`Mars (or the Red Planet)`) each
  accepted.
- Properties:
  - `normalize(normalize(x)) == normalize(x)`.
  - `normalize("what is " + x) == normalize(x)` for any `x`.
  - Never raises on arbitrary Unicode.

Accent handling is a decision to make explicitly and pin with a test:
"Dali" should match "Dalí".

### 6.2 Pipeline (§7 steps 2–5)

- Exact match → `grader='exact'`, correct, no fuzzy or LLM call.
- Alias match via `entities.aliases` → correct.
- Score above the high threshold → `grader='fuzzy'`, correct.
- Score below the low threshold → `grader='fuzzy'`, incorrect, no LLM
  call.
- Score in the band → `fake_llm` called exactly once; verdict and
  rationale stored in `grader_detail`.
- Repeat of the same borderline input → served from cache;
  `fake_llm.calls` unchanged.
- Malformed LLM output → one retry, then falls back to marking the
  attempt `needs_review` for the user, never a silent default verdict.
- Blank and explicit pass → incorrect, `grader='exact'`, no LLM call.
- Override:
  - Sets `correct`, `overridden=True`, `grader='user'`.
  - Preserves the original grader's verdict in `grader_detail`.
  - Double override (flip back) records the final state correctly.
- Thresholds come from config; a test sets them explicitly and checks
  the band moves.

### 6.3 Golden set (`grading_golden.yaml`)

Format per case: `clue_text`, `response`, `user_response`, `expected`,
`note`. Seed with about 100 cases covering:

- Spelling variants (Dostoevsky / Dostoyevsky, Tchaikovsky /
  Chaikovsky).
- Surname-only for people normally accepted by surname.
- Too-vague responses that must fail (a first name alone when the
  surname is required; "a president" for a specific one).
- Plural/singular, numerals vs. words ("1984" / "Nineteen
  Eighty-Four"), abbreviations ("JFK", "U.S.").
- Near misses that fuzzy matching would wrongly accept (e.g. "Austria"
  for "Australia"; "Johnson" for "Jackson").

Two tests consume it:

- `test_golden_no_llm`: steps 1–3 only. Asserts no case is graded
  **confidently wrong**: every case either matches `expected` or falls
  into the LLM band. Run by default.
- `test_golden_live` (`live`): full pipeline with the real model.
  Reports accuracy; fails below a configured floor (start at 95%).

A small script appends the user's real overrides to a candidate file
for review before they join the golden set.

## 7. Tagging and entities (`test_tagging.py`, `test_entities.py`)

- Taxonomy YAML loads; duplicate `topic_id`s and missing descriptions
  are rejected with a clear error.
- Prompt files carry a version string; the loader refuses a prompt
  without one.
- Tagging a clue inserts a `clue_tags` row with `taxonomy_version`,
  `prompt_version`, and `model` filled.
- Re-tagging under a new taxonomy version **inserts** new rows; old
  rows are unchanged. Assert row counts and old values.
- Queries return tags for the configured version only.
- A response with a `topic_id` not in the taxonomy is rejected and
  logged, not stored.
- Malformed JSON from the batch result: retried once, then recorded as
  a failure with the raw text; the batch continues.
- Cache: tagging the same clue with the same prompt version makes no
  second call.
- Batch result parsing: maps results back to `clue_id` by custom id;
  handles partial batches (some errored, some succeeded).
- Entities:
  - "(Abraham) Lincoln" and "Abraham Lincoln" resolve to one entity.
  - A new alias for an existing entity is added, not duplicated.
  - Distinct entities with similar names stay distinct ("John Adams" /
    "John Quincy Adams").

## 8. Models

### 8.1 Ability, v1 (`test_ability.py`)

Unit properties of the update rule:

- A correct response raises the topic rating; an incorrect one lowers
  it.
- The update is larger for a surprising result (correct on a hard clue)
  than an expected one.
- Uncertainty shrinks after each update and never goes negative.
- Updates to one topic leave other topics unchanged.
- Difficulty prior is monotone in board row within a round, and Double
  Jeopardy row 1 is harder than Jeopardy row 1.
- Attempts with no tag under the current taxonomy are skipped and
  counted, not errors.

Recovery simulation (`sim/test_ability_recovery.py`, `slow`):

- Simulate a user with known per-topic abilities across 10 topics,
  answering 3,000 random-mode clues by the model's own response
  function.
- Assert the rank correlation between true and estimated ability is
  ≥ 0.9, and each estimate is within its reported uncertainty
  interval for at least the nominal coverage minus a tolerance (e.g.
  ≥ 85% of topics inside 90% intervals).
- Rerun the estimate from logged attempts alone (no in-memory state)
  and assert identical results — the model must be reproducible from
  the `attempts` table.

### 8.2 IRT, v2 (`sim/test_irt_recovery.py`, `slow`)

- Simulate clue difficulties and contestant outcomes; fit; assert
  difficulty recovery (correlation ≥ 0.9) and that the hierarchical
  prior shrinks low-data topics toward the global mean.
- Censoring: simulated "no one buzzed" outcomes bias naive estimates;
  assert the fitted model's bias is smaller than the naive estimator's.

### 8.3 Priority (`test_priority.py`)

- `priority = frequency × (1 − p̂)` on hand-computed inputs.
- A frequent topic with modest weakness outranks a rare topic with
  severe weakness when the numbers say so.
- Recency weighting: with decay on, a topic common only in old seasons
  loses priority; with decay off, it doesn't.
- Frequencies sum to 1 over the taxonomy.

### 8.4 Spaced review (`test_review.py`)

- An entity enters the queue on: a miss; a correct response with
  confidence 1; a correct response slower than the configured
  threshold. It does not enter on a fast, confident correct response.
- Due items are returned in due order; items not yet due are not.
- Review presents a different clue for the entity when one exists and
  avoids the most recently shown clue.
- Entity with a single clue: falls back to that clue.
- FSRS state round-trips through the JSON column unchanged.
- Advancing the fake clock past `due_at` makes the item due.

## 9. Sampling (`test_sampling.py`, `integration/test_ipw.py`)

- Eligibility: `needs_media`, tournament, and special-match clues are
  never served by default; a config flag includes them.
- Random mode: over 20,000 draws from a 50-clue eligible pool, a
  chi-square goodness-of-fit test against uniform does not reject at
  α = 0.001 (fixed seed, so this is deterministic, not flaky).
- Random mode records `sampling_prob = 1/N` for an eligible pool of N.
- Adaptive mix: over 10,000 draws, the shares of priority, review, and
  random picks are within ±2 percentage points of config.
- Adaptive mode with no due reviews redistributes the review share per
  config, and `sampling_prob` reflects the redistribution.
- `sampling_prob` is the exact probability the sampler assigned: for a
  small pool, compute it by brute-force enumeration and compare.
- No clue is repeated within a session unless the eligible pool is
  exhausted.
- IPW (`test_ipw.py`): simulate a user, run adaptive sampling, then
  assert the inverse-probability-weighted per-topic accuracy is within
  tolerance of the true accuracy on a uniform board, while the naive
  unweighted accuracy is measurably biased. This test is the reason
  `sampling_prob` exists; it must not be skipped.

## 10. TUI (`tests/tui/`)

Use Textual's `App.run_test()` with `Pilot`, the fake clock, `rng`,
`fake_llm`, and `db_loaded`.

Quiz screen:

- Starting a random session shows a clue with category, round, and
  value.
- Typing a response, pressing a confidence key (1–3), and submitting
  writes one `attempts` row with all fields populated.
- Timing: advance the fake clock 1,200 ms before the first keystroke
  and 4,000 ms before submit; assert `first_key_ms = 1200`,
  `submit_ms = 5200`. Assert `first_key_ms ≤ submit_ms` always.
- Submitting without a confidence key prompts for one; no row written.
- Verdict screen shows the recorded response; the override key flips
  the verdict and updates the row.
- Pass key records an incorrect attempt without grading.
- Quitting mid-session closes the session with `ended_at` set and
  loses no submitted attempts.

Review screen: shows only due items; empty queue shows an empty state,
not an error.

Stats screen: renders with zero attempts, with one attempt, and with
`db_loaded` plus simulated history; values match the model functions
called directly.

Tag review screen (milestone 2): accept, change topic, and flag
actions each write the expected rows.

Snapshots: quiz, verdict, stats, and review screens at a fixed
terminal size (100×30) and fixed data. Snapshot changes are reviewed
in the diff, never blindly regenerated.

## 11. Database and integration

`test_migrations.py`:

- All migrations apply cleanly to an empty database.
- Applying them again is a no-op.
- A database at each historical schema version upgrades to current
  (keep a small fixture database per released version).
- Constraints hold: `confidence` in 1–3; `round` in 1–3; `board_row`
  in 1–5 or NULL; foreign keys valid.

Append-only attempts:

- The data-access layer exposes no delete for `attempts`; a test
  asserts the repository class has no such method and that overrides
  update in place with `overridden=True`.

`test_session_flow.py`: an end-to-end session through the service
layer (no TUI): 20 attempts across modes, then assert table state,
ability updates, and review queue contents.

`test_retroactive_tags.py`: log attempts before any tags exist, then
tag, then assert per-topic stats include those attempts. This guards
the design decision that lets quizzing precede tagging.

## 12. Scrape (milestone 5)

Parser (`test_jarchive_parser.py`), against saved pages in
`fixtures/jarchive/`:

- Extracts all 61 clues from a complete game (30 + 30 + Final), or the
  correct smaller count for a game with unrevealed clues.
- Extracts round, value, Daily Double status and wager, category,
  clue text, correct response, and per-clue correct/incorrect
  contestants.
- Identifies triple stumpers.
- Handles a game page with a tiebreaker clue.
- Season page parsing yields game ids and air dates.

Client (`test_scrape_client.py`), with `fake_http` and the fake clock:

- Requests are spaced at least the configured interval apart.
- A cached page is never re-fetched; the transport sees zero requests.
- Disallowed paths per a fixture `robots.txt` are never requested.
- 429 and 5xx responses back off and retry a bounded number of times;
  404 is recorded and not retried.
- Merging scraped clues with jwolle1 clues deduplicates on air date,
  round, category, and clue text.

## 13. Game simulator (`test_game_rules.py`, milestone 6)

- Board: 6 categories × 5 clues in each of Jeopardy and Double
  Jeopardy; one Daily Double in Jeopardy, two in Double Jeopardy; one
  Final clue.
- Daily Double wager limits: minimum $5; maximum is the greater of the
  player's score and the round's top board value ($1,000 / $2,000
  post-2001; $500 / $1,000 before). Out-of-range wagers rejected.
- Final Jeopardy: wager from $0 to current score; a player at $0 or
  below does not play Final.
- Scoring: correct adds the value or wager; incorrect subtracts it.
- Coryat: counts each regular clue at board value (plus if right,
  minus if wrong); Daily Doubles at board value if right and zero if
  wrong; excludes Final. Check against a hand-scored fixture game.
- Replaying an archived game serves its clues in board positions as
  aired.

## 14. Live tests (`tests/live/`)

Marked `live`; skipped without the API key. Kept few and cheap:

- Tag 10 fixed clues; assert valid JSON, valid `topic_id`s, and
  agreement with hand labels on at least 8.
- Judge 10 borderline golden cases; assert agreement on at least 9.
- The golden-set live run (§6.3).

Never run in CI by default; run manually before changing the model
name or a prompt version.

## 15. CI

On every push: `ruff check`, `ruff format --check`, `pytest` (default
markers) with coverage, failing under the §2 floors.

Nightly or on demand: `pytest -m slow`.

Manual: `pytest -m live`.

## 16. Out of scope for tests

- Textual's own widget behavior.
- DuckDB, FSRS, and rapidfuzz internals.
- Visual styling beyond snapshots.
- Real J! Archive availability or page changes (caught by the parser
  failing on new pages, not by tests).
