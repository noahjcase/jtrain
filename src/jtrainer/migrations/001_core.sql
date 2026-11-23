-- Milestone 1: clues, sessions, attempts, LLM cache.
CREATE SEQUENCE seq_clue_id START 1;
CREATE TABLE clues (
    clue_id          INTEGER PRIMARY KEY DEFAULT nextval('seq_clue_id'),
    source           VARCHAR NOT NULL,              -- 'jwolle1' | 'jarchive'
    source_ref       VARCHAR NOT NULL UNIQUE,       -- content hash / url, for dedup
    source_line      INTEGER,                       -- row order in source (board order)
    air_date         DATE,
    season           INTEGER,
    round            TINYINT NOT NULL CHECK (round BETWEEN 1 AND 3),
    board_row        TINYINT CHECK (board_row BETWEEN 1 AND 5),
    clue_value       INTEGER,
    is_daily_double  BOOLEAN NOT NULL DEFAULT FALSE,
    dd_wager         INTEGER,
    category         VARCHAR,
    category_comment VARCHAR,
    clue_text        VARCHAR NOT NULL,
    response         VARCHAR NOT NULL,
    response_key     VARCHAR,                       -- normalized response (heuristic entity key)
    notes            VARCHAR,
    is_tournament    BOOLEAN NOT NULL DEFAULT FALSE,
    is_special       BOOLEAN NOT NULL DEFAULT FALSE,
    needs_media      BOOLEAN NOT NULL DEFAULT FALSE
);
CREATE INDEX idx_clues_response_key ON clues(response_key);
CREATE INDEX idx_clues_air_date ON clues(air_date);

CREATE SEQUENCE seq_session_id START 1;
CREATE TABLE sessions (
    session_id   INTEGER PRIMARY KEY DEFAULT nextval('seq_session_id'),
    started_at   TIMESTAMP NOT NULL,
    ended_at     TIMESTAMP,
    mode         VARCHAR NOT NULL CHECK (mode IN ('random', 'adaptive', 'review', 'game')),
    app_version  VARCHAR NOT NULL
);

CREATE SEQUENCE seq_attempt_id START 1;
CREATE TABLE attempts (
    attempt_id     INTEGER PRIMARY KEY DEFAULT nextval('seq_attempt_id'),
    session_id     INTEGER NOT NULL REFERENCES sessions(session_id),
    clue_id        INTEGER NOT NULL REFERENCES clues(clue_id),
    shown_at       TIMESTAMP NOT NULL,
    first_key_ms   INTEGER,
    submit_ms      INTEGER,
    user_response  VARCHAR,
    confidence     TINYINT CHECK (confidence BETWEEN 1 AND 3),
    correct        BOOLEAN,                 -- NULL only while needs_review
    grader         VARCHAR CHECK (grader IN ('exact', 'fuzzy', 'llm', 'user')),
    grader_detail  VARCHAR,                 -- JSON
    overridden     BOOLEAN NOT NULL DEFAULT FALSE,
    needs_review   BOOLEAN NOT NULL DEFAULT FALSE,
    sampling_mode  VARCHAR NOT NULL,
    sampling_prob  DOUBLE CHECK (sampling_prob > 0 AND sampling_prob <= 1),
    wager          INTEGER,
    CHECK (first_key_ms IS NULL OR submit_ms IS NULL OR first_key_ms <= submit_ms)
);

CREATE TABLE llm_cache (
    cache_key       VARCHAR PRIMARY KEY,    -- sha256(prompt_version, model, input)
    prompt_version  VARCHAR NOT NULL,
    model           VARCHAR NOT NULL,
    input_hash      VARCHAR NOT NULL,
    response_text   VARCHAR NOT NULL,
    created_at      TIMESTAMP NOT NULL
);
