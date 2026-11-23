-- Milestone 5: contestant outcomes, scrape log, fitted difficulties.
CREATE TABLE clue_outcomes (
    clue_id        INTEGER PRIMARY KEY REFERENCES clues(clue_id),
    n_correct      TINYINT NOT NULL,
    n_incorrect    TINYINT NOT NULL,
    triple_stumper BOOLEAN NOT NULL
);
CREATE TABLE scrape_log (
    url         VARCHAR PRIMARY KEY,
    status      INTEGER NOT NULL,
    fetched_at  TIMESTAMP NOT NULL,
    cache_path  VARCHAR
);
CREATE TABLE clue_difficulty (
    clue_id      INTEGER PRIMARY KEY REFERENCES clues(clue_id),
    model        VARCHAR NOT NULL,
    b_mean       DOUBLE NOT NULL,
    b_sd         DOUBLE NOT NULL,
    fitted_at    TIMESTAMP NOT NULL
);
