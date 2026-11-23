-- Milestone 6: game simulator history.
CREATE SEQUENCE seq_game_id START 1;
CREATE TABLE games (
    game_id      INTEGER PRIMARY KEY DEFAULT nextval('seq_game_id'),
    session_id   INTEGER NOT NULL REFERENCES sessions(session_id),
    source       VARCHAR NOT NULL,          -- 'archive:<air_date>' | 'synthetic'
    started_at   TIMESTAMP NOT NULL,
    ended_at     TIMESTAMP,
    final_score  INTEGER,
    coryat       INTEGER
);
CREATE TABLE game_clues (
    game_id     INTEGER NOT NULL REFERENCES games(game_id),
    attempt_id  INTEGER NOT NULL,
    round       TINYINT NOT NULL,
    col         TINYINT,
    board_row   TINYINT,
    value       INTEGER,
    outcome     VARCHAR NOT NULL CHECK (outcome IN ('right', 'wrong', 'pass')),
    score_delta INTEGER NOT NULL
);
