-- Milestone 4: FSRS state per entity (a cache derivable from attempts).
CREATE TABLE review_state (
    entity_id        INTEGER PRIMARY KEY REFERENCES entities(entity_id),
    fsrs_card        JSON NOT NULL,
    due_at           TIMESTAMP NOT NULL,
    last_attempt_id  INTEGER,
    last_clue_id     INTEGER
);
CREATE INDEX idx_review_due ON review_state(due_at);
