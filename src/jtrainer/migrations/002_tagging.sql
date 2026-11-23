-- Milestone 2: taxonomy, versioned tags, entities.
CREATE TABLE taxonomies (
    taxonomy_version VARCHAR PRIMARY KEY,
    created_at       TIMESTAMP NOT NULL,
    notes            VARCHAR
);
CREATE TABLE topics (
    taxonomy_version VARCHAR NOT NULL REFERENCES taxonomies(taxonomy_version),
    topic_id         VARCHAR NOT NULL,
    label            VARCHAR NOT NULL,
    description      VARCHAR NOT NULL,
    PRIMARY KEY (taxonomy_version, topic_id)
);

CREATE SEQUENCE seq_entity_id START 1;
CREATE TABLE entities (
    entity_id      INTEGER PRIMARY KEY DEFAULT nextval('seq_entity_id'),
    canonical_name VARCHAR NOT NULL,
    canonical_key  VARCHAR NOT NULL UNIQUE,  -- normalized canonical name
    aliases        VARCHAR[] NOT NULL DEFAULT [],
    source         VARCHAR NOT NULL DEFAULT 'llm',   -- 'llm' | 'heuristic'
    wikidata_qid   VARCHAR
);

CREATE TABLE clue_tags (
    clue_id          INTEGER NOT NULL REFERENCES clues(clue_id),
    taxonomy_version VARCHAR NOT NULL REFERENCES taxonomies(taxonomy_version),
    prompt_version   VARCHAR NOT NULL,
    model            VARCHAR NOT NULL,      -- 'human' for hand-review corrections
    topic_id         VARCHAR NOT NULL,
    secondary_topic  VARCHAR,
    entity_id        INTEGER REFERENCES entities(entity_id),
    tagged_at        TIMESTAMP NOT NULL
);
CREATE INDEX idx_clue_tags_clue ON clue_tags(clue_id, taxonomy_version);

CREATE TABLE tag_failures (
    clue_id          INTEGER NOT NULL,
    taxonomy_version VARCHAR NOT NULL,
    prompt_version   VARCHAR NOT NULL,
    model            VARCHAR NOT NULL,
    reason           VARCHAR NOT NULL,
    raw_text         VARCHAR,
    failed_at        TIMESTAMP NOT NULL
);

CREATE TABLE tag_reviews (
    clue_id          INTEGER NOT NULL,
    taxonomy_version VARCHAR NOT NULL,
    action           VARCHAR NOT NULL CHECK (action IN ('accept', 'change', 'flag')),
    original_topic   VARCHAR,
    new_topic        VARCHAR,
    note             VARCHAR,
    reviewed_at      TIMESTAMP NOT NULL
);

CREATE TABLE tag_batches (
    batch_id         VARCHAR PRIMARY KEY,
    taxonomy_version VARCHAR NOT NULL,
    prompt_version   VARCHAR NOT NULL,
    model            VARCHAR NOT NULL,
    n_requests       INTEGER NOT NULL,
    submitted_at     TIMESTAMP NOT NULL,
    collected_at     TIMESTAMP
);
