-- Entity aliases move to their own table. DuckDB executes an UPDATE of a list column as
-- delete+insert, which a foreign key into `entities` (clue_tags, review_state) rejects,
-- so aliases could not be added to an entity once anything referenced it.
--
-- `entities.aliases` cannot be dropped while those foreign keys exist; it is left in
-- place, no longer read or written.
CREATE TABLE entity_aliases (
    entity_id INTEGER NOT NULL REFERENCES entities(entity_id),
    position  INTEGER NOT NULL,   -- insertion order
    alias     VARCHAR NOT NULL,
    PRIMARY KEY (entity_id, position)
);

INSERT INTO entity_aliases
SELECT entity_id, generate_subscripts(aliases, 1) - 1, unnest(aliases)
FROM entities
WHERE len(aliases) > 0;
