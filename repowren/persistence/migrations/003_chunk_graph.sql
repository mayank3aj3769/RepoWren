-- Keep graph identity deterministic across complete repository rebuilds.
ALTER TABLE repository_chunks ADD COLUMN chunk_key TEXT;

UPDATE repository_chunks
SET chunk_key = repository_id::TEXT || ':' || (metadata->>'path') || ':' ||
    (metadata->>'start_line') || ':' || (metadata->>'end_line');

ALTER TABLE repository_chunks ALTER COLUMN chunk_key SET NOT NULL;

CREATE UNIQUE INDEX repository_chunks_repository_key
    ON repository_chunks (repository_id, chunk_key);

-- Edges contain no source text or embeddings. Both endpoints must belong to
-- the same registered repository and disappear with its rebuilt chunks.
CREATE TABLE repository_chunk_edges (
    repository_id BIGINT NOT NULL REFERENCES repositories(id) ON DELETE CASCADE,
    source_chunk_key TEXT NOT NULL,
    target_chunk_key TEXT NOT NULL,
    relation TEXT NOT NULL CHECK (relation IN ('adjacent', 'calls', 'imports', 'tests')),
    detail TEXT,
    PRIMARY KEY (repository_id, source_chunk_key, target_chunk_key, relation),
    FOREIGN KEY (repository_id, source_chunk_key)
        REFERENCES repository_chunks(repository_id, chunk_key) ON DELETE CASCADE,
    FOREIGN KEY (repository_id, target_chunk_key)
        REFERENCES repository_chunks(repository_id, chunk_key) ON DELETE CASCADE,
    CHECK (source_chunk_key <> target_chunk_key)
);

CREATE INDEX repository_chunk_edges_target
    ON repository_chunk_edges (repository_id, target_chunk_key);
