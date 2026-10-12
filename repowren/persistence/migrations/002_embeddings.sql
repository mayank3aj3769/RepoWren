CREATE EXTENSION IF NOT EXISTS vector;

-- The earlier milestone stored hashes for files. Semantic retrieval only needs
-- regenerable chunks, so this migration replaces that table with one clear shape.
DROP TABLE IF EXISTS repository_files;

CREATE TABLE repository_chunks (
    id BIGSERIAL PRIMARY KEY,
    repository_id BIGINT NOT NULL REFERENCES repositories(id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    metadata JSONB NOT NULL,
    embedding VECTOR(384) NOT NULL
);
