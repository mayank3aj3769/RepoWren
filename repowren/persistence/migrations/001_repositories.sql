CREATE TABLE repositories (
    id BIGSERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    root_path TEXT NOT NULL UNIQUE,
    is_active BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE UNIQUE INDEX one_active_repository
    ON repositories (is_active)
    WHERE is_active;

CREATE TABLE repository_files (
    repository_id BIGINT NOT NULL REFERENCES repositories(id) ON DELETE CASCADE,
    path TEXT NOT NULL,
    size_bytes BIGINT NOT NULL,
    modified_ns BIGINT NOT NULL,
    sha256 TEXT NOT NULL,
    PRIMARY KEY (repository_id, path)
);
