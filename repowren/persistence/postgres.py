"""Small psycopg data-access layer for repository metadata."""

from __future__ import annotations

from collections.abc import Sequence
from importlib import resources

import psycopg
from psycopg.rows import dict_row

from repowren.repositories.models import FileMetadata, RepositoryRecord
from repowren.repositories.store import RepositoryStoreError


class PostgresRepositoryStore:
    """Persist registered repositories without adding a full ORM."""

    def __init__(self, database_url: str) -> None:
        self._database_url = database_url

    async def initialize(self) -> None:
        try:
            async with await self._connect() as connection:
                await connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS schema_migrations (
                        version TEXT PRIMARY KEY,
                        applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    )
                    """
                )
                migration_root = resources.files(
                    "repowren.persistence.migrations"
                )
                for migration in sorted(
                    migration_root.iterdir(), key=lambda item: item.name
                ):
                    if migration.suffix != ".sql":
                        continue
                    cursor = await connection.execute(
                        "SELECT 1 FROM schema_migrations WHERE version = %s",
                        (migration.name,),
                    )
                    if await cursor.fetchone():
                        continue
                    await connection.execute(migration.read_text(encoding="utf-8"))
                    await connection.execute(
                        "INSERT INTO schema_migrations (version) VALUES (%s)",
                        (migration.name,),
                    )
        except psycopg.Error as exc:
            raise RepositoryStoreError(f"PostgreSQL initialization failed: {exc}") from exc

    async def close(self) -> None:
        """Connections are short-lived, so there is no pool to close."""

    async def register(self, name: str, root_path: str) -> RepositoryRecord:
        try:
            async with await self._connect() as connection:
                cursor = await connection.execute(
                    """
                    INSERT INTO repositories (name, root_path)
                    VALUES (%s, %s)
                    ON CONFLICT (root_path)
                    DO UPDATE SET name = EXCLUDED.name, updated_at = NOW()
                    RETURNING id, name, root_path, is_active
                    """,
                    (name, root_path),
                )
                row = await cursor.fetchone()
                if row is None:
                    raise RepositoryStoreError("PostgreSQL returned no repository row.")
                active_cursor = await connection.execute(
                    "SELECT 1 FROM repositories WHERE is_active LIMIT 1"
                )
                if not await active_cursor.fetchone():
                    cursor = await connection.execute(
                        """
                        UPDATE repositories SET is_active = TRUE, updated_at = NOW()
                        WHERE id = %s
                        RETURNING id, name, root_path, is_active
                        """,
                        (row["id"],),
                    )
                    row = await cursor.fetchone()
                return self._record(row)
        except psycopg.Error as exc:
            raise RepositoryStoreError(f"Could not register repository: {exc}") from exc

    async def list_repositories(self) -> list[RepositoryRecord]:
        try:
            async with await self._connect() as connection:
                cursor = await connection.execute(
                    """
                    SELECT id, name, root_path, is_active
                    FROM repositories
                    ORDER BY is_active DESC, name, id
                    """
                )
                return [self._record(row) for row in await cursor.fetchall()]
        except psycopg.Error as exc:
            raise RepositoryStoreError(f"Could not list repositories: {exc}") from exc

    async def get(self, repository_id: int) -> RepositoryRecord | None:
        try:
            async with await self._connect() as connection:
                cursor = await connection.execute(
                    """
                    SELECT id, name, root_path, is_active
                    FROM repositories WHERE id = %s
                    """,
                    (repository_id,),
                )
                row = await cursor.fetchone()
                return self._record(row) if row else None
        except psycopg.Error as exc:
            raise RepositoryStoreError(f"Could not read repository: {exc}") from exc

    async def select(self, repository_id: int) -> RepositoryRecord | None:
        try:
            async with await self._connect() as connection:
                exists = await connection.execute(
                    "SELECT 1 FROM repositories WHERE id = %s", (repository_id,)
                )
                if not await exists.fetchone():
                    return None
                await connection.execute(
                    "UPDATE repositories SET is_active = FALSE WHERE is_active"
                )
                cursor = await connection.execute(
                    """
                    UPDATE repositories SET is_active = TRUE, updated_at = NOW()
                    WHERE id = %s
                    RETURNING id, name, root_path, is_active
                    """,
                    (repository_id,),
                )
                row = await cursor.fetchone()
                return self._record(row) if row else None
        except psycopg.Error as exc:
            raise RepositoryStoreError(f"Could not select repository: {exc}") from exc

    async def replace_files(
        self,
        repository_id: int,
        files: Sequence[FileMetadata],
    ) -> None:
        try:
            async with await self._connect() as connection:
                await connection.execute(
                    "DELETE FROM repository_files WHERE repository_id = %s",
                    (repository_id,),
                )
                if not files:
                    return
                async with connection.cursor() as cursor:
                    await cursor.executemany(
                        """
                        INSERT INTO repository_files
                            (repository_id, path, size_bytes, modified_ns, sha256)
                        VALUES (%s, %s, %s, %s, %s)
                        """,
                        [
                            (
                                repository_id,
                                item.path,
                                item.size_bytes,
                                item.modified_ns,
                                item.sha256,
                            )
                            for item in files
                        ],
                    )
        except psycopg.Error as exc:
            raise RepositoryStoreError(f"Could not update file metadata: {exc}") from exc

    async def _connect(self):
        return await psycopg.AsyncConnection.connect(
            self._database_url,
            row_factory=dict_row,
        )

    @staticmethod
    def _record(row) -> RepositoryRecord:
        return RepositoryRecord(
            id=row["id"],
            name=row["name"],
            root_path=row["root_path"],
            is_active=row["is_active"],
        )
