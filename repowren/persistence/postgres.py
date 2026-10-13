"""Small psycopg data-access layer for repository metadata."""

from __future__ import annotations

from collections.abc import Sequence
from importlib import resources

import psycopg
from pgvector import Vector
from pgvector.psycopg import register_vector_async
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from repowren.repositories.models import (
    RepositoryChunk,
    RepositoryRecord,
    SearchMatch,
    build_chunk_key,
)
from repowren.repositories.store import RepositoryStoreError


class PostgresRepositoryStore:
    """Persist registered repositories without adding a full ORM."""

    def __init__(self, database_url: str) -> None:
        """Remember the connection URL; connections are opened per operation."""
        self._database_url = database_url

    async def initialize(self) -> None:
        """Create migration tracking and apply each new packaged SQL file once.

        If ``001_repositories.sql`` is already present in ``schema_migrations``,
        it is skipped. A later ``002_chunks.sql`` would run once and then be
        recorded in exactly the same way.
        """
        try:
            # A brand-new database cannot register pgvector's Python adapters
            # until the migration below has first created the extension.
            async with await self._connect(register_vectors=False) as connection:
                # This table exists before the first real migration so RepoWren
                # can remember which schema files have already been applied.
                await connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS schema_migrations (
                        version TEXT PRIMARY KEY,
                        applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    )
                    """
                )
                # importlib.resources works both in an editable checkout and in
                # an installed wheel; it does not assume a hard-coded disk path.
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
                    # Filename ordering gives migrations a predictable order.
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
        """Insert a repository, or refresh its name when the root already exists.

        Example input ``('shop', 'C:\\code\\shop')`` may return a dict-like
        row ``{'id': 3, 'name': 'shop', 'root_path': 'C:\\code\\shop',
        'is_active': True}``, which is then converted to `RepositoryRecord`.
        """
        try:
            async with await self._connect() as connection:
                # root_path is UNIQUE. ON CONFLICT makes registration repeatable:
                # adding the same folder twice updates its display name instead
                # of creating a duplicate database row.
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
                # RETURNING lets PostgreSQL give us the saved row immediately;
                # no second lookup is needed.
                row = await cursor.fetchone()
                if row is None:
                    raise RepositoryStoreError("PostgreSQL returned no repository row.")
                active_cursor = await connection.execute(
                    "SELECT 1 FROM repositories WHERE is_active LIMIT 1"
                )
                if not await active_cursor.fetchone():
                    # The first repository becomes active automatically so the
                    # user can start chatting without a separate select step.
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
        """List repositories with the active selection at the top.

        Database dictionaries are converted into domain objects so callers do
        not need to know column access or psycopg row details.
        """
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
        """Return one repository, or None when the ID is unknown.

        For example, ``get(3)`` may return the `shop` record, while ``get(999)``
        returns ``None`` and lets the API turn that into HTTP 404.
        """
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
        """Clear the old active flag and make the requested repository active.

        If repository 1 is active and the user selects repository 3, the first
        update changes repository 1 to false and the second changes repository
        3 to true. Both changes are committed together.
        """
        try:
            async with await self._connect() as connection:
                exists = await connection.execute(
                    "SELECT 1 FROM repositories WHERE id = %s", (repository_id,)
                )
                if not await exists.fetchone():
                    return None
                # Both updates share one connection context, so they commit as
                # one transaction when the context exits successfully.
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

    async def replace_chunks(
        self,
        repository_id: int,
        chunks: Sequence[RepositoryChunk],
    ) -> None:
        """Replace every old chunk with one newly generated snapshot.

        One item contains source text, JSON metadata such as
        ``{'path': 'src/cart.py', 'start_line': 1, 'end_line': 20}``, and its
        384-number vector. Deleting and inserting on the same connection makes
        the rebuild one transaction: failure leaves the older index intact.
        """
        try:
            async with await self._connect() as connection:
                await connection.execute(
                    "DELETE FROM repository_chunks WHERE repository_id = %s",
                    (repository_id,),
                )
                if not chunks:
                    return
                async with connection.cursor() as cursor:
                    await cursor.executemany(
                        """
                        INSERT INTO repository_chunks
                            (repository_id, content, metadata, embedding)
                        VALUES (%s, %s, %s, %s)
                        """,
                        [
                            (
                                repository_id,
                                item.content,
                                Jsonb(item.metadata),
                                # Vector prevents psycopg from treating the
                                # Python list as a PostgreSQL float array.
                                Vector(item.embedding),
                            )
                            for item in chunks
                        ],
                    )
        except psycopg.Error as exc:
            raise RepositoryStoreError(f"Could not replace repository chunks: {exc}") from exc

    async def search_chunks(
        self,
        repository_id: int,
        query_embedding: Sequence[float],
        limit: int,
    ) -> list[SearchMatch]:
        """Return exact cosine-nearest chunks; no vector index is involved yet."""
        try:
            async with await self._connect() as connection:
                # Without this wrapper PostgreSQL sees ``double precision[]``;
                # the cosine operator is defined for pgvector values instead.
                query_vector = Vector(query_embedding)
                cursor = await connection.execute(
                    """
                    SELECT content, metadata,
                           1 - (embedding <=> %s) AS score
                    FROM repository_chunks
                    WHERE repository_id = %s
                    ORDER BY embedding <=> %s
                    LIMIT %s
                    """,
                    (query_vector, repository_id, query_vector, limit),
                )
                matches: list[SearchMatch] = []
                for row in await cursor.fetchall():
                    metadata = row["metadata"]
                    matches.append(
                        SearchMatch(
                            chunk_key=build_chunk_key(
                                repository_id,
                                str(metadata["path"]),
                                int(metadata["start_line"]),
                                int(metadata["end_line"]),
                            ),
                            path=str(metadata["path"]),
                            start_line=int(metadata["start_line"]),
                            end_line=int(metadata["end_line"]),
                            text=row["content"],
                            score=float(row["score"]),
                        )
                    )
                return matches
        except psycopg.Error as exc:
            raise RepositoryStoreError(f"Could not search repository chunks: {exc}") from exc

    async def _connect(self, *, register_vectors: bool = True):
        """Open a short-lived async connection that returns dict-like rows.

        With ``dict_row``, code reads ``row['name']`` instead of remembering
        that the name happened to be the second selected column.
        """
        connection = await psycopg.AsyncConnection.connect(
            self._database_url,
            row_factory=dict_row,
        )
        if register_vectors:
            await register_vector_async(connection)
        return connection

    @staticmethod
    def _record(row) -> RepositoryRecord:
        """Translate a database row into the application's data object.

        This conversion keeps PostgreSQL-shaped data at the persistence edge;
        the repository and API layers receive a typed `RepositoryRecord`.
        """
        return RepositoryRecord(
            id=row["id"],
            name=row["name"],
            root_path=row["root_path"],
            is_active=row["is_active"],
        )
