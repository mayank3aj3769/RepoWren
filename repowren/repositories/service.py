"""Simple, safe repository scanning, chunking, and semantic retrieval."""

from __future__ import annotations

import asyncio
import os
from collections.abc import Iterator
from pathlib import Path

import pathspec

from repowren.embeddings import EmbeddingBackend
from repowren.repositories.models import (
    FileMetadata,
    RepositoryChunk,
    RepositoryRecord,
    SearchMatch,
)
from repowren.repositories.store import RepositoryStore


MAX_FILE_BYTES = 1_000_000
MAX_READ_BYTES = 200_000
MAX_LIST_RESULTS = 1_000
MAX_SEARCH_RESULTS = 50
MAX_CONTEXT_CHARACTERS = 9_000
CHUNK_LINES = 20
CONTEXT_RESULTS = 5

# This is the one obvious list to extend when another kind of file should never
# be read or embedded. The repository's root .gitignore is added to these rules.
EXCLUDED_PATTERNS = (
    ".git/",
    ".venv/",
    "venv/",
    ".idea/",
    ".vscode/",
    "__pycache__/",
    "node_modules/",
    "build/",
    "dist/",
    "models/",
    ".env",
    ".env.*",
    ".gitignore",
    "*.pyc",
    "*.pyo",
    "*.gguf",
    "*.safetensors",
    "*.bin",
)


class RepositoryAccessError(ValueError):
    """A requested path or search request is unsafe or invalid."""


class RepositoryService:
    """Coordinate safe file access with embeddings stored in PostgreSQL."""

    def __init__(self, store: RepositoryStore, embeddings: EmbeddingBackend) -> None:
        self._store = store
        self._embeddings = embeddings

    async def register(self, path: str, name: str | None = None) -> RepositoryRecord:
        """Register a Git root and completely rebuild its semantic index.

        The intentionally simple flow is:

        1. Validate ``C:/code/shop`` and save the repository row.
        2. Read every allowed text file in 20-line pieces.
        3. Turn each piece into a vector on a worker thread.
        4. Replace all old chunks for that repository in one database transaction.

        Re-registering a repository is therefore also the current re-index action.
        There is no hash comparison or partial update to understand yet.
        """
        root = self.validate_repository_root(path)
        record = await self._store.register(name or root.name, str(root))

        # Disk reads and model inference are blocking work. Moving them off the
        # API event loop lets health requests continue while indexing runs.
        chunk_inputs = await asyncio.to_thread(self._read_chunk_inputs, record)
        contents = [content for content, _ in chunk_inputs]
        vectors = await asyncio.to_thread(self._embeddings.encode, contents)
        if len(vectors) != len(chunk_inputs):
            raise RepositoryAccessError(
                "Embedding model returned a different number of vectors than chunks."
            )

        chunks = [
            RepositoryChunk(content=content, metadata=metadata, embedding=vector)
            for (content, metadata), vector in zip(chunk_inputs, vectors, strict=True)
        ]
        await self._store.replace_chunks(record.id, chunks)
        return record

    @staticmethod
    def validate_repository_root(path: str) -> Path:
        """Return a real Git repository root or raise a readable error."""
        try:
            # Resolving gives one canonical security boundary even if the user
            # enters a path containing ``..`` or ``~``.
            root = Path(path).expanduser().resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise RepositoryAccessError(f"Repository path does not exist: {path}") from exc
        if not root.is_dir():
            raise RepositoryAccessError(f"Repository path is not a directory: {root}")
        # Worktrees use a .git file; ordinary clones use a .git directory. Both
        # are valid, so only existence is checked.
        if not (root / ".git").exists():
            raise RepositoryAccessError(f"Not a Git repository: {root}")
        return root

    def scan_files(self, repository: RepositoryRecord) -> list[FileMetadata]:
        """Return live path and size information for allowed text files."""
        root = Path(repository.root_path).resolve(strict=True)
        files: list[FileMetadata] = []
        for file_path in self._iter_safe_files(root):
            files.append(
                FileMetadata(
                    path=file_path.relative_to(root).as_posix(),
                    size_bytes=file_path.stat().st_size,
                )
            )
            if len(files) >= MAX_LIST_RESULTS:
                break
        return files

    def list_files(
        self,
        repository: RepositoryRecord,
        *,
        query: str | None = None,
    ) -> list[FileMetadata]:
        """List current safe files, optionally filtering by path text."""
        files = self.scan_files(repository)
        if query:
            wanted = query.casefold()
            files = [item for item in files if wanted in item.path.casefold()]
        return files

    def read_file(self, repository: RepositoryRecord, relative_path: str) -> str:
        """Read one safe UTF-8 file without allowing secret or path escape.

        ``src/cart.py`` can pass. ``../secret.txt``, ``.env``, a binary file,
        an ignored file, and a symlink that leaves the repository all fail.
        """
        root = Path(repository.root_path).resolve(strict=True)
        file_path = self._resolve_file(root, relative_path)
        normalized_path = file_path.relative_to(root).as_posix()
        if self._ignore_spec(root).match_file(normalized_path):
            raise RepositoryAccessError("File is excluded from repository access.")
        if file_path.stat().st_size > MAX_READ_BYTES:
            raise RepositoryAccessError(
                f"File is too large to read ({file_path.stat().st_size} bytes)."
            )
        if not self._is_text_file(file_path):
            raise RepositoryAccessError("File is not a supported text file.")
        try:
            return file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise RepositoryAccessError("File is not valid UTF-8 text.") from exc

    async def search_code(
        self,
        repository: RepositoryRecord,
        query: str,
        *,
        max_results: int = MAX_SEARCH_RESULTS,
    ) -> list[SearchMatch]:
        """Find chunks with meaning similar to the question.

        For example, a question containing ``How is an order total computed?``
        can retrieve a chunk containing ``def calculate_price(...)`` even though
        the exact words differ. PostgreSQL ranks the stored vectors by cosine
        distance; this method only creates the query vector and requests a limit.
        """
        cleaned_query = query.strip()
        if not cleaned_query:
            raise RepositoryAccessError("Search query cannot be empty.")
        vector_rows = await asyncio.to_thread(self._embeddings.encode, [cleaned_query])
        if len(vector_rows) != 1:
            raise RepositoryAccessError("Embedding model did not return a query vector.")
        return await self._store.search_chunks(
            repository.id,
            vector_rows[0],
            max_results,
        )

    async def build_context(
        self,
        repository: RepositoryRecord,
        user_request: str,
    ) -> str:
        """Join the five closest stored chunks into one model instruction.

        A database result for ``src/cart.py`` lines 21-40 becomes:

        ``### src/cart.py lines 21-40`` followed by the stored source. There is
        no keyword extraction, duplicate-removal loop, or second file read.
        """
        matches = await self.search_code(
            repository,
            user_request,
            max_results=CONTEXT_RESULTS,
        )
        excerpts = [
            f"### {match.path} lines {match.start_line}-{match.end_line}\n{match.text}"
            for match in matches
        ]
        context = "\n\n".join(excerpts)[:MAX_CONTEXT_CHARACTERS]
        return (
            f"You are answering a question about the registered repository "
            f"'{repository.name}'. Use the semantic source chunks below. Cite file "
            "paths and line ranges when possible. If the chunks are insufficient, "
            "say so.\n\n"
            f"{context}"
        )

    def _read_chunk_inputs(
        self,
        repository: RepositoryRecord,
    ) -> list[tuple[str, dict[str, str | int]]]:
        """Read allowed files and split them into non-overlapping 20-line blocks.

        If ``app.py`` has 25 lines, this returns one item for lines 1-20 and one
        for lines 21-25. Each item carries JSON-friendly metadata beside its text.
        """
        root = Path(repository.root_path).resolve(strict=True)
        chunks: list[tuple[str, dict[str, str | int]]] = []
        for file_number, file_path in enumerate(self._iter_safe_files(root), start=1):
            if file_number > MAX_LIST_RESULTS:
                break
            try:
                lines = file_path.read_text(encoding="utf-8").splitlines()
            except (OSError, UnicodeDecodeError):
                continue
            relative_path = file_path.relative_to(root).as_posix()
            for offset in range(0, len(lines), CHUNK_LINES):
                block = lines[offset : offset + CHUNK_LINES]
                chunks.append(
                    (
                        "\n".join(block),
                        {
                            "path": relative_path,
                            "start_line": offset + 1,
                            "end_line": offset + len(block),
                        },
                    )
                )
        return chunks

    def _iter_safe_files(self, root: Path) -> Iterator[Path]:
        """Yield only small text files allowed by exclusions and .gitignore."""
        matcher = self._ignore_spec(root)
        for directory, directory_names, file_names in os.walk(
            root, topdown=True, followlinks=False
        ):
            directory_path = Path(directory)
            allowed_directories: list[str] = []
            for name in sorted(directory_names):
                candidate = directory_path / name
                relative = candidate.relative_to(root).as_posix() + "/"
                if candidate.is_symlink() or matcher.match_file(relative):
                    continue
                allowed_directories.append(name)
            # Changing the list in place tells os.walk not to enter excluded
            # directories such as .git or node_modules on its next step.
            directory_names[:] = allowed_directories

            for name in sorted(file_names):
                candidate = directory_path / name
                relative = candidate.relative_to(root).as_posix()
                if candidate.is_symlink() or matcher.match_file(relative):
                    continue
                try:
                    if candidate.stat().st_size > MAX_FILE_BYTES:
                        continue
                    if not self._is_text_file(candidate):
                        continue
                except OSError:
                    continue
                yield candidate

    @staticmethod
    def _ignore_spec(root: Path) -> pathspec.PathSpec:
        """Combine the visible exclusion list with the root .gitignore."""
        patterns = list(EXCLUDED_PATTERNS)
        gitignore = root / ".gitignore"
        if gitignore.is_file():
            try:
                patterns.extend(gitignore.read_text(encoding="utf-8").splitlines())
            except (OSError, UnicodeDecodeError):
                pass
        return pathspec.PathSpec.from_lines("gitwildmatch", patterns)

    @staticmethod
    def _resolve_file(root: Path, relative_path: str) -> Path:
        """Resolve a file and prove that its final target stays below root."""
        requested = Path(relative_path)
        if requested.is_absolute() or ".." in requested.parts:
            raise RepositoryAccessError("File path must stay inside the repository.")
        try:
            file_path = (root / requested).resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise RepositoryAccessError(f"File does not exist: {relative_path}") from exc
        if not file_path.is_relative_to(root) or not file_path.is_file():
            raise RepositoryAccessError("File path must stay inside the repository.")
        return file_path

    @staticmethod
    def _is_text_file(path: Path) -> bool:
        """Reject likely binary files by checking their first 4 KiB for nulls."""
        try:
            with path.open("rb") as stream:
                return b"\x00" not in stream.read(4_096)
        except OSError:
            return False
