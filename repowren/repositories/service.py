"""Safe, bounded access to registered source repositories."""

from __future__ import annotations

import asyncio
import hashlib
import os
import re
from collections.abc import Iterator
from pathlib import Path

import pathspec

from repowren.repositories.models import FileMetadata, RepositoryRecord, SearchMatch
from repowren.repositories.store import RepositoryStore


MAX_FILE_BYTES = 1_000_000
MAX_READ_CHARACTERS = 200_000
MAX_LIST_RESULTS = 1_000
MAX_SEARCH_RESULTS = 50
MAX_CONTEXT_CHARACTERS = 9_000

DEFAULT_EXCLUSIONS = (
    ".git/",
    ".venv/",
    ".idea/",
    ".vscode/",
    "__pycache__/",
    "node_modules/",
    "build/",
    "dist/",
    "models/",
    ".env",
    ".env.*",
    "*.pyc",
    "*.pyo",
    "*.gguf",
    "*.safetensors",
    "*.bin",
)

STOP_WORDS = {
    "about",
    "code",
    "does",
    "explain",
    "from",
    "have",
    "how",
    "into",
    "repository",
    "that",
    "this",
    "what",
    "where",
    "which",
    "with",
}


class RepositoryAccessError(ValueError):
    """A requested path is invalid or outside the registered repository."""


class RepositoryService:
    """Coordinate registration, safe file access, and basic retrieval."""

    def __init__(self, store: RepositoryStore) -> None:
        self._store = store

    async def register(self, path: str, name: str | None = None) -> RepositoryRecord:
        root = self.validate_repository_root(path)
        record = await self._store.register(name or root.name, str(root))
        files = await asyncio.to_thread(self.scan_files, record)
        await self._store.replace_files(record.id, files)
        return record

    @staticmethod
    def validate_repository_root(path: str) -> Path:
        try:
            root = Path(path).expanduser().resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise RepositoryAccessError(f"Repository path does not exist: {path}") from exc
        if not root.is_dir():
            raise RepositoryAccessError(f"Repository path is not a directory: {root}")
        if not (root / ".git").exists():
            raise RepositoryAccessError(f"Not a Git repository: {root}")
        return root

    def scan_files(self, repository: RepositoryRecord) -> list[FileMetadata]:
        root = Path(repository.root_path).resolve(strict=True)
        metadata: list[FileMetadata] = []
        for file_path in self._iter_safe_files(root):
            stat = file_path.stat()
            metadata.append(
                FileMetadata(
                    path=file_path.relative_to(root).as_posix(),
                    size_bytes=stat.st_size,
                    modified_ns=stat.st_mtime_ns,
                    sha256=self._hash_file(file_path),
                )
            )
            if len(metadata) >= MAX_LIST_RESULTS:
                break
        return metadata

    def list_files(
        self,
        repository: RepositoryRecord,
        *,
        query: str | None = None,
    ) -> list[FileMetadata]:
        files = self.scan_files(repository)
        if query:
            lowered = query.casefold()
            files = [item for item in files if lowered in item.path.casefold()]
        return files

    def read_file(self, repository: RepositoryRecord, relative_path: str) -> str:
        root = Path(repository.root_path).resolve(strict=True)
        file_path = self._resolve_file(root, relative_path)
        if file_path.stat().st_size > MAX_READ_CHARACTERS:
            raise RepositoryAccessError(
                f"File is too large to read ({file_path.stat().st_size} bytes)."
            )
        try:
            return file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise RepositoryAccessError("File is not valid UTF-8 text.") from exc

    def search_code(
        self,
        repository: RepositoryRecord,
        query: str,
        *,
        max_results: int = MAX_SEARCH_RESULTS,
    ) -> list[SearchMatch]:
        cleaned_query = query.strip()
        if not cleaned_query:
            raise RepositoryAccessError("Search query cannot be empty.")
        lowered_query = cleaned_query.casefold()
        root = Path(repository.root_path).resolve(strict=True)
        matches: list[SearchMatch] = []
        for file_path in self._iter_safe_files(root):
            try:
                lines = file_path.read_text(encoding="utf-8").splitlines()
            except (OSError, UnicodeDecodeError):
                continue
            relative_path = file_path.relative_to(root).as_posix()
            for line_number, line in enumerate(lines, start=1):
                if lowered_query in line.casefold():
                    matches.append(
                        SearchMatch(
                            path=relative_path,
                            line=line_number,
                            text=line[:500],
                        )
                    )
                    if len(matches) >= max_results:
                        return matches
        return matches

    def build_context(self, repository: RepositoryRecord, user_request: str) -> str:
        """Build small line-numbered excerpts for a repository-aware prompt."""
        terms = [
            term
            for term in re.findall(r"[A-Za-z_][A-Za-z0-9_.-]{2,}", user_request)
            if term.casefold() not in STOP_WORDS
        ]
        unique_terms = list(dict.fromkeys(term.casefold() for term in terms))[:8]
        candidates: list[SearchMatch] = []
        seen_locations: set[tuple[str, int]] = set()
        for term in unique_terms:
            for match in self.search_code(repository, term, max_results=5):
                location = (match.path, match.line)
                if location not in seen_locations:
                    seen_locations.add(location)
                    candidates.append(match)
                if len(candidates) >= 12:
                    break
            if len(candidates) >= 12:
                break

        excerpts: list[str] = []
        root = Path(repository.root_path).resolve(strict=True)
        for match in candidates:
            file_path = self._resolve_file(root, match.path)
            try:
                lines = file_path.read_text(encoding="utf-8").splitlines()
            except (OSError, UnicodeDecodeError):
                continue
            start = max(0, match.line - 3)
            end = min(len(lines), match.line + 2)
            numbered = "\n".join(
                f"{line_number + 1}: {lines[line_number]}"
                for line_number in range(start, end)
            )
            excerpts.append(f"### {match.path}\n{numbered}")
            if sum(len(excerpt) for excerpt in excerpts) >= MAX_CONTEXT_CHARACTERS:
                break

        if not excerpts:
            for fallback in ("README.md", "pyproject.toml"):
                try:
                    content = self.read_file(repository, fallback)
                except RepositoryAccessError:
                    continue
                excerpts.append(f"### {fallback}\n{content[:4_000]}")
                break

        context = "\n\n".join(excerpts)[:MAX_CONTEXT_CHARACTERS]
        return (
            f"You are answering a question about the registered repository "
            f"'{repository.name}'. Use the source excerpts below. Cite file paths "
            "and line numbers when possible. If the excerpts are insufficient, say so.\n\n"
            f"{context}"
        )

    def _iter_safe_files(self, root: Path) -> Iterator[Path]:
        matcher = self._ignore_spec(root)
        for directory, directory_names, file_names in os.walk(
            root, topdown=True, followlinks=False
        ):
            directory_path = Path(directory)
            safe_directories: list[str] = []
            for name in sorted(directory_names):
                candidate = directory_path / name
                relative = candidate.relative_to(root).as_posix() + "/"
                if candidate.is_symlink() or matcher.match_file(relative):
                    continue
                safe_directories.append(name)
            directory_names[:] = safe_directories

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
        patterns = list(DEFAULT_EXCLUSIONS)
        gitignore = root / ".gitignore"
        if gitignore.is_file():
            try:
                patterns.extend(gitignore.read_text(encoding="utf-8").splitlines())
            except (OSError, UnicodeDecodeError):
                pass
        return pathspec.PathSpec.from_lines("gitwildmatch", patterns)

    @staticmethod
    def _resolve_file(root: Path, relative_path: str) -> Path:
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
        try:
            with path.open("rb") as stream:
                return b"\x00" not in stream.read(4_096)
        except OSError:
            return False

    @staticmethod
    def _hash_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while chunk := stream.read(64 * 1_024):
                digest.update(chunk)
        return digest.hexdigest()
