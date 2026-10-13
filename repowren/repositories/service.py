"""Simple, safe repository scanning, chunking, and semantic retrieval."""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import pathspec

from repowren.api.schemas import ChatMessage
from repowren.embeddings import EmbeddingBackend
from repowren.inference.base import InferenceBackend, InferenceError
from repowren.lexical import LexicalSearchIndex
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
MAX_CONTEXT_MATCH_CHARACTERS = 2_400
CHUNK_LINES = 20
CONTEXT_RESULTS = 5
SEARCH_CANDIDATE_LIMIT = 20
RRF_RANK_OFFSET = 60
MAX_LEXICAL_QUERIES = 8
MAX_PLANNER_TOKENS = 192
PLANNER_FILLER_WORDS = frozenset(
    {
        "a",
        "an",
        "code",
        "do",
        "does",
        "function",
        "how",
        "many",
        "method",
        "the",
        "too",
        "what",
        "where",
        "which",
    }
)

SEARCH_PLANNER_PROMPT = """You plan hybrid searches over source-code chunks.
Return only a JSON object with this exact shape:
{"embedding_query":"...", "lexical_queries":["...", "..."]}

Rules:
- embedding_query is one concise natural-language description of the code sought.
- lexical_queries contains 1 to 8 high-signal strings to try verbatim in code.
- Prefer supplied or plausible identifiers, API names, error text, configuration
  keys, and path fragments. For an important code concept, include a snake_case
  identifier and, when useful, a camelCase or PascalCase alternative.
- Exclude question words and filler such as where, how, code, too, and many.
- Preserve technical names exactly. The probes need not be claims that code exists.
- Do not include Markdown or any text outside the JSON object.

Example:
User: Where does the app refresh an expired auth token?
Output: {"embedding_query":"authentication token refresh after expiration","lexical_queries":["refresh_token","refreshToken","TokenExpired","token expired"]}
"""

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


@dataclass(frozen=True, slots=True)
class SearchQueryPlan:
    """The semantic sentence and exact strings derived from one user query."""

    embedding_query: str
    lexical_queries: tuple[str, ...]


class RepositoryService:
    """Coordinate safe file access with embeddings stored in PostgreSQL."""

    def __init__(
        self,
        store: RepositoryStore,
        embeddings: EmbeddingBackend,
        lexical_index: LexicalSearchIndex,
        inference: InferenceBackend,
    ) -> None:
        self._store = store
        self._embeddings = embeddings
        self._lexical_index = lexical_index
        self._inference = inference

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
        # PostgreSQL owns the authoritative semantic snapshot. Elasticsearch is
        # replaced second because the two services cannot share a transaction.
        # If the lexical write fails, re-registering repeats both full rebuilds.
        await self._store.replace_chunks(record.id, chunks)
        await self._lexical_index.replace_repository_chunks(record.id, chunks)
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
        """Return only fused matches for repository-chat context."""
        _, matches = await self.search_code_with_plan(
            repository,
            query,
            max_results=max_results,
        )
        return matches

    async def search_code_with_plan(
        self,
        repository: RepositoryRecord,
        query: str,
        *,
        max_results: int = MAX_SEARCH_RESULTS,
    ) -> tuple[SearchQueryPlan, list[SearchMatch]]:
        """Combine semantic and lexical candidates with reciprocal rank fusion.

        For example, a question containing ``How is an order total computed?``
        can retrieve a chunk containing ``def calculate_price(...)`` even though
        the exact words differ. PostgreSQL ranks the stored vectors by cosine
        distance; this method only creates the query vector and requests a limit.
        """
        cleaned_query = query.strip()
        if not cleaned_query:
            raise RepositoryAccessError("Search query cannot be empty.")
        query_plan = await self._plan_search_queries(cleaned_query)
        vector_rows = await asyncio.to_thread(
            self._embeddings.encode,
            [query_plan.embedding_query],
        )
        if len(vector_rows) != 1:
            raise RepositoryAccessError("Embedding model did not return a query vector.")
        # The backends are independent once the single query vector exists.
        # Running them together avoids adding their network latencies while
        # preserving clear failures instead of silently dropping one result set.
        semantic_matches, lexical_matches = await asyncio.gather(
            self._store.search_chunks(
                repository.id,
                vector_rows[0],
                SEARCH_CANDIDATE_LIMIT,
            ),
            self._lexical_index.search(
                repository.id,
                query_plan.lexical_queries,
                SEARCH_CANDIDATE_LIMIT,
            ),
        )
        return (
            query_plan,
            fuse_search_results(
                semantic_matches,
                lexical_matches,
                max_results=max_results,
            ),
        )

    async def _plan_search_queries(self, query: str) -> SearchQueryPlan:
        """Ask vLLM for focused semantic and exact-text retrieval strings.

        Planning uses the existing local chat endpoint with deterministic
        sampling. If a model returns prose or invalid JSON, search still works
        with the original user query rather than failing on model formatting.
        """
        try:
            parts = [
                part
                async for part in self._inference.stream_chat(
                    [
                        ChatMessage(role="system", content=SEARCH_PLANNER_PROMPT),
                        ChatMessage(role="user", content=query),
                    ],
                    max_tokens=MAX_PLANNER_TOKENS,
                    temperature=0.0,
                )
            ]
        except InferenceError:
            # Search remains useful while vLLM is starting or temporarily down.
            # Storage failures are still surfaced rather than hidden here.
            return SearchQueryPlan(query, (query,))
        return parse_search_plan("".join(parts), fallback_query=query)

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
        excerpts: list[str] = []
        remaining = MAX_CONTEXT_CHARACTERS
        for match in matches:
            excerpt = f"### {match.path} lines {match.start_line}-{match.end_line}\n{match.text}"
            if len(excerpt) > MAX_CONTEXT_MATCH_CHARACTERS:
                excerpt = (
                    excerpt[:MAX_CONTEXT_MATCH_CHARACTERS].rstrip()
                    + "\n[chunk truncated]"
                )
            if len(excerpt) > remaining:
                excerpt = excerpt[:remaining].rstrip() + "\n[context truncated]"
            if not excerpt:
                break
            excerpts.append(excerpt)
            remaining -= len(excerpt) + 2
            if remaining <= 0:
                break
        context = "\n\n".join(excerpts)
        return (
            f"You are answering a question about the registered repository "
            f"'{repository.name}'. Use the hybrid-ranked source chunks below. Cite file "
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


def fuse_search_results(
    semantic_matches: Sequence[SearchMatch],
    lexical_matches: Sequence[SearchMatch],
    *,
    max_results: int,
) -> list[SearchMatch]:
    """Merge two rankings without comparing cosine and BM25 scores directly.

    A chunk ranked first by both backends receives ``1/61 + 1/61``. A chunk
    present in just one list receives only that list's contribution. A small
    dictionary keyed by ``chunk_key`` is enough to combine the two rankings.
    """
    if max_results < 1:
        raise RepositoryAccessError("Search result limit must be positive.")

    scores: dict[str, float] = {}
    matches_by_key: dict[str, SearchMatch] = {}
    first_seen: dict[str, int] = {}
    for ranked_matches in (semantic_matches, lexical_matches):
        for rank, match in enumerate(ranked_matches, start=1):
            if match.chunk_key not in first_seen:
                first_seen[match.chunk_key] = len(first_seen)
                matches_by_key[match.chunk_key] = match
            scores[match.chunk_key] = scores.get(match.chunk_key, 0.0) + (
                1.0 / (RRF_RANK_OFFSET + rank)
            )

    ordered_keys = sorted(
        scores,
        key=lambda key: (-scores[key], first_seen[key]),
    )
    return [
        replace(matches_by_key[key], score=scores[key])
        for key in ordered_keys[:max_results]
    ]


def parse_search_plan(response: str, *, fallback_query: str) -> SearchQueryPlan:
    """Validate the model's small JSON plan and provide a safe fallback."""
    fallback = SearchQueryPlan(fallback_query, (fallback_query,))
    text = response.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1]).strip() if len(lines) >= 3 else ""
    try:
        value = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return fallback
    if not isinstance(value, dict):
        return fallback

    embedding_query = value.get("embedding_query")
    raw_lexical_queries = value.get("lexical_queries")
    if not isinstance(embedding_query, str) or not isinstance(
        raw_lexical_queries, list
    ):
        return fallback
    embedding_query = " ".join(embedding_query.split())[:500]
    if not embedding_query:
        return fallback

    lexical_queries: list[str] = []
    seen: set[str] = set()
    for item in raw_lexical_queries:
        if not isinstance(item, str):
            continue
        cleaned = " ".join(item.split())[:200]
        key = cleaned.casefold()
        if cleaned and key not in seen and key not in PLANNER_FILLER_WORDS:
            lexical_queries.append(cleaned)
            seen.add(key)
        if len(lexical_queries) == MAX_LEXICAL_QUERIES:
            break
    if not lexical_queries:
        return fallback
    return SearchQueryPlan(embedding_query, tuple(lexical_queries))
