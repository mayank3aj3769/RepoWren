"""Simple, safe repository scanning, chunking, and semantic retrieval."""

from __future__ import annotations

import asyncio
import ast
import json
import os
from collections import defaultdict
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TypedDict

import pathspec
from langgraph.graph import END, START, StateGraph

from repowren.api.schemas import ChatMessage
from repowren.embeddings import EmbeddingBackend
from repowren.inference.base import InferenceBackend, InferenceError
from repowren.lexical import LexicalSearchIndex
from repowren.repositories.models import (
    FileMetadata,
    GraphRelation,
    GraphSearchMatch,
    RepositoryChunk,
    RepositoryGraphEdge,
    RepositoryRecord,
    SearchMatch,
    build_chunk_key,
)
from repowren.repositories.store import RepositoryStore


MAX_FILE_BYTES = 1_000_000
MAX_READ_BYTES = 200_000
MAX_LIST_RESULTS = 1_000
MAX_SEARCH_RESULTS = 50
MAX_CONTEXT_CHARACTERS = 8_000
MAX_CONTEXT_MATCH_CHARACTERS = 2_400
GENERIC_CHUNK_LINES = 60
GENERIC_CHUNK_OVERLAP = 8
MAX_PYTHON_CHUNK_LINES = 100
CONTEXT_RESULTS = 5
GRAPH_CONTEXT_RESULTS = 8
GRAPH_EXPANSION_DEPTH = 2
SEARCH_CANDIDATE_LIMIT = 20
MAX_GRAPH_EDGES = 50_000
MAX_GRAPH_EDGES_PER_CHUNK = 24
RRF_RANK_OFFSET = 60
MAX_LEXICAL_QUERIES = 8
MAX_CROSS_QUESTIONS = 3
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
{"embedding_query":"...", "lexical_queries":["...", "..."], "cross_questions":["...", "..."]}

Rules:
- embedding_query is one concise natural-language description of the code sought.
- lexical_queries contains 1 to 8 high-signal strings to try verbatim in code.
- cross_questions contains 1 to 3 short complementary questions about callers,
  data flow, configuration, tests, or failure handling that would reveal connected code.
- Prefer supplied or plausible identifiers, API names, error text, configuration
  keys, and path fragments. For an important code concept, include a snake_case
  identifier and, when useful, a camelCase or PascalCase alternative.
- Exclude question words and filler such as where, how, code, too, and many.
- Preserve technical names exactly. The probes need not be claims that code exists.
- Do not include Markdown or any text outside the JSON object.

Example:
User: Where does the app refresh an expired auth token?
Output: {"embedding_query":"authentication token refresh after expiration","lexical_queries":["refresh_token","refreshToken","TokenExpired","token expired"],"cross_questions":["Who calls the token refresh function?","How are refresh failures handled?"]}
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
    cross_questions: tuple[str, ...] = ()


class IndexingState(TypedDict, total=False):
    """Values passed between the four explicit repository-indexing nodes."""

    repository: RepositoryRecord
    chunk_inputs: list[tuple[str, dict[str, str | int]]]
    chunks: list[RepositoryChunk]
    graph_edges: list[RepositoryGraphEdge]


class RetrievalState(TypedDict, total=False):
    """Values passed through planning, retrieval, fusion, graph, and context nodes."""

    repository: RepositoryRecord
    query: str
    max_results: int
    max_graph_results: int
    query_plan: SearchQueryPlan
    ranked_matches: list[list[SearchMatch]]
    matches: list[SearchMatch]
    graph_matches: list[GraphSearchMatch]
    context: str


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
        self._indexing_graph = self._build_indexing_graph()
        self._retrieval_graph = self._build_retrieval_graph()

    async def register(self, path: str, name: str | None = None) -> RepositoryRecord:
        """Register a Git root and completely rebuild its semantic index.

        The intentionally simple flow is:

        1. Validate ``C:/code/shop`` and save the repository row.
        2. Run syntax-aware chunking.
        3. Embed the chunks and construct relationships.
        4. Replace the PostgreSQL and Elasticsearch snapshots.

        Re-registering a repository is therefore also the current re-index action.
        There is no hash comparison or partial update to understand yet.
        """
        root = self.validate_repository_root(path)
        record = await self._store.register(name or root.name, str(root))

        await self._indexing_graph.ainvoke({"repository": record})
        return record

    def _build_indexing_graph(self):
        graph = StateGraph(IndexingState)
        graph.add_node("scan_and_chunk", self._scan_and_chunk_node)
        graph.add_node("embed_chunks", self._embed_chunks_node)
        graph.add_node("link_relationships", self._link_relationships_node)
        graph.add_node("persist_snapshot", self._persist_snapshot_node)
        graph.add_edge(START, "scan_and_chunk")
        graph.add_edge("scan_and_chunk", "embed_chunks")
        graph.add_edge("embed_chunks", "link_relationships")
        graph.add_edge("link_relationships", "persist_snapshot")
        graph.add_edge("persist_snapshot", END)
        return graph.compile()

    def _build_retrieval_graph(self):
        graph = StateGraph(RetrievalState)
        graph.add_node("plan_queries", self._plan_queries_node)
        graph.add_node("retrieve_candidates", self._retrieve_candidates_node)
        graph.add_node("fuse_rankings", self._fuse_rankings_node)
        graph.add_node("expand_relationships", self._expand_relationships_node)
        graph.add_node("assemble_context", self._assemble_context_node)
        graph.add_edge(START, "plan_queries")
        graph.add_edge("plan_queries", "retrieve_candidates")
        graph.add_edge("retrieve_candidates", "fuse_rankings")
        graph.add_edge("fuse_rankings", "expand_relationships")
        graph.add_edge("expand_relationships", "assemble_context")
        graph.add_edge("assemble_context", END)
        return graph.compile()

    async def _scan_and_chunk_node(self, state: IndexingState) -> IndexingState:
        chunk_inputs = await asyncio.to_thread(
            self._read_chunk_inputs, state["repository"]
        )
        return {"chunk_inputs": chunk_inputs}

    async def _embed_chunks_node(self, state: IndexingState) -> IndexingState:
        chunk_inputs = state["chunk_inputs"]
        vectors = await asyncio.to_thread(
            self._embeddings.encode, [content for content, _ in chunk_inputs]
        )
        if len(vectors) != len(chunk_inputs):
            raise RepositoryAccessError(
                "Embedding model returned a different number of vectors than chunks."
            )
        return {
            "chunks": [
                RepositoryChunk(content=content, metadata=metadata, embedding=vector)
                for (content, metadata), vector in zip(
                    chunk_inputs, vectors, strict=True
                )
            ]
        }

    async def _link_relationships_node(self, state: IndexingState) -> IndexingState:
        return {
            "graph_edges": await asyncio.to_thread(
                build_repository_graph,
                state["repository"].id,
                state["chunks"],
            )
        }

    async def _persist_snapshot_node(self, state: IndexingState) -> IndexingState:
        repository_id = state["repository"].id
        chunks = state["chunks"]
        await self._store.replace_chunks(
            repository_id, chunks, state["graph_edges"]
        )
        await self._lexical_index.replace_repository_chunks(repository_id, chunks)
        return {}

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
        query_plan, matches, _ = await self.search_code_with_graph(
            repository,
            query,
            max_results=max_results,
            max_graph_results=0,
        )
        return query_plan, matches

    async def search_code_with_graph(
        self,
        repository: RepositoryRecord,
        query: str,
        *,
        max_results: int = MAX_SEARCH_RESULTS,
        max_graph_results: int = GRAPH_CONTEXT_RESULTS,
    ) -> tuple[SearchQueryPlan, list[SearchMatch], list[GraphSearchMatch]]:
        """Run the multi-node retrieval graph and return ranked and linked code.

        Graph neighbors never receive synthetic relevance scores and never
        displace hybrid results. Keeping the two lists separate makes their
        provenance visible to API clients and prompts.
        """
        cleaned_query = query.strip()
        if not cleaned_query:
            raise RepositoryAccessError("Search query cannot be empty.")
        if max_results < 1:
            raise RepositoryAccessError("Search result limit must be positive.")
        state = await self._retrieval_graph.ainvoke(
            {
                "repository": repository,
                "query": cleaned_query,
                "max_results": max_results,
                "max_graph_results": max_graph_results,
            }
        )
        return state["query_plan"], state["matches"], state["graph_matches"]

    async def _plan_queries_node(self, state: RetrievalState) -> RetrievalState:
        return {"query_plan": await self._plan_search_queries(state["query"])}

    async def _retrieve_candidates_node(
        self, state: RetrievalState
    ) -> RetrievalState:
        plan = state["query_plan"]
        semantic_queries = (plan.embedding_query, *plan.cross_questions)
        vector_rows = await asyncio.to_thread(
            self._embeddings.encode, list(semantic_queries)
        )
        if len(vector_rows) != len(semantic_queries):
            raise RepositoryAccessError("Embedding model did not return query vectors.")
        tasks = [
            self._store.search_chunks(
                state["repository"].id, vector, SEARCH_CANDIDATE_LIMIT
            )
            for vector in vector_rows
        ]
        tasks.append(
            self._lexical_index.search(
                state["repository"].id,
                plan.lexical_queries,
                SEARCH_CANDIDATE_LIMIT,
            )
        )
        return {"ranked_matches": list(await asyncio.gather(*tasks))}

    async def _fuse_rankings_node(self, state: RetrievalState) -> RetrievalState:
        return {
            "matches": fuse_ranked_results(
                state["ranked_matches"], max_results=state["max_results"]
            )
        }

    async def _expand_relationships_node(
        self, state: RetrievalState
    ) -> RetrievalState:
        limit = state["max_graph_results"]
        if limit < 1 or not state["matches"]:
            return {"graph_matches": []}

        repository_id = state["repository"].id
        seed_keys = [match.chunk_key for match in state["matches"]]
        first_limit = min(limit, max(1, (limit + 1) // 2))
        first_hop = await self._store.graph_neighbors(
            repository_id, seed_keys, first_limit
        )
        selected = list(first_hop)
        known_keys = set(seed_keys)
        known_keys.update(item.match.chunk_key for item in first_hop)

        if GRAPH_EXPANSION_DEPTH > 1 and len(selected) < limit and first_hop:
            second_hop = await self._store.graph_neighbors(
                repository_id,
                [item.match.chunk_key for item in first_hop],
                limit - len(selected) + len(seed_keys),
            )
            for item in second_hop:
                if item.match.chunk_key in known_keys:
                    continue
                selected.append(replace(item, depth=2))
                known_keys.add(item.match.chunk_key)
                if len(selected) == limit:
                    break
        return {"graph_matches": selected}

    async def _assemble_context_node(self, state: RetrievalState) -> RetrievalState:
        return {
            "context": self._format_context(
                state["repository"], state["matches"], state["graph_matches"]
            )
        }

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
            return SearchQueryPlan(query, (query,), ())
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
        context, _, _, _ = await self.build_context_with_trace(
            repository, user_request
        )
        return context

    async def build_context_with_trace(
        self,
        repository: RepositoryRecord,
        user_request: str,
    ) -> tuple[str, SearchQueryPlan, list[SearchMatch], list[GraphSearchMatch]]:
        """Return the prompt plus the exact retrieval trace shown by chat."""
        cleaned_request = user_request.strip()
        if not cleaned_request:
            raise RepositoryAccessError("Search query cannot be empty.")
        state = await self._retrieval_graph.ainvoke(
            {
                "repository": repository,
                "query": cleaned_request,
                "max_results": CONTEXT_RESULTS,
                "max_graph_results": GRAPH_CONTEXT_RESULTS,
            }
        )
        return (
            state["context"],
            state["query_plan"],
            state["matches"],
            state["graph_matches"],
        )

    def _format_context(
        self,
        repository: RepositoryRecord,
        matches: Sequence[SearchMatch],
        graph_matches: Sequence[GraphSearchMatch],
    ) -> str:
        """Build a bounded prompt while retaining relationship provenance."""
        raw_excerpts = [
            f"### {match.path} lines {match.start_line}-{match.end_line} "
            "[hybrid retrieval seed]\n"
            f"{match.text}"
            for match in matches
        ]
        sources_by_key = {match.chunk_key: match for match in matches}
        sources_by_key.update(
            {item.match.chunk_key: item.match for item in graph_matches}
        )
        for graph_match in graph_matches:
            seed = sources_by_key.get(graph_match.seed_chunk_key)
            if seed is None:
                continue
            detail = f" ({graph_match.detail})" if graph_match.detail else ""
            match = graph_match.match
            raw_excerpts.append(
                f"### {match.path} lines {match.start_line}-{match.end_line} "
                f"[graph depth {graph_match.depth}: {seed.path} lines {seed.start_line}-{seed.end_line} "
                f"{graph_match.relation}{detail}]\n{match.text}"
            )
        excerpts: list[str] = []
        remaining = MAX_CONTEXT_CHARACTERS
        for excerpt in raw_excerpts:
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
            "paths and line ranges when possible. Graph-expanded chunks are related "
            "evidence, not proof of relevance. If the chunks are insufficient or a "
            "relationship does not support the answer, say so.\n\n"
            f"{context}"
        )

    def _read_chunk_inputs(
        self,
        repository: RepositoryRecord,
    ) -> list[tuple[str, dict[str, str | int]]]:
        """Read allowed files and choose syntax-aware or bounded text chunks."""
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
            for start_line, end_line, extra_metadata in chunk_ranges(
                relative_path, lines
            ):
                block = lines[start_line - 1 : end_line]
                chunks.append(
                    (
                        "\n".join(block),
                        {
                            "path": relative_path,
                            "start_line": start_line,
                            "end_line": end_line,
                            **extra_metadata,
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

    return fuse_ranked_results(
        (semantic_matches, lexical_matches), max_results=max_results
    )


def fuse_ranked_results(
    rankings: Sequence[Sequence[SearchMatch]],
    *,
    max_results: int,
) -> list[SearchMatch]:
    """RRF-merge any number of semantic and lexical rankings."""
    if max_results < 1:
        raise RepositoryAccessError("Search result limit must be positive.")

    scores: dict[str, float] = {}
    matches_by_key: dict[str, SearchMatch] = {}
    first_seen: dict[str, int] = {}
    for ranked_matches in rankings:
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


def chunk_ranges(
    relative_path: str,
    lines: Sequence[str],
) -> list[tuple[int, int, dict[str, str | int]]]:
    """Return deterministic line ranges without separating normal Python symbols.

    A Python function or class up to 100 lines stays whole. Oversized symbols
    are split into bounded parts. Other text uses 60-line windows with eight
    lines of overlap so a relationship near a boundary remains visible.
    """
    if not lines:
        return []
    if relative_path.endswith(".py"):
        try:
            tree = ast.parse("\n".join(lines), filename=relative_path)
        except (SyntaxError, ValueError, TypeError):
            pass
        else:
            symbols = [
                node
                for node in tree.body
                if isinstance(
                    node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
                )
            ]
            if symbols:
                result: list[tuple[int, int, dict[str, str | int]]] = []
                first_start = _node_start_line(symbols[0])
                if first_start > 1:
                    result.extend(
                        _split_range(
                            1,
                            first_start - 1,
                            MAX_PYTHON_CHUNK_LINES,
                            {"chunk_kind": "python_module"},
                        )
                    )
                for index, symbol in enumerate(symbols):
                    start = _node_start_line(symbol)
                    end = (
                        _node_start_line(symbols[index + 1]) - 1
                        if index + 1 < len(symbols)
                        else len(lines)
                    )
                    result.extend(
                        _split_range(
                            start,
                            end,
                            MAX_PYTHON_CHUNK_LINES,
                            {
                                "chunk_kind": "python_symbol",
                                "symbol": symbol.name,
                            },
                        )
                    )
                return result

    step = GENERIC_CHUNK_LINES - GENERIC_CHUNK_OVERLAP
    windows: list[tuple[int, int, dict[str, str | int]]] = []
    offset = 0
    while offset < len(lines):
        end = min(offset + GENERIC_CHUNK_LINES, len(lines))
        windows.append((offset + 1, end, {"chunk_kind": "text_window"}))
        if end == len(lines):
            break
        offset += step
    return windows


def _node_start_line(
    node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef,
) -> int:
    decorator_lines = [decorator.lineno for decorator in node.decorator_list]
    return min([node.lineno, *decorator_lines])


def _split_range(
    start: int,
    end: int,
    size: int,
    metadata: dict[str, str | int],
) -> list[tuple[int, int, dict[str, str | int]]]:
    parts: list[tuple[int, int, dict[str, str | int]]] = []
    for part_number, part_start in enumerate(range(start, end + 1, size), start=1):
        part_metadata = dict(metadata)
        if end - start + 1 > size:
            part_metadata["part"] = part_number
        parts.append((part_start, min(part_start + size - 1, end), part_metadata))
    return parts


def build_repository_graph(
    repository_id: int,
    chunks: Sequence[RepositoryChunk],
) -> list[RepositoryGraphEdge]:
    """Build a small deterministic graph over already-approved source chunks.

    All files receive undirected-at-query-time adjacency links. Python files
    additionally receive conservative call, import, and test links. A call
    is linked only when its simple symbol name has exactly one definition in
    the repository; ambiguous names are intentionally ignored.
    """
    chunks_by_path: dict[str, list[RepositoryChunk]] = defaultdict(list)
    chunk_by_key: dict[str, RepositoryChunk] = {}
    for chunk in chunks:
        path = str(chunk.metadata["path"])
        chunks_by_path[path].append(chunk)
        chunk_by_key[_chunk_key(repository_id, chunk)] = chunk
    for path_chunks in chunks_by_path.values():
        path_chunks.sort(key=lambda item: int(item.metadata["start_line"]))

    candidates: set[tuple[str, str, GraphRelation, str | None]] = set()
    for path_chunks in chunks_by_path.values():
        for source, target in zip(path_chunks, path_chunks[1:]):
            candidates.add(
                (
                    _chunk_key(repository_id, source),
                    _chunk_key(repository_id, target),
                    "adjacent",
                    None,
                )
            )

    trees: dict[str, ast.AST] = {}
    for path, path_chunks in chunks_by_path.items():
        if not path.endswith(".py"):
            continue
        source = _reconstruct_source(path_chunks)
        try:
            trees[path] = ast.parse(source, filename=path)
        except (SyntaxError, ValueError, TypeError):
            # A malformed or generated Python file still keeps adjacency edges.
            continue

    definitions: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for path, tree in trees.items():
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                definitions[node.name].append((path, node.lineno))

    unique_definitions = {
        name: locations[0]
        for name, locations in definitions.items()
        if len(locations) == 1
    }
    modules: dict[str, list[str]] = defaultdict(list)
    primary_modules: dict[str, str] = {}
    for path in trees:
        parts = list(Path(path).with_suffix("").parts)
        if parts and parts[-1] == "__init__":
            parts.pop()
        if not parts:
            continue
        module = ".".join(parts)
        modules[module].append(path)
        if parts[0] == "src" and len(parts) > 1:
            module = ".".join(parts[1:])
            modules[module].append(path)
        primary_modules[path] = module
    unique_modules = {
        name: paths[0] for name, paths in modules.items() if len(set(paths)) == 1
    }

    for path, tree in trees.items():
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                source_chunk = _chunk_at_line(chunks_by_path[path], node.lineno)
                if source_chunk is None:
                    continue
                source_key = _chunk_key(repository_id, source_chunk)
                symbol = _called_symbol(node.func)
                destination = unique_definitions.get(symbol) if symbol else None
                if destination:
                    target_chunk = _chunk_at_line(
                        chunks_by_path[destination[0]], destination[1]
                    )
                    if target_chunk is not None:
                        relation: GraphRelation = (
                            "tests"
                            if _is_test_path(path) and not _is_test_path(destination[0])
                            else "calls"
                        )
                        candidates.add(
                            (
                                source_key,
                                _chunk_key(repository_id, target_chunk),
                                relation,
                                symbol,
                            )
                        )
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                source_chunk = _chunk_at_line(chunks_by_path[path], node.lineno)
                if source_chunk is None:
                    continue
                source_key = _chunk_key(repository_id, source_chunk)
                module_names = _imported_modules(
                    node,
                    current_module=primary_modules.get(path, ""),
                    current_is_package=Path(path).stem == "__init__",
                )
                for module_name in module_names:
                    target_path = unique_modules.get(module_name)
                    if target_path is None:
                        continue
                    target_chunk = chunks_by_path[target_path][0]
                    candidates.add(
                        (
                            source_key,
                            _chunk_key(repository_id, target_chunk),
                            "imports",
                            module_name,
                        )
                    )

    # Remove self-links and enforce both a global and per-source bound. Ordering
    # makes a complete rebuild produce the same graph for the same source tree.
    relation_order: dict[GraphRelation, int] = {
        "tests": 0,
        "calls": 1,
        "imports": 2,
        "adjacent": 3,
    }
    ordered = sorted(
        candidates,
        key=lambda item: (
            relation_order[item[2]],
            item[0],
            item[1],
            item[3] or "",
        ),
    )
    edges: list[RepositoryGraphEdge] = []
    edge_identities: set[tuple[str, str, GraphRelation]] = set()
    per_source: dict[str, int] = defaultdict(int)
    known_keys = set(chunk_by_key)
    for source_key, target_key, relation, detail in ordered:
        if source_key == target_key:
            continue
        if source_key not in known_keys or target_key not in known_keys:
            continue
        # Detail is explanatory metadata, not part of the database key.
        edge_identity = (source_key, target_key, relation)
        if edge_identity in edge_identities:
            continue
        if per_source[source_key] >= MAX_GRAPH_EDGES_PER_CHUNK:
            continue
        edges.append(
            RepositoryGraphEdge(
                source_chunk_key=source_key,
                target_chunk_key=target_key,
                relation=relation,
                detail=detail,
            )
        )
        edge_identities.add(edge_identity)
        per_source[source_key] += 1
        if len(edges) == MAX_GRAPH_EDGES:
            break
    return edges


def _chunk_key(repository_id: int, chunk: RepositoryChunk) -> str:
    return build_chunk_key(
        repository_id,
        str(chunk.metadata["path"]),
        int(chunk.metadata["start_line"]),
        int(chunk.metadata["end_line"]),
    )


def _chunk_at_line(
    chunks: Sequence[RepositoryChunk],
    line: int,
) -> RepositoryChunk | None:
    for chunk in chunks:
        if int(chunk.metadata["start_line"]) <= line <= int(
            chunk.metadata["end_line"]
        ):
            return chunk
    return None


def _called_symbol(function: ast.expr) -> str | None:
    if isinstance(function, ast.Name):
        return function.id
    if isinstance(function, ast.Attribute):
        return function.attr
    return None


def _imported_modules(
    node: ast.Import | ast.ImportFrom,
    *,
    current_module: str,
    current_is_package: bool,
) -> tuple[str, ...]:
    if isinstance(node, ast.Import):
        return tuple(alias.name for alias in node.names)
    if not node.level:
        return (node.module,) if node.module else ()
    package_parts = current_module.split(".") if current_module else []
    if not current_is_package and package_parts:
        package_parts.pop()
    ascents = node.level - 1
    if ascents > len(package_parts):
        return ()
    if ascents:
        package_parts = package_parts[:-ascents]
    if node.module:
        package_parts.extend(node.module.split("."))
        resolved = ".".join(package_parts)
        return (resolved,) if resolved else ()
    # ``from . import pricing`` points at a child module, while
    # ``from .pricing import total`` is handled by the branch above.
    return tuple(
        ".".join([*package_parts, alias.name])
        for alias in node.names
        if alias.name != "*"
    )


def _is_test_path(path: str) -> bool:
    parts = Path(path).parts
    return "tests" in parts or Path(path).name.startswith("test_")


def _reconstruct_source(chunks: Sequence[RepositoryChunk]) -> str:
    """Rebuild a file by line number even when text windows overlap."""
    end_line = max(int(chunk.metadata["end_line"]) for chunk in chunks)
    lines = [""] * end_line
    for chunk in chunks:
        start = int(chunk.metadata["start_line"]) - 1
        for offset, line in enumerate(chunk.content.splitlines()):
            position = start + offset
            if position < len(lines) and not lines[position]:
                lines[position] = line
    return "\n".join(lines)


def parse_search_plan(response: str, *, fallback_query: str) -> SearchQueryPlan:
    """Validate the model's small JSON plan and provide a safe fallback."""
    fallback = SearchQueryPlan(fallback_query, (fallback_query,), ())
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
    raw_cross_questions = value.get("cross_questions", [])
    if not isinstance(embedding_query, str) or not isinstance(
        raw_lexical_queries, list
    ) or not isinstance(raw_cross_questions, list):
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

    cross_questions: list[str] = []
    seen_questions: set[str] = set()
    for item in raw_cross_questions:
        if not isinstance(item, str):
            continue
        cleaned = " ".join(item.split())[:300]
        key = cleaned.casefold()
        if cleaned and key not in seen_questions:
            cross_questions.append(cleaned)
            seen_questions.add(key)
        if len(cross_questions) == MAX_CROSS_QUESTIONS:
            break
    return SearchQueryPlan(
        embedding_query, tuple(lexical_queries), tuple(cross_questions)
    )
