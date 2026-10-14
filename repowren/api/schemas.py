"""Validated request and response shapes for the local HTTP API."""

from typing import Literal

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    """One validated message in a conversation.

    Example: ``{'role': 'user', 'content': 'Explain the cart service'}`` passes.
    An unknown role or empty content is rejected before inference starts.
    """

    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1, max_length=12_000)


class ChatRequest(BaseModel):
    """A bounded request sent to the local model.

    The defaults allow a useful long-form answer with low randomness. Pydantic
    rejects an empty message list, more than 32 messages, or values outside the
    declared ranges and FastAPI returns HTTP 422 automatically.
    """

    messages: list[ChatMessage] = Field(min_length=1, max_length=32)
    max_tokens: int = Field(default=2_048, ge=1, le=4_096)
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)


class StatusResponse(BaseModel):
    """State of the API and its lazily loaded model.

    The `Literal` types prevent accidental status spellings outside the small
    public set understood by clients.
    """

    api: Literal["ok"] = "ok"
    inference: Literal["not_loaded", "loading", "ready", "error"]
    model: str


class RepositoryCreate(BaseModel):
    """Register a local Git repository that RepoWren may inspect.

    Example: ``{'path': 'C:\\code\\shop', 'name': 'Shop API'}``. The name is
    optional; when omitted, the folder name becomes the display name.
    """

    path: str = Field(min_length=1, max_length=4_096)
    name: str | None = Field(default=None, min_length=1, max_length=200)


class RepositoryResponse(BaseModel):
    """Public representation of a registered repository.

    This separates the JSON returned to clients from database-specific rows.
    """

    id: int
    name: str
    root_path: str
    is_active: bool


class FileMetadataResponse(BaseModel):
    """Live path and byte size returned for one safe repository file."""

    path: str
    size_bytes: int


class FileContentResponse(BaseModel):
    """UTF-8 contents of one repository file plus the requested relative path."""

    path: str
    content: str


class SearchRequest(BaseModel):
    """A bounded hybrid source-code search.

    Example: ``{'query': 'send_order', 'max_results': 10}``. An empty query or
    result limit above 100 is rejected before the filesystem is scanned.
    """

    query: str = Field(min_length=1, max_length=500)
    max_results: int = Field(default=50, ge=1, le=100)


class SearchMatchResponse(BaseModel):
    """One hybrid-ranked indexed source chunk.

    ``score`` is the combined reciprocal-rank value, not raw cosine similarity
    or a raw Elasticsearch BM25 score. Larger values rank ahead of smaller ones.
    """

    path: str
    start_line: int
    end_line: int
    text: str
    score: float


class SearchQueryPlanResponse(BaseModel):
    """The model-produced strings used by the two retrieval backends."""

    embedding_query: str
    lexical_queries: list[str]
    cross_questions: list[str] = Field(default_factory=list)


class GraphSearchMatchResponse(BaseModel):
    """One bounded graph expansion with the retrieval seed that led to it."""

    path: str
    start_line: int
    end_line: int
    text: str
    relation: str
    detail: str | None
    seed_path: str
    seed_start_line: int
    seed_end_line: int
    depth: int = 1


class SearchResponse(BaseModel):
    """The visible query plan and its aggregate hybrid search results."""

    query_plan: SearchQueryPlanResponse
    matches: list[SearchMatchResponse]
    graph_matches: list[GraphSearchMatchResponse] = Field(default_factory=list)
