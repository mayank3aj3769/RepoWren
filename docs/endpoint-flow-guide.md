# RepoWren Endpoint Flow Guide

This guide traces every HTTP endpoint in the current RepoWren project. It
explains which module and function runs next, what object is passed between
layers, and what the caller receives.

The examples use a tiny imaginary repository so the flow stays concrete.

## 1. The dummy repository used in this guide

Imagine this folder exists on the computer:

```text
C:\demo\bookshop
|-- .git\
|-- .gitignore
|-- README.md
|-- .env
|-- src\
|   |-- pricing.py
|   `-- api.py
|-- tests\
|   `-- test_pricing.py
`-- dist\
    `-- generated.js
```

`src/pricing.py` contains:

```python
def calculate_total(prices):
    return sum(prices)
```

The `.gitignore` contains:

```text
dist/
```

When this repository is registered, RepoWren should scan files such as
`README.md`, `src/pricing.py`, and `tests/test_pricing.py`. It should skip:

- `.git`, because Git internals are not source context;
- `.env`, because the built-in rules treat it as sensitive;
- `dist/generated.js`, because the repository's `.gitignore` excludes it.

Throughout this guide, assume PostgreSQL assigns the repository ID `7`.

## 2. The layers involved in a request

Most requests move through these layers:

```text
HTTP client
   |
   v
Pydantic schema validation
   |
   v
FastAPI route
   |
   +------> RepositoryService ------> local files
   |                 |------> Sentence Transformers on CPU
   |                 `-------------> RepositoryStore ------> PostgreSQL + pgvector
   |
   `------> ChatService ------> VLLMClient ------> Docker vLLM/GPU
```

Each layer has one main responsibility.

### `repowren/api/schemas.py`

Pydantic models validate incoming JSON and define outgoing JSON shapes.
FastAPI runs this validation before the route function. Bad input normally
produces HTTP `422` without calling repository or inference code.

### `repowren/api/routes.py`

Routes form the HTTP boundary. They load shared dependencies from
`request.app.state`, call a service/store, translate known exceptions to HTTP
errors, and return response objects.

### `repowren/repositories/service.py`

`RepositoryService` owns filesystem behavior, 20-line chunk creation, embedding
coordination, semantic search, and source-context construction.

### `repowren/embeddings.py`

`SentenceTransformerEmbeddings` lazy-loads `all-MiniLM-L6-v2` on CPU. Its
protocol is intentionally tiny, so tests can replace it with deterministic
vectors and never download a model.

### `repowren/persistence/postgres.py`

`PostgresRepositoryStore` stores repository rows and semantic chunks. Each
chunk has source text, JSONB metadata, and a `VECTOR(384)` value.

### `repowren/services/chat.py`

`ChatService` turns inference tokens into RepoWren's newline-delimited JSON
event format.

### `repowren/inference/vllm.py`

`VLLMClient` checks model readiness and translates RepoWren messages into
vLLM's OpenAI-compatible HTTP format.

## 3. Application startup before any endpoint runs

The application begins in `repowren/api/app.py`.

### `create_app()`

In a normal run, this function creates:

```python
settings = Settings.from_environment()
inference = VLLMClient(settings)
repository_store = PostgresRepositoryStore(settings.database_url)
embeddings = SentenceTransformerEmbeddings(settings.embedding_model_id)
repository_service = RepositoryService(repository_store, embeddings)
```

The objects are stored on the FastAPI application:

```python
application.state.inference = inference
application.state.repository_store = repository_store
application.state.repository_service = repository_service
```

This means all requests share the same `VLLMClient` and store object. A route
does not reconstruct application dependencies from scratch.

Tests can pass fake objects into `create_app()`. That is why API tests can run
without Docker, PostgreSQL, or a GPU.

### The lifespan function

Before accepting requests, the lifespan handler calls:

```python
await repository_store.initialize()
```

`PostgresRepositoryStore.initialize()`:

1. opens a database connection;
2. creates `schema_migrations` if it does not exist;
3. finds packaged `.sql` files;
4. checks whether each filename is already recorded;
5. runs new migrations;
6. records successfully applied filenames.

For example, after applying `001_repositories.sql`, PostgreSQL contains:

```text
schema_migrations
  version = "001_repositories.sql"
```

Migration `001` creates the original repository tables. Migration `002` enables
the pgvector extension, removes the obsolete file-hash table, and creates
`repository_chunks`. Applied filenames make both migrations one-time actions.

On Windows, the `repowren-api` entry point gives Uvicorn a Selector event-loop
factory because async psycopg cannot run on Uvicorn's default Proactor loop.

On shutdown, the lifespan handler closes resources owned by the app.

## 4. Shared route helpers

Several endpoints use the same small helpers in `repowren/api/routes.py`.

### `_inference(request)`

Returns the shared `InferenceBackend` from `request.app.state.inference`.

### `_store(request)`

Returns the shared `RepositoryStore` from
`request.app.state.repository_store`.

### `_repository_response(record)`

Converts an internal object:

```python
RepositoryRecord(
    id=7,
    name="bookshop",
    root_path="C:\\demo\\bookshop",
    is_active=True,
)
```

into a Pydantic `RepositoryResponse`, which FastAPI serializes as:

```json
{
  "id": 7,
  "name": "bookshop",
  "root_path": "C:\\demo\\bookshop",
  "is_active": true
}
```

### `_repository_or_404(store, repository_id)`

Calls `store.get(repository_id)`.

- If the row exists, the helper returns a `RepositoryRecord`.
- If the ID does not exist, it raises HTTP `404`.
- If PostgreSQL fails, it raises HTTP `503`.

This prevents every file/search/chat route from repeating the same lookup and
error handling.

## 5. `GET /health`

### Purpose

Answers one narrow question: “Is the RepoWren API process running?”

It does not check PostgreSQL or vLLM.

### Code path

```text
GET /health
  -> repowren.api.app.health()
  -> HealthResponse(status="ok")
  -> JSON response
```

### Response

```json
{"status": "ok"}
```

This endpoint is intentionally simple. If `/health` works while `/status`
fails, FastAPI is alive but the model server may not be ready.

## 6. `GET /status`

### Purpose

Checks the configured vLLM server and reports model readiness.

### Code path

```text
GET /status
  -> routes.status_endpoint(request)
  -> _inference(request)
  -> VLLMClient.check_ready()
  -> GET http://127.0.0.1:8001/health
  -> StatusResponse(...)
```

### What `check_ready()` does

- HTTP `200` sets status to `ready`.
- HTTP `202` or `503` sets status to `loading`.
- Another HTTP status sets status to `error`.
- A network failure sets status to `not_loaded`.

### Example response

```json
{
  "api": "ok",
  "inference": "ready",
  "model": "Qwen/Qwen2.5-Coder-0.5B-Instruct"
}
```

No generation occurs here. It is only a readiness probe.

## 7. `POST /v1/chat/stream`

### Purpose

Runs ordinary chat without repository context.

### Example request

```json
{
  "messages": [
    {"role": "user", "content": "What is a Kafka consumer group?"}
  ],
  "max_tokens": 256,
  "temperature": 0.2
}
```

### Step 1: Pydantic validation

FastAPI creates a `ChatRequest` from the JSON. Validation includes:

- at least one and at most 32 messages;
- only `system`, `user`, or `assistant` roles;
- non-empty message content up to 12,000 characters;
- `max_tokens` from 1 through 1,024;
- `temperature` from 0.0 through 2.0.

### Step 2: Route and service

```text
routes.stream_chat()
  -> _inference(request)
  -> ChatService(inference)
  -> ChatService.stream_events(body)
```

The route wraps the generator in FastAPI's `StreamingResponse` with media type
`application/x-ndjson`.

### Step 3: Inference

`ChatService.stream_events()` calls:

```python
inference.stream_chat(
    request.messages,
    max_tokens=request.max_tokens,
    temperature=request.temperature,
)
```

`VLLMClient.stream_chat()` first checks `/health`, then creates a payload like:

```python
{
    "model": "Qwen/Qwen2.5-Coder-0.5B-Instruct",
    "messages": [
        {"role": "user", "content": "What is a Kafka consumer group?"}
    ],
    "max_tokens": 256,
    "temperature": 0.2,
    "stream": True,
}
```

It posts this to vLLM's `/v1/chat/completions` endpoint.

### Step 4: Streaming response

Suppose vLLM produces `"A consumer"` and `" group..."`. `ChatService` turns
those pieces into separate JSON lines:

```json
{"type": "token", "text": "A consumer"}
{"type": "token", "text": " group..."}
{"type": "done"}
```

The client can print each token before the whole answer is finished.

## 8. `POST /v1/repositories`

### Purpose

Registers a local Git repository and stores a metadata snapshot of its safe
files.

### Example request

```json
{
  "path": "C:\\demo\\bookshop",
  "name": "Book Shop"
}
```

`name` is optional. If omitted, `bookshop` is taken from the folder name.

### Step 1: Request validation

Pydantic creates:

```python
RepositoryCreate(path="C:\\demo\\bookshop", name="Book Shop")
```

An empty path or overly long input is rejected before the route runs.

### Step 2: Route

```text
routes.register_repository()
  -> RepositoryService(store)
  -> service.register(body.path, body.name)
```

### Step 3: Root validation

`validate_repository_root()`:

1. expands `~` if present;
2. resolves the path to its real absolute location;
3. requires the path to exist;
4. requires it to be a directory;
5. requires a `.git` entry.

For example:

```text
C:\demo\something\..\bookshop
```

resolves to:

```text
C:\demo\bookshop
```

This canonical path becomes the repository boundary used by later safety
checks.

### Step 4: Store the repository

`PostgresRepositoryStore.register()` performs an upsert:

```sql
INSERT INTO repositories (name, root_path)
...
ON CONFLICT (root_path)
DO UPDATE SET name = EXCLUDED.name
RETURNING id, name, root_path, is_active
```

The unique root path prevents duplicate rows for the same folder. The first
registered repository is automatically made active.

The returned database row is converted into:

```python
RepositoryRecord(
    id=7,
    name="Book Shop",
    root_path="C:\\demo\\bookshop",
    is_active=True,
)
```

### Step 5: Scan safe files

`RepositoryService.register()` reads safe files in a worker thread because
directory walking and file reads are blocking disk work.

`_iter_safe_files()` uses `os.walk()` with `followlinks=False`. At the dummy
root, an early walk item may look conceptually like:

```python
directory = "C:\\demo\\bookshop"
directory_names = [".git", "src", "tests", "dist"]
file_names = [".gitignore", "README.md", ".env"]
```

The function:

- removes `.git` and `dist` from future traversal;
- skips symlinks;
- skips `.env`, `.gitignore`, and repository `.gitignore` matches;
- skips files over 1 MB;
- skips likely binary files containing a null byte near the beginning;
- yields safe files in stable sorted order.

### Step 6: Build 20-line chunks

`_read_chunk_inputs()` reads each allowed UTF-8 file and slices its line list
with offsets `0, 20, 40, ...`. A 25-line file therefore produces metadata like:

```python
{"path": "src/pricing.py", "start_line": 1, "end_line": 20}
{"path": "src/pricing.py", "start_line": 21, "end_line": 25}
```

### Step 7: Embed and save the chunks

The service passes all chunk texts to `SentenceTransformerEmbeddings.encode()`.
The lazy-loaded CPU model returns one normalized 384-number vector per text.
`PostgresRepositoryStore.replace_chunks()` then:

1. deletes old chunk rows for repository `7`;
2. inserts each chunk's content, JSONB metadata, and vector;
3. commits deletion and insertion together.

This is a full rebuild. Re-registering the same path is the simple current way
to refresh the semantic index after source changes.

### Response

```json
{
  "id": 7,
  "name": "Book Shop",
  "root_path": "C:\\demo\\bookshop",
  "is_active": true
}
```

## 9. `GET /v1/repositories`

### Purpose

Lists registered repositories.

### Code path

```text
routes.list_repositories()
  -> _store(request)
  -> PostgresRepositoryStore.list_repositories()
  -> SELECT ... ORDER BY is_active DESC, name, id
  -> _repository_response() for each row
```

The active repository appears first.

### Example response

```json
[
  {
    "id": 7,
    "name": "Book Shop",
    "root_path": "C:\\demo\\bookshop",
    "is_active": true
  },
  {
    "id": 9,
    "name": "Other Project",
    "root_path": "C:\\demo\\other",
    "is_active": false
  }
]
```

## 10. `POST /v1/repositories/{id}/select`

### Purpose

Marks one registered repository as active in PostgreSQL.

### Example

```text
POST /v1/repositories/9/select
```

### Code path

```text
routes.select_repository(repository_id=9)
  -> store.select(9)
  -> confirm repository 9 exists
  -> set current active row to false
  -> set repository 9 to true
  -> return updated RepositoryRecord
```

The two updates use the same connection context and commit as one transaction.
A partial unique database index guarantees that at most one row can be active.

An unknown ID returns `404`.

The CLI also remembers its selected repository ID locally. The database active
flag and CLI variable usually agree, but repository-aware chat chooses the ID
present in the CLI endpoint URL rather than silently looking up the active row.

## 11. `GET /v1/repositories/{id}/files`

### Purpose

Returns current safe file paths and byte sizes, optionally filtered by path text.

### Examples

```text
GET /v1/repositories/7/files
GET /v1/repositories/7/files?query=src
```

### Code path

```text
routes.list_repository_files()
  -> _repository_or_404(store, 7)
  -> asyncio.to_thread(service.list_files, repository, query="src")
  -> service.scan_files(repository)
  -> service._iter_safe_files(root)
  -> filter FileMetadata.path values
  -> FileMetadataResponse objects
```

The endpoint scans the live filesystem again rather than reading the saved
metadata snapshot. This makes the list reflect files changed after registration.

`query` is a case-insensitive substring of the relative path. It is not file
content search.

With `query=src`, the dummy result is:

```json
[
  {
    "path": "src/api.py",
    "size_bytes": 140
  },
  {
    "path": "src/pricing.py",
    "size_bytes": 65
  }
]
```

## 12. `GET /v1/repositories/{id}/file`

### Purpose

Reads one bounded UTF-8 file.

### Example

```text
GET /v1/repositories/7/file?path=src/pricing.py
```

### Code path

```text
routes.read_repository_file()
  -> _repository_or_404(store, 7)
  -> asyncio.to_thread(service.read_file, repository, "src/pricing.py")
  -> service._resolve_file(root, relative_path)
  -> built-in and .gitignore exclusion check
  -> size check
  -> binary check
  -> UTF-8 decode
  -> FileContentResponse
```

### Path validation

`_resolve_file()` rejects:

- absolute requested paths;
- paths containing `..`;
- missing paths;
- directories instead of files;
- resolved targets outside the registered root.

The final resolved containment check also blocks a symlink whose visible path
is inside the repository but whose target is outside it.

### Example response

```json
{
  "path": "src/pricing.py",
  "content": "def calculate_total(prices):\n    return sum(prices)\n"
}
```

Direct reading reapplies the same exclusion matcher used by indexing. Supplying
an exact `.env`, `.git/config`, ignored file, binary file, oversized file, or
symlink escape therefore produces HTTP `422` rather than returning contents.

## 13. `POST /v1/repositories/{id}/search`

### Purpose

Performs bounded semantic search over the stored repository chunks.

### Example request

```json
{
  "query": "calculate_total",
  "max_results": 10
}
```

Pydantic requires non-empty query text, caps it at 500 characters, and allows
`max_results` from 1 through 100.

### Code path

```text
routes.search_repository()
  -> _repository_or_404(store, 7)
  -> await service.search_code(repository, query, max_results=10)
  -> embed the query on a worker thread
  -> store.search_chunks(repository_id, query_vector, 10)
  -> PostgreSQL exact cosine-distance query
  -> SearchMatch objects
  -> SearchMatchResponse objects
```

Inside `search_code()`, whitespace-only queries are rejected. The query becomes
one vector, and PostgreSQL orders rows with `embedding <=> query_vector`. This
is exact cosine search with no HNSW or IVFFlat index in the current simple version.

For the dummy repository, the internal object is similar to:

```python
SearchMatch(
    path="src/pricing.py",
    start_line=1,
    end_line=2,
    text="def calculate_total(prices):\n    return sum(prices)",
    score=0.91,
)
```

The response is:

```json
[
  {
    "path": "src/pricing.py",
    "start_line": 1,
    "end_line": 2,
    "text": "def calculate_total(prices):\n    return sum(prices)",
    "score": 0.91
  }
]
```

The embedding model can connect related wording, although it does not parse the
program like Python or Tree-sitter would.

## 14. `POST /v1/repositories/{id}/chat/stream`

### Purpose

Builds a small amount of source context and then runs the normal streaming chat
flow.

### Example request

```json
{
  "messages": [
    {
      "role": "user",
      "content": "How does calculate_total work?"
    }
  ]
}
```

### Step 1: Load the repository

`stream_repository_chat()` calls `_repository_or_404(store, 7)` and receives:

```python
RepositoryRecord(
    id=7,
    name="Book Shop",
    root_path="C:\\demo\\bookshop",
    is_active=True,
)
```

### Step 2: Find the newest user message

The route scans `body.messages` backward and selects the latest message whose
role is `user`. This matters in a multi-turn conversation containing both user
and assistant messages.

If no user message exists, the route returns HTTP `422`.

### Step 3: Retrieve five semantic chunks

`RepositoryService.build_context()` sends the entire newest user question to
`search_code(..., max_results=5)`. There is no regular expression, stop-word
list, per-term search loop, or deduplication pass.

For the dummy repository, one result is:

```python
SearchMatch(
    path="src/pricing.py",
    start_line=1,
    end_line=2,
    text="def calculate_total(prices):\n    return sum(prices)",
    score=0.91,
)
```

### Step 4: Format the stored chunks

The source is already in PostgreSQL, so the function does not reopen files. An
excerpt becomes:

```text
### src/pricing.py lines 1-2
def calculate_total(prices):
    return sum(prices)
```

The five excerpts together are capped at 9,000 characters.

### Step 6: Build the system message

The final context begins with instructions such as:

```text
You are answering a question about the registered repository 'Book Shop'.
Use the source excerpts below. Cite file paths and line numbers when possible.
```

The route creates a copy of the request with a new first message:

```python
[
    ChatMessage(
        role="system",
        content="<instructions plus src/pricing.py excerpt>",
    ),
    ChatMessage(
        role="user",
        content="How does calculate_total work?",
    ),
]
```

The user's original request object is not mutated.

### Step 7: Stream through normal chat

The enriched `ChatRequest` goes through:

```text
ChatService.stream_events()
  -> VLLMClient.stream_chat()
  -> vLLM /v1/chat/completions
  -> token events
  -> done event
```

The model only receives selected excerpts, not the entire repository.

## 15. How CLI commands map to endpoints

`repowren/cli.py` is an HTTP client. It does not access PostgreSQL, repository
files, or vLLM directly.

| CLI input | HTTP request |
| --- | --- |
| `/repo add C:\demo\bookshop` | `POST /v1/repositories` |
| `/repo list` | `GET /v1/repositories` |
| `/repo use 7` | `POST /v1/repositories/7/select` |
| `/files` | `GET /v1/repositories/7/files` |
| `/files src` | `GET /v1/repositories/7/files?query=src` |
| `/read src/pricing.py` | `GET /v1/repositories/7/file?path=src/pricing.py` |
| `/search calculate_total` | `POST /v1/repositories/7/search` |
| Ordinary text with repository selected | `POST /v1/repositories/7/chat/stream` |
| Ordinary text without repository selected | `POST /v1/chat/stream` |

The CLI keeps conversation messages and its selected repository ID in memory.
Repository registrations persist in PostgreSQL, but chat history disappears
when the CLI process exits.

`/read` accepts only a file path. This input is incorrect:

```text
/read src/pricing.py How does this function work?
```

Everything after `/read ` becomes part of the requested path. Use two inputs:

```text
/read src/pricing.py
How does this function work?
```

## 16. Common error flow

### Pydantic validation error: HTTP 422

Examples include empty messages, invalid roles, empty search text, and overly
large result limits. FastAPI rejects these before a route does real work.

### Repository access error: HTTP 422

Examples include a non-Git root, `../` traversal, missing file, oversized file,
or invalid UTF-8.

### Missing repository: HTTP 404

The requested database ID does not exist.

### Store failure: HTTP 503

PostgreSQL could not complete the operation.

### Inference failure: streamed `error` event

`VLLMClient` raises `InferenceError`; `ChatService` converts it into an event:

```json
{"type": "error", "message": "vLLM request failed: ..."}
```

Because chat is already a streaming response, inference errors are represented
inside the stream rather than as a new HTTP error status.

## 17. Where the project is now

The current project can:

- run a small coding model locally through Docker vLLM;
- stream ordinary chat;
- persist local repository registrations in PostgreSQL;
- select among registered repositories;
- list and safely read repository files;
- semantically search pgvector chunks and build source context for chat;
- enforce repository-root containment and bounded results;
- run offline tests using fake stores and inference backends.

It does not yet include:

- file editing or patches;
- command/test execution;
- approval-controlled actions;
- Git commits, branches, or pushes;
- Tree-sitter code parsing;
- persistent chat history;
- multi-agent orchestration.

## 18. Important current technical limitations

### Windows Uvicorn event loop

psycopg's async driver needs a selector event loop on Windows. `repowren-api`
now supplies Uvicorn with `selector_loop_factory`, so the normal command uses a
compatible loop without a manual one-line launcher.

### Index refresh is manual and complete

Search and repository chat use the PostgreSQL snapshot. There is no file watcher
or incremental hash comparison. Register the same repository path again after
editing source to rebuild all of its chunks.

### Search is intentionally exact and simple

There is no approximate vector index, language-aware parser, overlap between
chunks, or reranking. Those can be introduced later when measurements justify
their extra complexity.

## 19. The central idea to remember

RepoWren is not one large “agent” function. It is a set of small boundaries:

- Pydantic decides whether request data is structurally valid.
- FastAPI routes coordinate HTTP behavior.
- `RepositoryService` controls filesystem access and source selection.
- `RepositoryStore` and PostgreSQL own persistent chunks and vectors.
- `ChatService` owns RepoWren's stream event format.
- `VLLMClient` owns model-server communication.
- Docker vLLM owns model loading and GPU inference.

Following one endpoint through these boundaries is the easiest way to understand
the project and the safest way to extend it.
