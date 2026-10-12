# RepoWren: End-to-End Developer Guide

This guide explains how RepoWren is built, how its parts communicate, and what
happens when you ask it a question about a repository. It assumes ordinary
Python and command-line experience; no knowledge of FastAPI, vLLM, or model
serving is required.

## 1. The short version

RepoWren is a local, read-only coding assistant. You select a local Git
repository, type a question, and RepoWren retrieves semantically related
20-line source chunks before sending the question to a local language model.

There are three runtime pieces:

```text
repowren-chat (terminal client)
             |
             | HTTP
             v
RepoWren FastAPI API :8000 --------> PostgreSQL :5432
             |                       with pgvector
             +----> Sentence Transformers on CPU
             |
             | OpenAI-compatible HTTP
             v
Docker vLLM server :8001 ----------> NVIDIA GPU
```

- The terminal client handles user interaction.
- The FastAPI application validates requests and coordinates the work.
- PostgreSQL stores repository registrations, source chunks, JSON metadata,
  and 384-number embedding vectors.
- vLLM loads the model and performs inference on the GPU.
- Source code stays in its original folder. PostgreSQL contains only small,
  regenerable copies of allowed chunks rather than complete files.

Phase 2 is intentionally read-only. It cannot edit files, run repository
commands or tests, or make Git changes.

## 2. Why the project is split this way

The Python application and the model server have very different needs.

The RepoWren API is a normal Python application using FastAPI, HTTPX, psycopg,
pgvector, Sentence Transformers, Pydantic, and pathspec. vLLM needs Linux, PyTorch,
CUDA, and a compatible NVIDIA runtime. Keeping vLLM in Docker prevents those
large, sensitive dependencies from leaking into the Windows Python environment.

PostgreSQL is also in Docker, but the RepoWren API itself runs on Windows from
the project's virtual environment. All three parts communicate over localhost.

The code also uses small interfaces between layers. For example, chat code
depends on `InferenceBackend`, not directly on `VLLMClient`. Tests can therefore
replace vLLM with a small fake and run without Docker or a GPU.

## 3. Top-level files and folders

```text
RepoWren/
|-- repowren/                  Main Python package
|   |-- api/                   FastAPI app, routes, and schemas
|   |-- inference/             Model interface and vLLM HTTP adapter
|   |-- persistence/           PostgreSQL implementation and migrations
|   |-- repositories/          Safe filesystem inspection and context building
|   |-- embeddings.py          Lazy CPU Sentence Transformers adapter
|   |-- services/              Chat orchestration
|   |-- cli.py                 Interactive terminal client
|   `-- config.py              Environment configuration
|-- tests/                     Offline unit and API tests
|-- benchmarks/                End-to-end inference timing probe
|-- docs/architecture.md       Short architectural overview
|-- Dockerfile.vllm            Compatibility layer over the vLLM image
|-- compose.yaml               vLLM and PostgreSQL containers
|-- pyproject.toml             Package metadata, dependencies, commands
|-- .env.example               Safe configuration template
`-- README.md                  Installation and usage instructions
```

## 4. Configuration

`repowren/config.py` defines an immutable `Settings` dataclass. When the module
loads, `python-dotenv` reads the ignored local `.env` file. Existing process
environment variables win over values in `.env`.

The main settings are:

| Variable | Meaning | Default |
| --- | --- | --- |
| `VLLM_MODEL_ID` | Hugging Face model served by vLLM | `Qwen/Qwen2.5-Coder-0.5B-Instruct` |
| `VLLM_BASE_URL` | Where the API reaches vLLM | `http://127.0.0.1:8001` |
| `REPOWREN_API_URL` | Where the terminal client reaches RepoWren | `http://127.0.0.1:8000` |
| `DATABASE_URL` | PostgreSQL connection URL | local `repowren` database on port `5432` |
| `EMBEDDING_MODEL_ID` | CPU embedding model | `sentence-transformers/all-MiniLM-L6-v2` |
| `HF_TOKEN` | Optional Hugging Face authentication | empty |

The Compose-only variables configure ports, model context length, data type,
and GPU memory use. `.env` is ignored by Git because it can contain secrets.

## 5. Docker services

`compose.yaml` defines two services.

### PostgreSQL

The `postgres` service uses `pgvector/pgvector:pg17`, publishes port `5432`,
and stores its data in the `postgres-data` volume. Its health check runs
`pg_isready`, so Docker can report whether the database is ready to accept
connections.

### vLLM

The `vllm` service:

- builds from `Dockerfile.vllm`;
- reserves one NVIDIA GPU;
- exposes container port `8000` as host port `8001`;
- persists Hugging Face and vLLM caches in named volumes;
- serves the model selected by `VLLM_MODEL_ID`;
- limits model context and GPU memory using `.env` values;
- uses vLLM's OpenAI-compatible HTTP API.

`Dockerfile.vllm` starts from the official CUDA 12.9 vLLM image and uninstalls
TorchCodec. RepoWren is text-only, and the TorchCodec binary in that image was
linked against a different CUDA version and prevented vLLM from importing.

Typical startup is:

```text
docker compose up -d postgres
docker compose up -d vllm
docker compose ps
```

The first vLLM launch may take a long time because Docker downloads the image
and vLLM downloads the model.

## 6. Python packaging and commands

`pyproject.toml` defines the package and its dependencies. Installing the
project in editable mode creates two console commands:

```text
repowren-api   -> repowren.api.app:main
repowren-chat  -> repowren.cli:main
```

Editable installation means Python imports source directly from the repository,
so normal source edits do not require reinstalling the package. A reinstall is
still needed if dependencies or console-command definitions change.

SQL files under `repowren/persistence/migrations` are declared as package data,
which makes them available through `importlib.resources` after installation.

## 7. API application lifecycle

`repowren/api/app.py` contains `create_app()`. This function makes the FastAPI
application and wires together its dependencies.

With normal production objects, it creates:

1. `VLLMClient(settings)` for inference.
2. `PostgresRepositoryStore(settings.database_url)` for metadata.
3. The FastAPI application and HTTP routes.

The objects are stored in `application.state`. Routes retrieve them from the
incoming request rather than creating a new client or store every time.

FastAPI's lifespan handler initializes PostgreSQL before accepting traffic.
Initialization creates `schema_migrations` if necessary and applies migration
files that have not already run. On shutdown, the app closes resources it owns.

`create_app()` also accepts injected inference and store objects. Tests use this
to replace Docker services with predictable in-memory fakes.

### Current Windows event-loop nuance

The repository currently uses psycopg's asynchronous connection API. psycopg
requires a selector event loop on Windows, while the installed Uvicorn version
chooses a Proactor loop by default. If startup fails with:

```text
Psycopg cannot use the 'ProactorEventLoop'
```

the database credentials are not necessarily wrong; the event-loop types are
incompatible. Until the launcher is permanently fixed, this command overrides
Uvicorn's loop factory:

```text
python -c "import asyncio,uvicorn; c=uvicorn.Config('repowren.api.app:app',host='127.0.0.1',port=8000); c.get_loop_factory=lambda: asyncio.SelectorEventLoop; uvicorn.Server(c).run()"
```

## 8. HTTP schemas and routes

`repowren/api/schemas.py` contains Pydantic models. They validate data before
it reaches the filesystem, database, or model. Bounds include message count,
message length, output token count, temperature, path length, search length,
and search-result count.

`repowren/api/routes.py` is the HTTP boundary. It obtains shared services from
`app.state`, performs small coordination steps, and returns response models.

| Route | Purpose |
| --- | --- |
| `GET /health` | Confirms that the API process is alive. |
| `GET /status` | Checks vLLM and reports its status and model. |
| `POST /v1/chat/stream` | Streams chat without repository context. |
| `POST /v1/repositories` | Registers and scans a local Git repository. |
| `GET /v1/repositories` | Lists registered repositories. |
| `POST /v1/repositories/{id}/select` | Selects one repository. |
| `GET /v1/repositories/{id}/files` | Lists safe paths and byte sizes. |
| `GET /v1/repositories/{id}/file` | Reads one UTF-8 file. |
| `POST /v1/repositories/{id}/search` | Semantically searches stored chunks. |
| `POST /v1/repositories/{id}/chat/stream` | Streams repository-aware chat. |

Filesystem scanning, file reading, and CPU embedding are blocking operations,
so they run through `asyncio.to_thread()`. PostgreSQL search and context
assembly remain async and do not reopen every source file.

Expected repository errors become HTTP `422` responses, missing repository IDs
become `404`, and PostgreSQL failures become `503`.

## 9. Repository registration flow

When the user enters:

```text
/repo add C:\Users\you\Desktop\my-project
```

the following happens:

1. `repowren/cli.py` recognizes a slash command.
2. It sends `POST /v1/repositories` with the folder path.
3. `RepositoryService.validate_repository_root()` expands and resolves the
   path, requires an existing directory, and checks for a `.git` entry.
4. `PostgresRepositoryStore.register()` inserts the repository or updates its
   name if that root was already registered.
5. If no repository is active yet, the new one becomes active.
6. `RepositoryService._read_chunk_inputs()` walks safe text files and splits
   each into fixed, non-overlapping 20-line chunks.
7. `SentenceTransformerEmbeddings.encode()` creates one normalized vector per
   chunk on CPU. The model is lazy-loaded on the first use.
8. `replace_chunks()` deletes the repository's old chunks and inserts the new
   content, JSON metadata, and vectors in one transaction.
9. The CLI remembers the returned repository ID for later commands and chat.

Registration requires Git because a repository root is a clearer safety
boundary than an arbitrary folder.

## 10. Filesystem safety

`repowren/repositories/service.py` is the main filesystem trust boundary.

### Canonical paths

Repository roots and requested files are resolved to their real absolute paths.
Absolute file requests and paths containing `..` are rejected. After resolving
a file, RepoWren checks that it is still under the registered root.

This blocks ordinary path traversal and symlink escapes.

### Safe scanning

Directory walks do not follow symlinks. The scanner respects the root
`.gitignore` and adds built-in exclusions for:

- `.git`, virtual environments, editor folders, and Python caches;
- `node_modules`, `build`, `dist`, and model directories;
- `.env` files;
- compiled Python files;
- `.gguf`, `.safetensors`, and `.bin` model/binary files.

Files over 1 MB and files whose first bytes contain a null byte are excluded.
The scan is capped at 1,000 files. Searches return at most 50 chunks by default.
Direct `/read` reapplies the same built-in and `.gitignore` rules after path
containment checks, so `.env`, `.git/config`, ignored files, binary files, and
symlink escapes are rejected even when their exact path is supplied.

## 11. File listing, reading, and search

### Listing

`list_files()` rescans the live repository, optionally filters results by a
case-insensitive path substring, and returns only relative path and byte size.

For example, `/files src` means “show safe files whose path contains `src`.” It
does not mean “recursively list a folder named `src`.” If no path contains that
text, the result is empty.

### Reading

`read_file()` resolves a relative path, caps it at 200,000 bytes, and decodes
it as UTF-8. A `/read` command accepts only the path:

```text
/read project_05_consumer_groups/producer.py
```

A question must be entered as the next prompt. If extra words are placed after
the path, the CLI treats all of them as part of the filename and the API returns
`422` because that file does not exist.

### Searching

`search_code()` embeds the query and asks PostgreSQL for the closest stored
vectors using exact cosine distance. A result contains path, start line, end
line, source text, and similarity score. It is semantic search, not regex or
AST search, and intentionally has no approximate vector index yet.

## 12. How repository-aware context is built

When a normal chat prompt is sent with an active repository, the CLI uses:

```text
POST /v1/repositories/{id}/chat/stream
```

The route finds the latest user message and calls `build_context()`.

Context building works like this:

1. Embed the newest user question.
2. Ask pgvector for the five closest chunks in the selected repository.
3. Format each as `### path lines start-end` followed by its stored source.
4. Join the chunks and cap total source context at 9,000 characters.

The route adds this context as a system message before the original
conversation and then starts normal chat streaming.

There is no stop-word list, keyword loop, deduplication pass, or second file
read. Question and code can use different wording as long as the embedding
model places their meanings near each other.

## 13. PostgreSQL persistence

`RepositoryStore` is a Python protocol describing the persistence operations
the rest of the application needs. `PostgresRepositoryStore` implements it with
psycopg.

The migration creates:

- `schema_migrations`, which records applied migration filenames;
- `repositories`, which stores names, canonical roots, timestamps, and the
  active flag;
- `repository_chunks`, which stores source content, JSONB metadata, and
  `VECTOR(384)` embeddings.

A partial unique index allows at most one repository with `is_active = TRUE`.
Selecting a repository clears the old active flag and sets the requested one.

The store currently opens short-lived connections for each operation rather
than using a pool. This keeps the first implementation small but adds connection
overhead under heavier usage.

Registration performs a full rebuild. Listing and direct reads inspect the live
filesystem; search and repository chat use the stored snapshot. Re-register a
path after changing source so its semantic index matches disk contents.

## 14. Inference and streaming

`repowren/inference/base.py` defines `InferenceBackend`, the interface needed by
chat code:

- a model ID;
- a readiness status;
- `check_ready()`;
- `stream_chat()`;
- `close()`.

`VLLMClient` implements this interface. It first calls vLLM's `/health`. When
ready, it posts OpenAI-style messages to `/v1/chat/completions` with streaming
enabled. It parses `data:` lines, extracts text deltas, and yields them one at a
time. Network and protocol problems become `InferenceError` exceptions.

`ChatService` converts inference output into RepoWren's simpler newline-delimited
JSON event format:

```json
{"type":"status","message":"Waiting for vLLM model server..."}
{"type":"token","text":"Hello"}
{"type":"token","text":" world"}
{"type":"done"}
```

The API uses `StreamingResponse`, so the client can print text before the full
answer is complete.

## 15. Terminal client behavior

`repowren/cli.py` maintains two pieces of local session state:

- the conversation messages;
- the active repository ID.

Supported commands are:

```text
/repo add C:\path\to\repo       Register and select a Git repository
/repo list                       Show repositories and the active one
/repo use 2                      Select repository ID 2
/files                           List safe files
/files project_05                Filter file paths
/read path\to\file.py           Print one file
/search producer                 Search source lines
/help                            Show help
/exit                            Quit
```

Slash commands call repository endpoints and do not enter the chat history.
Ordinary text is appended as a user message, streamed from the appropriate chat
endpoint, and followed by the completed assistant message.

The conversation exists only in the CLI process. Restarting `repowren-chat`
starts a new conversation, although repository registrations remain in
PostgreSQL.

## 16. Testing strategy

The test suite is designed to run without Docker, PostgreSQL, a GPU, or a model:

- chat API tests inject a fake inference backend;
- repository API tests inject an in-memory store;
- CLI tests use `httpx.MockTransport`;
- repository tests create temporary Git-shaped folders;
- vLLM-client tests fake HTTP responses and streaming data.

The repository tests cover exclusions, chunk metadata, semantic retrieval,
context building, and path traversal. Hardware validation remains separate because it depends on
Docker Desktop, model downloads, CUDA, and the local GPU.

Run offline tests with:

```text
.venv\Scripts\python.exe -m pytest -q
```

## 17. A complete manual session

Start the services:

```text
docker compose up -d postgres
docker compose up -d vllm
```

Start the API in one terminal. The `repowren-api` entry point automatically
selects the Windows event loop required by async psycopg.

Start the client in another terminal:

```text
.venv\Scripts\activate.bat
repowren-chat
```

Then try:

```text
/repo add C:\Users\you\Desktop\some-git-repository
/repo list
/files
/files src
/search important_function
/read path\to\module.py
Explain how important_function works and mention the relevant files.
/exit
```

## 18. Current boundaries and likely next steps

RepoWren currently has no:

- file-edit or patch workflow;
- command or test runner;
- Git mutation support;
- approval system for risky actions;
- Tree-sitter parsing;
- multi-agent orchestration;
- persistent conversation history.

The existing boundaries make those features easier to add safely. Semantic
retrieval can later gain language-aware chunking or a vector index without changing the CLI.
An approval-controlled editing service can be added without granting the model
general shell access. Git operations can remain a separate, explicitly approved
layer.

The key architectural idea is simple: the model is only one component. FastAPI
owns validation and coordination, the repository service owns filesystem safety,
PostgreSQL owns chunks and vectors, and Docker vLLM owns GPU inference.
