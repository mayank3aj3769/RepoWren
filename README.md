# RepoWren

RepoWren is a lightweight, local-first coding-agent foundation. A terminal
client and a Next.js web UI talk to a small FastAPI backend, and the backend
streams generated text from a separate vLLM server running in Docker. PostgreSQL with pgvector stores
registered repositories, source chunks, metadata, and embeddings. Elasticsearch
stores a second, text-only copy of those chunks for exact words and identifiers.

```text
terminal / Next.js :3000 -> RepoWren API :8000 -> Docker vLLM :8001 -> local NVIDIA GPU
                                                -> CPU Sentence Transformers
                                                -> Docker PostgreSQL + pgvector :5432
                                                -> Docker Elasticsearch :9200
```

This milestone adds read-only repository awareness: safe file discovery,
reading, semantic search, and bounded source context. Code editing, command
execution, and Git mutations remain intentionally out of scope.

## Requirements

- Windows with Docker Desktop running Linux containers.
- Docker Desktop GPU support enabled with an NVIDIA driver and WSL 2 backend.
- Python 3.12 or newer for the RepoWren API and terminal client.
- Node.js 20.9 or newer for the Next.js web UI.
- Enough storage for the vLLM image and the selected Hugging Face model.

The default model is `Qwen/Qwen2.5-Coder-0.5B-Instruct`. It is intentionally
small enough for initial validation on a GTX 1650 with 4 GB VRAM. Larger models
are outside the current milestone.

## Configure RepoWren

Create the ignored local environment file:

```text
copy .env.example .env
```

Set `HF_TOKEN` only when the model repository requires authentication. The
default public model does not require one. Important settings include:

```dotenv
VLLM_IMAGE=vllm/vllm-openai:v0.31.0-cu129
VLLM_MODEL_ID=Qwen/Qwen2.5-Coder-0.5B-Instruct
VLLM_BASE_URL=http://127.0.0.1:8001
VLLM_PORT=8001
REPOWREN_API_URL=http://127.0.0.1:8000
DATABASE_URL=postgresql://repowren:repowren@127.0.0.1:5432/repowren
EMBEDDING_MODEL_ID=sentence-transformers/all-MiniLM-L6-v2
ELASTICSEARCH_URL=http://127.0.0.1:9200
ELASTICSEARCH_INDEX=repowren-chunks
ELASTICSEARCH_PORT=9200
```

Secrets, model weights, caches, and virtual environments are excluded from Git.

## Install the API and terminal client

```text
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m pip check
python -m pytest -q
```

Sentence Transformers runs on CPU so it does not compete with vLLM for GPU
memory. The first repository registration or search downloads
`all-MiniLM-L6-v2`; later runs reuse the normal Hugging Face cache.

## Start vLLM in Docker

Start the container:

```text
docker compose up -d
```

The first start pulls the official CUDA 12.9 image, builds a tiny local layer,
and downloads the model into Docker-managed volumes. The local layer removes
TorchCodec because RepoWren serves text only and the current TorchCodec binary
in this image expects CUDA 13. Follow progress or inspect container state with:

```text
docker compose logs -f vllm
docker compose ps
docker compose down
docker compose build --pull vllm
```

The Compose file reserves one NVIDIA GPU, persists Hugging Face and vLLM
compile caches, starts PostgreSQL with pgvector, and starts a single-node
Elasticsearch service with a 512 MB heap. PostgreSQL and Elasticsearch both
have health checks and named data volumes. Elasticsearch security is disabled
only for this local development setup.

Check vLLM directly once it finishes loading:

```text
curl http://127.0.0.1:8001/health
curl http://127.0.0.1:8001/v1/models
```

## Start RepoWren

In a second terminal:

```text
.venv\Scripts\activate
repowren-api
```

Check both the API and its connection to vLLM:

```text
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/status
```

## Chat from the terminal

In a third terminal with the virtual environment activated:

```text
repowren-chat
```

The client keeps the current conversation in memory and streams model output as
it arrives. Register a local Git repository, inspect it, then ask a question:

```text
/repo add C:\path\to\your-project
/repo list
/files
/search function_name
/read src/example.py
How does function_name work?
```

Registering reads allowed UTF-8 files once, splits them into fixed 20-line
chunks, creates CPU embeddings, and replaces that repository's PostgreSQL rows.
It then replaces the matching text-only Elasticsearch documents. PostgreSQL is
written first; if Elasticsearch fails, the API reports the failure instead of
silently using only semantic results. Register the same path again to rebuild
both snapshots and repair either index.

Search first asks the local vLLM model to turn the user's request into one
semantic query and up to eight short exact-text strings such as identifiers,
error text, or configuration keys. PostgreSQL embeds the semantic query and
finds conceptually similar code with exact pgvector cosine search, while
Elasticsearch searches the exact-text strings with boosted phrase matches and
BM25. RepoWren asks each backend for 20 candidates concurrently and uses
Reciprocal Rank Fusion (RRF) to combine their ranks. Raw cosine and BM25 scores
are never compared. The terminal labels the aggregate hybrid score, and
repository chat sends the best five hybrid chunks to vLLM. If vLLM is starting,
temporarily unavailable, or emits malformed planner JSON, search safely falls
back to the original user query instead of returning a planner-related 503.
Compose starts vLLM with a 4,096-token context window by default. If a prompt
still exceeds the model limit, the adapter retries once with bounded source
context and a shorter completion instead of failing the chat request outright.
Type `/help` for commands or `/exit` to finish. Registration accepts only an
existing Git repository. RepoWren ignores `.gitignore` entries, secrets such as
`.env`, binary files, generated directories, model weights, files over 1 MB,
and paths that escape the repository root.

## Use the web UI

The local Next.js application mirrors the terminal workflow with repository
registration and selection, streamed chat, visible hybrid-search query plans,
RRF-ranked search results, and safe file browsing. Install and start it after
the RepoWren API is running:

```text
cd web
npm install
npm run dev
```

Open `http://127.0.0.1:3000`. Next.js proxies `/api/repowren/*` to
`http://127.0.0.1:8000`, so the browser does not need cross-origin access. Set
`REPOWREN_API_URL` before starting Next.js if the API uses another address.

The HTTP API remains available to other clients:

- `GET /health` confirms that the RepoWren process is running.
- `GET /status` checks vLLM readiness and reports the configured model.
- `POST /v1/chat/stream` streams newline-delimited `status`, `token`, `done`,
  and `error` events.
- `POST/GET /v1/repositories` registers or lists repositories.
- `POST /v1/repositories/{id}/select` selects the active repository.
- `GET /v1/repositories/{id}/files` lists live safe paths and byte sizes.
- `GET /v1/repositories/{id}/file?path=...` reads one safe UTF-8 file.
- `POST /v1/repositories/{id}/search` returns the semantic and exact-text query
  plan plus hybrid-ranked chunks with path, line range, text, and combined RRF
  score. The terminal prints that plan before the aggregate results.
- `POST /v1/repositories/{id}/chat/stream` injects the five closest chunks.

## Benchmark

With both services running, measure one end-to-end response:

```text
python benchmarks/inference.py
```

The probe reports time to first text and total request time. Docker Desktop and
`nvidia-smi` can be used separately to observe peak RAM and VRAM.

## Troubleshooting

- If the Docker command is missing, start Docker Desktop and add its
  `resources\bin` directory to `PATH`.
- Docker GPU passthrough on Windows requires Linux containers and Docker
  Desktop's WSL 2 backend. RepoWren no longer installs or runs vLLM inside the
  user's Ubuntu distribution.
- `/health` on port `8000` checks only RepoWren. Use `/status` or vLLM's port
  `8001` to verify inference readiness.
- The first container start may be slow while the image and model are
  downloaded. Inspect `docker compose logs -f vllm` before assuming it failed.
- If vLLM runs out of memory, reduce `VLLM_MAX_MODEL_LEN` or
  `VLLM_GPU_MEMORY_UTILIZATION`; do not select a larger model for this GPU.
- If port `8001` is occupied, change both `VLLM_PORT` and `VLLM_BASE_URL` in
  `.env`.
- If port `5432` is occupied, change `POSTGRES_PORT` and the port in
  `DATABASE_URL` together.
- If port `9200` is occupied, change both `ELASTICSEARCH_PORT` and the port in
  `ELASTICSEARCH_URL`.
- If repository registration reports an Elasticsearch failure, check
  `docker compose logs elasticsearch`, then register the repository again to
  rebuild both complete snapshots.
- If startup says the `vector` extension is unavailable, recreate only the
  PostgreSQL container using the current Compose file. Do not remove the named
  volume unless you intentionally want to erase registered repositories.
- `repowren-api` explicitly uses a Selector event loop on Windows because
  psycopg async connections cannot run on the default Proactor event loop.

## Project layout

```text
RepoWren/
|-- repowren/             API, repository tools, search adapters, and vLLM adapter
|-- web/                  Next.js repository chat, search, and file browser
|-- tests/                deterministic offline tests
|-- benchmarks/           end-to-end timing probe
|-- docs/architecture.md  architecture and milestone boundaries
|-- Dockerfile.vllm       text-only compatibility layer over official vLLM
|-- compose.yaml          Docker vLLM and PostgreSQL services
|-- pyproject.toml        package metadata and dependencies
`-- .env.example          safe local configuration template
```

See [docs/architecture.md](docs/architecture.md) for the request flow, safety
boundary, and later architectural milestones.
