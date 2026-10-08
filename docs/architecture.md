# RepoWren architecture

This document explains the current design in ordinary language. It is meant to help a developer with average Python skills understand where a request goes and where a future feature should live.

## The one-sentence design

RepoWren is a small Python application with one local HTTP process. FastAPI validates a chat request, a chat service coordinates it, and an AirLLM backend generates text from a Hugging Face model.

```text
developer
    |
    v
terminal client
    |
    v
FastAPI routes ---- health and status
    |
    v
chat service
    |
    v
AirLLM backend ---- Hugging Face cache and layer shards
    |
    v
NVIDIA GPU (one layer at a time)
```

## Why AirLLM replaced llama.cpp

llama.cpp is a very good native runtime for quantized GGUF files. In the previous design it ran as a separate executable, and Python called its OpenAI-compatible HTTP server.

AirLLM fits the next goal better: trying models that are much larger than the available GPU memory. AirLLM loads a Transformers model and keeps only the layer needed for the current calculation on the GPU. It stores reusable layer shards on disk. The application can therefore select a larger Hugging Face model by changing `AIRLLM_MODEL_ID` instead of converting it to GGUF and starting another server.

The trade-off is speed and disk use. Layers must be read repeatedly, so generation is slower than a fully resident model. The original Hugging Face checkpoint and the split layer files also need disk space during preparation. This is an intentional trade-off for a low-VRAM development machine.

The adapter only imports AirLLM and Torch when a model is actually loaded. That keeps health checks and offline unit tests fast, and it gives the rest of the application one small inference interface that could be backed by another engine later.

## Main components

### `local_agent/api/app.py`

Creates the FastAPI application and installs one inference backend in `application.state`. It also exposes `/health`, which checks only that the Python process is alive.

### `local_agent/api/routes.py`

Defines the public HTTP endpoints. `/status` reads the backend state without loading a model. `/v1/chat/stream` creates a streaming response and delegates the request to the chat service.

### `local_agent/api/schemas.py`

Contains Pydantic models for incoming messages and outgoing status data. Limits here prevent an accidental request from sending an unbounded prompt or generation length to the GPU.

### `local_agent/services/chat.py`

Knows the small NDJSON event format used by the terminal client:

- `status` says that lazy loading has started;
- `token` carries visible generated text;
- `done` ends a successful response;
- `error` reports a failure inside the stream.

The service does not know AirLLM method names or Transformers details. That separation makes the service easy to test with a fake backend.

### `local_agent/inference/base.py`

Defines the narrow contract used by the application: a model ID, a status, and an asynchronous `stream_chat` method. A future inference engine can implement this contract without changing the HTTP routes.

### `local_agent/inference/airllm.py`

Implements the contract with AirLLM.

1. The backend starts in `not_loaded` state.
2. The first chat calls `ensure_loaded`.
3. AirLLM downloads the configured Hugging Face model, creates per-layer shards, and constructs the Transformers model.
4. A chat template converts the typed messages into the model's prompt format.
5. `TextIteratorStreamer` receives generated text from a worker thread while FastAPI yields NDJSON events.
6. An async lock serializes generations. This matters because two layer-streaming generations would compete for the same GPU and disk resources.

Loading and generation happen in worker threads so the FastAPI event loop remains responsive to health and status requests.

### `local_agent/config.py`

Reads `.env` and process variables. Process variables win over `.env`, which is useful in scripts and CI. The important settings are:

| Setting | Purpose |
| --- | --- |
| `AIRLLM_MODEL_ID` | Hugging Face model repository or local model path |
| `AIRLLM_DEVICE` | Usually `cuda:0` for the first NVIDIA GPU |
| `AIRLLM_MAX_CONTEXT` | Maximum prompt plus generated-token budget |
| `AIRLLM_SHARDS_DIR` | Root directory for reusable AirLLM layer shards |
| `HF_HOME` | Project-local Hugging Face download cache |
| `HF_TOKEN` | Optional token for private or gated repositories |
| `AIRLLM_COMPRESSION` | Empty, `4bit`, or `8bit`; empty is the tested default |

The backend adds a model-specific folder below `AIRLLM_SHARDS_DIR`, so changing from a small Qwen model to a larger model cannot accidentally reuse the wrong layers.

## One request from start to finish

1. The terminal client appends the user's text to its conversation and sends JSON to `/v1/chat/stream`.
2. FastAPI validates message roles, content length, temperature, and token limits.
3. The chat service emits a loading status if the model is not ready.
4. The AirLLM backend loads the model on the first request, or reuses the already initialized model.
5. The tokenizer applies the model's chat template and bounds the prompt to the configured context window.
6. Transformers generates in a worker thread. AirLLM moves one layer at a time between disk/CPU and GPU.
7. The streamer returns visible text chunks to the async backend.
8. The chat service wraps each chunk as a `token` event, then emits `done`.
9. The terminal prints the chunks as they arrive.

If loading or generation fails, the backend raises `InferenceError` and the service emits an `error` event. `/status` remains useful because it reports the last state without pretending that an unloaded model is ready.

## Why the package is at the repository root

The previous layout put the package below `src/local_agent`. The extra `src` wrapper was not providing value for this small application, so the package is now simply `local_agent/` at the top level. The `local_agent` directory itself is still necessary: it is the Python package that makes imports such as `from local_agent.config import Settings` work.

The project metadata tells setuptools to include `local_agent*`, and the editable virtual-environment install points directly at this package.

## Files that are deliberately local-only

Git tracks source, tests, and safe configuration examples. It does not track:

- `.venv/`, Python caches, or build output;
- `.env` and tokens;
- the Hugging Face cache;
- AirLLM split shards and model weight formats;
- temporary runtime files.

`scripts/prepare_model.py` is the reproducible way to fetch the configured model. It prints the resolved model ID and paths without exposing the token, and `--show-config` performs no download.

## Testing strategy

The normal test suite never downloads a model. It uses a fake tokenizer, fake streamer, and fake model to check prompt formatting, device selection, loading-once behavior, token order, and model-specific shard paths. API tests use a fake backend to check the NDJSON protocol.

The real GPU smoke path is separate:

1. install a CUDA Torch wheel;
2. run `scripts/prepare_model.py`;
3. start the API;
4. run `benchmarks/inference.py`.

This keeps normal tests deterministic while still giving the developer a clear hardware check.

## Future extension points

The next features should attach to the existing boundaries:

- a context builder can prepare bounded repository excerpts before `stream_chat`;
- repository tools can remain separate services with path validation and result limits;
- an approval layer can turn model proposals into explicit patches before writing files;
- a different local inference engine can implement `InferenceBackend` without changing the API.

State-changing operations—editing files, staging changes, committing, and pushing—should remain explicit user-approved actions. The model must not receive unrestricted filesystem or Git access.
