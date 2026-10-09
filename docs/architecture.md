# RepoWren architecture

This document explains the design in simple terms for a developer who knows ordinary Python and HTTP but does not know vLLM internals.

## The short version

RepoWren has two independent processes:

1. A FastAPI process accepts and validates requests from a terminal, editor, or other client.
2. A vLLM process owns the model and uses the GPU to generate text.

They communicate over HTTP. The RepoWren API does not import Torch, reserve GPU memory, or load model weights.

```text
API caller
   |
   v
RepoWren FastAPI :8000
   |
   | OpenAI-compatible HTTP request
   v
vLLM model server :8001
   |
   v
Hugging Face model + GPU
```

The two processes can run on the same machine or on different machines. RepoWren only needs the URL stored in `VLLM_BASE_URL`.

## What each folder does

### `local_agent/api`

`app.py` creates the FastAPI application. `routes.py` defines `/health`, `/status`, and `/v1/chat/stream`. `schemas.py` validates message roles and request limits before anything reaches the model server.

### `local_agent/services`

`chat.py` coordinates each request and turns generated text into a small newline-delimited event stream:

- `status` says that RepoWren is waiting for the model server;
- `token` carries a generated text fragment;
- `done` marks successful completion;
- `error` reports a failure without crashing the API process.

This layer does not know the details of vLLM's protocol.

### `local_agent/inference`

`base.py` defines the small interface used by the rest of the application: model ID, status, readiness check, cleanup, and an asynchronous text stream.

`vllm.py` is the only module that knows vLLM's OpenAI-compatible protocol. It:

1. Calls `GET /health` to check whether the model server is ready.
2. Sends validated messages to `POST /v1/chat/completions`.
3. Reads Server-Sent Events and yields only generated text deltas.
4. Converts network and protocol failures into `InferenceError`.

Keeping this adapter small makes the rest of RepoWren independent of the inference engine's deployment details.

### `scripts`

`setup_vllm_wsl.sh` creates a native Linux virtual environment and installs the selected prebuilt vLLM CUDA wheel. `start_vllm.sh` reads model-serving settings and starts `vllm serve`. `start_api_wsl.sh` is an optional helper that runs the API in the same Linux environment.

Docker users do not need these setup scripts because the container already contains vLLM and its CUDA dependencies.

## One request, step by step

1. A caller posts JSON to `http://127.0.0.1:8000/v1/chat/stream`.
2. FastAPI validates the JSON structure and token limits.
3. `ChatService` emits a `status` event if the model server is not already marked ready.
4. `VLLMClient` checks the `/health` endpoint configured by `VLLM_BASE_URL`.
5. The adapter sends the messages to vLLM and begins reading its streamed response.
6. Each vLLM text delta becomes a RepoWren `token` event.
7. RepoWren emits `done` when the stream finishes.
8. Network, readiness, or response-format failures become an `error` event.

The API stays responsive because model loading and GPU execution happen in the separate vLLM process.

## Configuration

Process environment variables take precedence over values in `.env`.

| Variable | Meaning |
| --- | --- |
| `VLLM_MODEL_ID` | Model name sent with chat-completion requests |
| `VLLM_BASE_URL` | Base URL of the vLLM server |
| `VLLM_VERSION` | Release used by the optional native installation helper |
| `VLLM_TORCH_BACKEND` | CUDA wheel variant used by the installation helper |
| `VLLM_HOST` / `VLLM_PORT` | Listen address and port used by the startup helper |
| `VLLM_MAX_MODEL_LEN` | Maximum prompt-plus-output context configured for vLLM |
| `VLLM_GPU_MEMORY_UTILIZATION` | Fraction of GPU memory vLLM may reserve |
| `VLLM_DTYPE` | Model data type passed to vLLM |
| `HF_TOKEN` | Optional Hugging Face token for private or gated models |

Tokens, caches, virtual environments, and common model-weight formats are ignored by Git.

## Why vLLM is separate

vLLM is a GPU model server with large CUDA and Torch dependencies. Keeping it outside the API process has three benefits:

1. The API environment remains small and quick to install.
2. Restarting or upgrading vLLM does not require changing the API code.
3. A future deployment can move inference to another machine by changing one URL.

The cost is one extra local service. `/status` exists so callers can distinguish a healthy RepoWren process from a model server that is still loading or unavailable.

## Testing

Unit tests use `httpx.MockTransport` and fake inference backends. They verify validation, readiness states, SSE parsing, and the NDJSON event protocol without downloading a model or requiring a GPU:

```bash
python -m pytest -q
```

Hardware validation is separate: start vLLM, confirm its `/health` endpoint, start RepoWren, and run `benchmarks/inference.py`.
