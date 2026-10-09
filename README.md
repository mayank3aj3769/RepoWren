# RepoWren

RepoWren is a small, local-first coding-agent API. A lightweight FastAPI process validates chat requests and forwards them to a separate [vLLM](https://github.com/vllm-project/vllm) model server through its OpenAI-compatible API.

```text
client -> RepoWren API :8000 -> vLLM :8001 -> model on GPU
```

Keeping inference in a separate process prevents CUDA and model-serving dependencies from being installed in the API environment. The vLLM server may run locally, in a container, or on another reachable machine.

## Requirements

- Python 3.12 or newer for RepoWren.
- A running vLLM server with a supported GPU environment.
- Enough disk space for the selected model and the Hugging Face download cache.

## Configure RepoWren

Create the local environment file:

```bash
cp .env.example .env
```

The important settings are:

```dotenv
VLLM_MODEL_ID=Qwen/Qwen2.5-Coder-0.5B-Instruct
VLLM_BASE_URL=http://127.0.0.1:8001
LOCAL_AGENT_API_URL=http://127.0.0.1:8000
```

Set `HF_TOKEN` only when the selected Hugging Face repository requires authentication. `.env`, virtual environments, model caches, and model-weight formats are excluded from Git.

## Install the API

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m pip check
python -m pytest -q
```

The API environment intentionally does not install Torch or vLLM.

## Start vLLM

The simplest isolated setup is the official vLLM container. The following command exposes the model server on port `8001` and keeps downloaded weights in the local Hugging Face cache:

```bash
docker run --rm --gpus all --ipc=host \
  -p 8001:8000 \
  -v "${HOME}/.cache/huggingface:/root/.cache/huggingface" \
  --env HF_TOKEN \
  --entrypoint /bin/bash \
  vllm/vllm-openai:latest-cu129 \
  -lc 'python3 -m pip uninstall -y torchcodec --root-user-action=ignore >/dev/null && exec vllm serve Qwen/Qwen2.5-Coder-0.5B-Instruct --host 0.0.0.0 --port 8000 --max-model-len 2048 --gpu-memory-utilization 0.75 --dtype half --enforce-eager'
```

The temporary `torchcodec` removal avoids a CUDA-version mismatch in the current CUDA 12.9 image. RepoWren serves text, so it does not require the optional audio/video decoder. The image itself is not modified.

For a native Linux installation, the included shell helpers install and start the configured CUDA build:

```bash
bash scripts/setup_vllm_wsl.sh
bash scripts/start_vllm.sh
```

The setup helper installs vLLM once in `${VLLM_VENV_DIR:-$HOME/.venvs/repowren-vllm}`. Later starts only require `scripts/start_vllm.sh`. The defaults can be changed in `.env`; verify that the selected vLLM wheel, CUDA backend, GPU, and driver are compatible before changing them.

Wait for vLLM to finish loading, then check it directly:

```bash
curl http://127.0.0.1:8001/health
curl http://127.0.0.1:8001/v1/models
```

## Start RepoWren

In another terminal with the API environment activated:

```bash
python -m uvicorn local_agent.api.app:app --host 127.0.0.1 --port 8000
```

Check both the API process and its connection to vLLM:

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/status
```

Send a streamed chat request:

```bash
curl -N \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Reply with one short word."}],"max_tokens":16,"temperature":0.0}' \
  http://127.0.0.1:8000/v1/chat/stream
```

The response is newline-delimited JSON containing `status`, `token`, `done`, or `error` events.

## API endpoints

- `GET /health` confirms that the RepoWren API process is running.
- `GET /status` checks vLLM readiness and reports the configured model.
- `POST /v1/chat/stream` accepts chat messages and streams RepoWren events.

Run the small end-to-end timing probe while both services are active:

```bash
python benchmarks/inference.py
```

## Troubleshooting

- `/health` on port `8000` only checks RepoWren. Use `/status` to verify that vLLM is reachable.
- If the first vLLM start fails during a model download, check free space in the mounted Hugging Face cache and retry after removing any incomplete download.
- If vLLM reports an out-of-memory error, lower `VLLM_MAX_MODEL_LEN` or `VLLM_GPU_MEMORY_UTILIZATION`, or choose a smaller model.
- Keep `HF_TOKEN`, model weights, caches, and local environment files out of commits.

## Project layout

```text
RepoWren/
|-- local_agent/          FastAPI application and vLLM HTTP adapter
|-- scripts/              optional vLLM and API launch helpers
|-- tests/                deterministic offline tests
|-- benchmarks/           end-to-end timing probe
|-- docs/architecture.md  plain-language architecture guide
|-- pyproject.toml        API dependencies and package metadata
`-- .env.example          safe configuration template
```

Read [docs/architecture.md](docs/architecture.md) for a step-by-step explanation of the request path.
