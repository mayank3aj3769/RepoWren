# RepoWren

RepoWren is a small, local-first coding agent. This branch uses [AirLLM](https://github.com/lyogavin/airllm) as the inference engine, so model weights are streamed layer by layer instead of being kept entirely in GPU memory.

The current working slice is:

```text
terminal client -> FastAPI -> AirLLM -> Hugging Face model
```

There is no separate llama.cpp executable or model server. The API process loads AirLLM lazily when the first chat request arrives.

## GPU setup on Windows

The tested environment is Python 3.12, an NVIDIA GTX 1650 with 4 GiB VRAM, and an NVIDIA driver that supports CUDA 12.6. Check that the driver can see the GPU first:

```powershell
nvidia-smi
```

Create the virtual environment in the repository and install the CUDA Torch wheel before installing RepoWren. Installing the normal PyPI Torch wheel can produce a CPU-only environment.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install "torch==2.14.1+cu126" --index-url https://download.pytorch.org/whl/cu126
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

Verify the GPU and dependencies:

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m pytest -q
```

If you need a different CUDA wheel, choose a compatible Windows build from the [official PyTorch installation list](https://pytorch.org/get-started/previous-versions/). AirLLM itself requires Torch 2.4 or newer.

## Configure the model

Create the ignored local environment file:

```powershell
Copy-Item .env.example .env
```

The example selects the small public coding model `Qwen/Qwen2.5-Coder-0.5B-Instruct`. Change `AIRLLM_MODEL_ID` to any compatible Hugging Face model when you are ready to try a larger model. The model name is configuration, not code.

For a private or gated repository, set the token only in `.env` or in the process environment:

```dotenv
HF_TOKEN=hf_your_token_here
```

Never commit `.env` or model files. `.gitignore` excludes the environment file, Hugging Face cache, AirLLM layer shards, and common model weight formats.

Inspect the resolved configuration without loading anything:

```powershell
.\.venv\Scripts\python.exe .\scripts\prepare_model.py --show-config
```

## Download and prepare model weights

AirLLM downloads the configured repository from Hugging Face and creates reusable per-layer shards. Run this once before starting the API if you want to prepare the model explicitly:

```powershell
.\.venv\Scripts\python.exe .\scripts\prepare_model.py
```

Original files are cached below `models/huggingface`. Split layer files are stored below `models/airllm/<model-name>`. Both locations are ignored by Git. The default keeps original files because they are useful when changing settings; set `AIRLLM_DELETE_ORIGINAL=true` only when disk space is more important than keeping that cache.

AirLLM compression is optional. Leave `AIRLLM_COMPRESSION` blank for the tested path. `4bit` and `8bit` compression require additional platform-specific packages and should be enabled only after checking the target machine.

## Run RepoWren

Start the local API:

```powershell
.\.venv\Scripts\python.exe -m uvicorn local_agent.api.app:app --host 127.0.0.1 --port 8000
```

The first chat request loads the model if `prepare_model.py` was not run. Start the terminal client in another PowerShell window:

```powershell
.\.venv\Scripts\local-agent.exe
```

Enter `/exit` to stop the client. The API and model stay on the local machine.

## Endpoints and benchmark

- `GET http://127.0.0.1:8000/health` confirms that the API process is alive.
- `GET http://127.0.0.1:8000/status` reports `not_loaded`, `loading`, `ready`, or `error` and the configured model.
- `POST http://127.0.0.1:8000/v1/chat/stream` returns newline-delimited status, token, done, or error events.

Run a short GPU smoke benchmark while the API is active:

```powershell
.\.venv\Scripts\python.exe .\benchmarks\inference.py
```

Layer streaming uses less VRAM but is slower than keeping the whole model resident. The first request also pays model initialization cost. For the tested GTX 1650, a four-token warm probe completed successfully through the API after AirLLM prepared the model.

## Project layout

```text
RepoWren/
|-- local_agent/          Python package and application code
|-- scripts/              model preparation helper
|-- tests/                deterministic offline tests
|-- benchmarks/           API smoke benchmark
|-- docs/architecture.md  plain-language design explanation
|-- pyproject.toml        dependencies and console entry point
`-- .env.example          safe configuration template
```

The old `src/local_agent` wrapper was removed. `local_agent` remains a package because Python needs a package directory for imports; it is now at the repository root so the layout is easier to follow.

For the design explained step by step, read [docs/architecture.md](docs/architecture.md).
