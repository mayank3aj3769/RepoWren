# RepoWren

RepoWren is a lightweight, local-first coding agent. The current checkpoint is a small working chat path:

```text
terminal -> FastAPI -> llama.cpp -> local Qwen model -> streamed response
```

Repository tools, code editing, PostgreSQL, and Git automation are deliberately deferred to later checkpoints. See `architecture.md` for the design in plain language.

## Tested environment

- Windows x64
- Python 3.12
- NVIDIA GTX 1650 with 4 GiB VRAM
- llama.cpp build `b11503` with CUDA 12.4
- `unsloth/Qwen3.5-2B-GGUF`, file `Qwen3.5-2B-Q5_K_M.gguf`

The model, llama.cpp binaries, virtual environment, caches, and `.env` file are local-only and are not committed to Git.

## 1. Clone and create the virtual environment

Open PowerShell in the cloned repository:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

Verify the Python installation:

```powershell
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m pytest -q
```

`pip check` should report `No broken requirements found`, and the test suite should pass.

## 2. Configure local settings

Create your ignored `.env` from the committed example:

```powershell
Copy-Item .env.example .env
```

The default model is public, so `HF_TOKEN` can stay empty. For a private or gated Hugging Face repository, set your token only in `.env` or in the process environment:

```dotenv
HF_TOKEN=hf_your_token_here
```

Environment variables already present in the shell take precedence over `.env`. Never commit `.env` or paste a token into `.env.example`.

The configurable model fields are:

```dotenv
HF_MODEL_ID=unsloth/Qwen3.5-2B-GGUF
HF_MODEL_FILE=Qwen3.5-2B-Q5_K_M.gguf
HF_MODEL_REVISION=main
HF_MODEL_DIR=models
```

`HF_MODEL_FILE` must name a GGUF supported by llama.cpp. The tested default has a pinned size and SHA-256 check. Hugging Face validates downloaded content for other configured models.

Inspect the resolved settings without downloading:

```powershell
.\.venv\Scripts\python.exe .\scripts\download_model.py --show-config
```

## 3. Download local dependencies

Install the pinned Windows CUDA build of llama.cpp and its runtime libraries. This downloads approximately 626 MiB into the ignored `.runtime` directory:

```powershell
.\.venv\Scripts\python.exe .\scripts\download_runtime.py
```

Download the configured model into the ignored `models` directory:

```powershell
.\.venv\Scripts\python.exe .\scripts\download_model.py
```

The default model is approximately 1.34 GiB. The Hugging Face downloader resumes cached downloads and uses `HF_TOKEN` when it is configured.

## 4. Run RepoWren

Start the model server in the first PowerShell terminal:

```powershell
.\scripts\start_model.ps1
```

Wait until the terminal says `model loaded`. The first generation after startup can be slow while CUDA compiles kernels.

Start the Python API in a second terminal:

```powershell
.\.venv\Scripts\python.exe -m uvicorn local_agent.api.app:app --host 127.0.0.1 --port 8000
```

Start the terminal client in a third terminal:

```powershell
.\.venv\Scripts\local-agent.exe
```

Enter `/exit` to stop the client. Stop each server with `Ctrl+C` in its terminal.

## Local endpoints

- `GET http://127.0.0.1:8000/health`: Python API health
- `GET http://127.0.0.1:8000/status`: API and model readiness
- `POST http://127.0.0.1:8000/v1/chat/stream`: newline-delimited streaming chat
- `GET http://127.0.0.1:8080/health`: llama.cpp health

Both servers bind to `127.0.0.1`, so they are not exposed to other computers by default.

## Observed performance on the tested computer

- Cold model load: about 43.5 seconds
- First request after startup: about 45.6 seconds due to CUDA compilation
- Warm time to first token: about 0.57 seconds
- Warm generation: about 67 tokens/second
- GPU memory after load: about 1,933 MiB total used

Run the repeatable warm benchmark while the model server is active:

```powershell
.\.venv\Scripts\python.exe .\benchmarks\inference.py
```
