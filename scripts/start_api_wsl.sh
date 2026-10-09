#!/usr/bin/env bash
set -euo pipefail

# Start RepoWren's FastAPI process inside WSL. vLLM should run in another
# terminal on port 8001; this API listens on port 8000.
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
# Use the same native WSL environment as start_vllm.sh by default.
VENV_DIR="${VLLM_VENV_DIR:-${HOME}/.venvs/repowren-vllm}"

if [[ ! -f "${VENV_DIR}/bin/activate" ]]; then
  echo "${VENV_DIR} is missing. Run bash scripts/setup_vllm_wsl.sh first." >&2
  exit 1
fi

# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"
cd "${PROJECT_ROOT}"

if ! python -c "import fastapi, httpx, uvicorn" >/dev/null 2>&1; then
  echo "Installing RepoWren API dependencies into the WSL environment..."
  python -m pip install -e .
fi

API_HOST="${LOCAL_AGENT_HOST:-0.0.0.0}"
API_PORT="${LOCAL_AGENT_PORT:-8000}"
exec python -m uvicorn local_agent.api.app:app \
  --host "${API_HOST}" \
  --port "${API_PORT}"
