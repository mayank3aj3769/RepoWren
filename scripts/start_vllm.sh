#!/usr/bin/env bash
set -euo pipefail

# Run this script inside WSL after setup_vllm_wsl.sh. vLLM downloads the model
# from Hugging Face into the normal WSL cache; no weights are stored in Git.
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
# Use native WSL storage by default; override with VLLM_VENV_DIR if needed.
VENV_DIR="${VLLM_VENV_DIR:-${HOME}/.venvs/repowren-vllm}"

if [[ -f "${PROJECT_ROOT}/.env" ]]; then
  set -a
  # Windows-created .env files can contain CRLF line endings. Strip the CR
  # before sourcing so model IDs and numeric flags are not passed with a CR.
  # shellcheck disable=SC1091
  source <(sed 's/\r$//' "${PROJECT_ROOT}/.env")
  set +a
fi

# Older .env files used a repository-relative HF_HOME for AirLLM. Keep model
# downloads on native WSL storage unless the user explicitly supplies an
# absolute cache path.
if [[ -n "${HF_HOME:-}" && "${HF_HOME}" != /* ]]; then
  echo "Ignoring relative HF_HOME=${HF_HOME}; using ${HOME}/.cache/huggingface."
  export HF_HOME="${HOME}/.cache/huggingface"
fi
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
mkdir -p "${HF_HOME}"

if [[ -f "${VENV_DIR}/bin/activate" ]]; then
  # shellcheck disable=SC1090
  source "${VENV_DIR}/bin/activate"
fi

if [[ -x "${VENV_DIR}/bin/vllm" ]]; then
  VLLM_COMMAND=("${VENV_DIR}/bin/vllm")
elif python -c "from vllm.entrypoints.cli.main import main" >/dev/null 2>&1; then
  # Some environments expose the package but do not install its console
  # script on PATH. Invoke the official CLI module directly in that case.
  VLLM_COMMAND=(python -m vllm.entrypoints.cli.main)
else
  echo "vllm cannot be imported from ${VENV_DIR}. Run bash scripts/setup_vllm_wsl.sh first." >&2
  exit 1
fi

MODEL_ID="${VLLM_MODEL_ID:-Qwen/Qwen2.5-Coder-0.5B-Instruct}"
HOST="${VLLM_HOST:-0.0.0.0}"
PORT="${VLLM_PORT:-8001}"
MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-2048}"
GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.75}"
DTYPE="${VLLM_DTYPE:-half}"

exec "${VLLM_COMMAND[@]}" serve "${MODEL_ID}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --dtype "${DTYPE}"
