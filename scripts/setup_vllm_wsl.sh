#!/usr/bin/env bash
set -euo pipefail

# Run this script from a Linux/WSL shell. The Windows API has its own .venv;
# this separate environment keeps CUDA/vLLM dependencies inside WSL.
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
# Keep CUDA wheels on the native Linux filesystem. Installing this venv under
# /mnt/c can fail while copying large shared libraries from the WSL cache.
VENV_DIR="${VLLM_VENV_DIR:-${HOME}/.venvs/repowren-vllm}"

if [[ -f "${PROJECT_ROOT}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source <(sed 's/\r$//' "${PROJECT_ROOT}/.env")
  set +a
fi

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 is required in WSL. Install it with your Linux distribution's package manager." >&2
  exit 1
fi

mkdir -p "$(dirname -- "${VENV_DIR}")"

if [[ ! -x "${VENV_DIR}/bin/python" ]] || ! "${VENV_DIR}/bin/python" -m pip --version >/dev/null 2>&1; then
  # --clear repairs a directory left behind when ensurepip was missing.
  python3 -m venv --clear "${VENV_DIR}"
fi

# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"

TORCH_BACKEND="${VLLM_TORCH_BACKEND:-cu129}"
VLLM_VERSION="${VLLM_VERSION:-0.31.0}"
VLLM_MARKER="${VENV_DIR}/.repowren-vllm-${VLLM_VERSION}-${TORCH_BACKEND}"

if [[ -f "${VLLM_MARKER}" ]] && python -c "from vllm.entrypoints.cli.main import main; import vllm._C_stable_libtorch" >/dev/null 2>&1; then
  echo "vLLM is already installed in ${VENV_DIR}; skipping dependency installation."
else
  CUDA_VERSION="${TORCH_BACKEND#cu}"
  CPU_ARCH="$(uname -m)"
  if [[ "${TORCH_BACKEND}" != cu[0-9]* ]]; then
    echo "VLLM_TORCH_BACKEND must look like cu129 or cu130." >&2
    exit 1
  fi
  if [[ "${CPU_ARCH}" != x86_64 && "${CPU_ARCH}" != aarch64 ]]; then
    echo "Unsupported WSL CPU architecture: ${CPU_ARCH}" >&2
    exit 1
  fi
  python -m pip install --upgrade pip uv
  # Install the matching prebuilt vLLM binary, not the generic PyPI package.
  # The generic package can resolve a CUDA 13 binary even when the driver only
  # provides CUDA 12.x runtime libraries.
  VLLM_WHEEL="https://github.com/vllm-project/vllm/releases/download/v${VLLM_VERSION}/vllm-${VLLM_VERSION}+${TORCH_BACKEND}-cp38-abi3-manylinux_2_28_${CPU_ARCH}.whl"
  echo "Installing vLLM ${VLLM_VERSION} with the ${TORCH_BACKEND} binary."
  uv pip install --reinstall "${VLLM_WHEEL}" \
    --extra-index-url "https://download.pytorch.org/whl/cu${CUDA_VERSION}" \
    --index-strategy unsafe-best-match
  # TorchCodec is optional for text generation. Its current PyPI binary links
  # against CUDA 13 even in this CUDA 12.9 environment, which prevents vLLM
  # from importing before the model server starts.
  python -m pip uninstall -y torchcodec >/dev/null
  touch "${VLLM_MARKER}"
fi

echo
echo "vLLM is installed in ${VENV_DIR}. Activate it with:"
echo "  source ${VENV_DIR}/bin/activate"
echo "Then start the GPU server with:"
echo "  bash ${PROJECT_ROOT}/scripts/start_vllm.sh"
