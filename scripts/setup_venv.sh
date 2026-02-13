#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python3.10}"
VENV_DIR="${VENV_DIR:-${ROOT_DIR}/.venv}"
TORCH_CHANNEL="${TORCH_CHANNEL:-cu124}" # use TORCH_CHANNEL=cpu for CPU-only

"${PYTHON_BIN}" -m venv "${VENV_DIR}"
# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip setuptools wheel

if [[ "${TORCH_CHANNEL}" == "cpu" ]]; then
  python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
else
  python -m pip install torch torchvision torchaudio --index-url "https://download.pytorch.org/whl/${TORCH_CHANNEL}"
fi

python -m pip install -e "${ROOT_DIR}[test]"

echo "Virtual environment ready at ${VENV_DIR}"
echo "Activate with: source ${VENV_DIR}/bin/activate"
