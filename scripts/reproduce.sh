#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH:-}"
# Required by PyTorch for deterministic CuBLAS behavior.
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export MPLBACKEND="${MPLBACKEND:-Agg}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
PRESET="${REPRO_PRESET:-quick}"
STAMP="$(date +"%Y%m%d_%H%M%S")"
OUT_DIR="${REPRO_OUT_DIR:-${ROOT_DIR}/results/main_${STAMP}}"
mkdir -p "${OUT_DIR}"

echo "[reproduce] output: ${OUT_DIR}"
python "${ROOT_DIR}/benchmarks/reproduce.py" --output "${OUT_DIR}" --study all --preset "${PRESET}"
python "${ROOT_DIR}/benchmarks/plots.py" --input "${OUT_DIR}/results.jsonl" --output "${OUT_DIR}"

echo "Results generated at ${OUT_DIR}"
