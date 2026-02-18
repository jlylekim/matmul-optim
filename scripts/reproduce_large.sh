#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH:-}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
STAMP="$(date +"%Y%m%d_%H%M%S")"
OUT_DIR="${REPRO_LARGE_OUT_DIR:-${ROOT_DIR}/results/large_${STAMP}}"
mkdir -p "${OUT_DIR}"

python "${ROOT_DIR}/benchmarks/reproduce.py" \
  --output "${OUT_DIR}" \
  --study all \
  --preset large \
  --repeats 5 \
  --warmups 1

python "${ROOT_DIR}/benchmarks/plots.py" \
  --input "${OUT_DIR}/results.jsonl" \
  --output "${OUT_DIR}"

echo "Large results generated at ${OUT_DIR}"
