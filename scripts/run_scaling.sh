#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH:-}"

OUT_BASE="${1:-${ROOT_DIR}/artifacts/scaling}"

for NPROC in 1 2 4 8; do
  OUT_DIR="${OUT_BASE}/gpus_${NPROC}"
  mkdir -p "${OUT_DIR}"

  torchrun --standalone --nproc_per_node="${NPROC}" "${ROOT_DIR}/benchmarks/reproduce.py" \
    --output "${OUT_DIR}/strong" \
    --study strong \
    --repeats 3 \
    --warmups 1

  torchrun --standalone --nproc_per_node="${NPROC}" "${ROOT_DIR}/benchmarks/reproduce.py" \
    --output "${OUT_DIR}/weak" \
    --study weak \
    --repeats 3 \
    --warmups 1

done

echo "Scaling studies complete under ${OUT_BASE}"
