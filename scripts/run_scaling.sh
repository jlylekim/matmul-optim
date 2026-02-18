#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH:-}"
# Required by PyTorch for deterministic CuBLAS behavior.
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

STAMP="$(date +"%Y%m%d_%H%M%S")"
OUT_BASE="${1:-${ROOT_DIR}/results/scaling_${STAMP}}"
PRESET="${SCALING_PRESET:-quick}"
AVAILABLE_GPUS="$(python - <<'PY'
import torch
print(torch.cuda.device_count() if torch.cuda.is_available() else 1)
PY
)"

NPROCS=(1 2 4 8)
FILTERED=()
for N in "${NPROCS[@]}"; do
  if [[ "${N}" -le "${AVAILABLE_GPUS}" ]]; then
    FILTERED+=("${N}")
  fi
done
if [[ "${AVAILABLE_GPUS}" -gt 1 ]]; then
  LAST="${FILTERED[${#FILTERED[@]}-1]:-1}"
  if [[ "${LAST}" -ne "${AVAILABLE_GPUS}" ]]; then
    FILTERED+=("${AVAILABLE_GPUS}")
  fi
fi

for NPROC in "${FILTERED[@]}"; do
  OUT_DIR="${OUT_BASE}/gpus_${NPROC}"
  mkdir -p "${OUT_DIR}"

  torchrun --standalone --nproc_per_node="${NPROC}" "${ROOT_DIR}/benchmarks/reproduce.py" \
    --output "${OUT_DIR}/strong" \
    --study strong \
    --preset "${PRESET}" \
    --repeats 3 \
    --warmups 1

  torchrun --standalone --nproc_per_node="${NPROC}" "${ROOT_DIR}/benchmarks/reproduce.py" \
    --output "${OUT_DIR}/weak" \
    --study weak \
    --preset "${PRESET}" \
    --repeats 3 \
    --warmups 1

done

echo "Scaling studies complete under ${OUT_BASE}"
