#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH:-}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export MPLBACKEND="${MPLBACKEND:-Agg}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

TARGET_HOURS="${TARGET_HOURS:-10}"
OVERNIGHT_MAX_EXPERIMENTS="${OVERNIGHT_MAX_EXPERIMENTS:-10}"
STAMP="$(date +"%Y%m%d_%H%M%S")"
OUT_BASE="${OVERNIGHT_OUT_DIR:-${ROOT_DIR}/results/overnight_${STAMP}}"
mkdir -p "${OUT_BASE}"

TARGET_SECONDS="$(python - <<PY
h = float(${TARGET_HOURS})
print(int(max(1.0, h) * 3600))
PY
)"

START_TS="$(date +%s)"
MANIFEST="${OUT_BASE}/manifest.csv"
{
  echo "case,status,duration_s,output_dir,args"
} > "${MANIFEST}"

run_case() {
  local case_name="$1"
  shift
  local case_dir="${OUT_BASE}/${case_name}"
  mkdir -p "${case_dir}"

  local now elapsed
  now="$(date +%s)"
  elapsed="$((now - START_TS))"
  if (( elapsed >= TARGET_SECONDS )); then
    echo "[overnight] time budget reached (${elapsed}s >= ${TARGET_SECONDS}s). stopping."
    return 1
  fi

  echo "[overnight] case=${case_name} elapsed=${elapsed}s/${TARGET_SECONDS}s"
  echo "[overnight] output=${case_dir}"

  local case_start status case_end dur
  case_start="$(date +%s)"
  if python "${ROOT_DIR}/benchmarks/reproduce.py" \
    --output "${case_dir}" \
    --study main \
    --repeats 5 \
    --warmups 1 \
    --max-experiments "${OVERNIGHT_MAX_EXPERIMENTS}" \
    --skip-splitting \
    --focus-ns-baselines \
    --target-tol 1e-2 \
    --max-iter-scale 2.0 \
    --eval-tiers 1e-1 5e-2 1e-2 \
    "$@"; then
    status="ok"
  else
    status="failed"
  fi

  if [[ "${status}" == "ok" ]]; then
    python "${ROOT_DIR}/benchmarks/plots.py" \
      --input "${case_dir}/results.jsonl" \
      --output "${case_dir}" \
      --category qp_conic \
      --include-solvers gemm_ipm_ns gemm_ipm_robust scipy_trust_constr_cpu osqp_cpu || true
  fi

  case_end="$(date +%s)"
  dur="$((case_end - case_start))"
  echo "${case_name},${status},${dur},${case_dir},$*" >> "${MANIFEST}"

  if [[ "${status}" != "ok" ]]; then
    echo "[overnight] case failed: ${case_name}"
  fi
  return 0
}

# Sweep plan: favorable, mixed, stress; each with multiple seeds.
# The loop naturally stops when TARGET_HOURS budget is reached.
for preset in ns_favor mixed stress; do
  for seed in 0 1 2 3; do
    case_name="${preset}_s${seed}"
    run_case "${case_name}" --preset "${preset}" --seed "${seed}" || break 2
  done
done

# Build combined JSONL + combined summary/plot.
python - "${OUT_BASE}" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

base = Path(sys.argv[1])
out = base / "combined_results.jsonl"
count = 0
with out.open("w", encoding="utf-8") as f_out:
    for case_dir in sorted(base.iterdir()):
        if not case_dir.is_dir():
            continue
        path = case_dir / "results.jsonl"
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as f_in:
            for line in f_in:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                md = row.get("metadata", {})
                md["overnight_case"] = case_dir.name
                row["metadata"] = md
                f_out.write(json.dumps(row) + "\n")
                count += 1
print(f"combined rows: {count}")
PY

mkdir -p "${OUT_BASE}/combined"
python "${ROOT_DIR}/benchmarks/plots.py" \
  --input "${OUT_BASE}/combined_results.jsonl" \
  --output "${OUT_BASE}/combined" \
  --category qp_conic \
  --include-solvers gemm_ipm_ns gemm_ipm_robust scipy_trust_constr_cpu osqp_cpu || true

echo "[overnight] done"
echo "[overnight] output base: ${OUT_BASE}"
echo "[overnight] manifest: ${MANIFEST}"
