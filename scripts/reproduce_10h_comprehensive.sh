#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH:-}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export MPLBACKEND="${MPLBACKEND:-Agg}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

TARGET_HOURS="${TARGET_HOURS:-10}"
# <=0 means "run all planned experiments" within each case.
MAX_EXPERIMENTS_PER_CASE="${MAX_EXPERIMENTS_PER_CASE:-0}"
REPEATS="${REPEATS:-7}"
WARMUPS="${WARMUPS:-2}"
TARGET_TOL="${TARGET_TOL:-1e-2}"
MAX_ITER_SCALE="${MAX_ITER_SCALE:-3.0}"
NS_GRID_TUNE_BATCH="${NS_GRID_TUNE_BATCH:-8}"
NS_GRID_TUNE_ITERS="${NS_GRID_TUNE_ITERS:-60}"
STAMP="$(date +"%Y%m%d_%H%M%S")"
OUT_BASE="${COMPREHENSIVE_OUT_DIR:-${ROOT_DIR}/results/comprehensive_${STAMP}}"
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
    echo "[comprehensive] time budget reached (${elapsed}s >= ${TARGET_SECONDS}s). stopping."
    return 1
  fi

  echo "[comprehensive] case=${case_name} elapsed=${elapsed}s/${TARGET_SECONDS}s"
  echo "[comprehensive] output=${case_dir}"

  local case_start status case_end dur
  case_start="$(date +%s)"
  if python "${ROOT_DIR}/benchmarks/reproduce.py" \
    --output "${case_dir}" \
    --study all \
    --repeats "${REPEATS}" \
    --warmups "${WARMUPS}" \
    --max-experiments "${MAX_EXPERIMENTS_PER_CASE}" \
    --target-tol "${TARGET_TOL}" \
    --max-iter-scale "${MAX_ITER_SCALE}" \
    --ns-grid-tune-batch "${NS_GRID_TUNE_BATCH}" \
    --ns-grid-tune-iters "${NS_GRID_TUNE_ITERS}" \
    --eval-tiers 1e-1 5e-2 1e-2 \
    "$@"; then
    status="ok"
  else
    status="failed"
  fi

  if [[ "${status}" == "ok" ]]; then
    python "${ROOT_DIR}/benchmarks/plots.py" \
      --input "${case_dir}/results.jsonl" \
      --output "${case_dir}" || true
  fi

  case_end="$(date +%s)"
  dur="$((case_end - case_start))"
  echo "${case_name},${status},${dur},${case_dir},$*" >> "${MANIFEST}"

  if [[ "${status}" != "ok" ]]; then
    echo "[comprehensive] case failed: ${case_name}"
  fi
  return 0
}

# Priority order: large realistic workloads first, then mixed/easier sweeps.
for preset in a100_heavy stress mixed ns_favor large; do
  for seed in 0 1 2 3; do
    case_name="${preset}_s${seed}"
    run_case "${case_name}" --preset "${preset}" --seed "${seed}" || break 2
  done
done

# Build combined JSONL.
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
                md["comprehensive_case"] = case_dir.name
                row["metadata"] = md
                f_out.write(json.dumps(row) + "\n")
                count += 1
print(f"combined rows: {count}")
PY

# Combined views: all, QP-only, LP-only.
mkdir -p "${OUT_BASE}/combined_all"
python "${ROOT_DIR}/benchmarks/plots.py" \
  --input "${OUT_BASE}/combined_results.jsonl" \
  --output "${OUT_BASE}/combined_all" || true

mkdir -p "${OUT_BASE}/combined_qp"
python "${ROOT_DIR}/benchmarks/plots.py" \
  --input "${OUT_BASE}/combined_results.jsonl" \
  --output "${OUT_BASE}/combined_qp" \
  --category qp_conic || true

mkdir -p "${OUT_BASE}/combined_lp"
python "${ROOT_DIR}/benchmarks/plots.py" \
  --input "${OUT_BASE}/combined_results.jsonl" \
  --output "${OUT_BASE}/combined_lp" \
  --category lp || true

echo "[comprehensive] done"
echo "[comprehensive] output base: ${OUT_BASE}"
echo "[comprehensive] manifest: ${MANIFEST}"
