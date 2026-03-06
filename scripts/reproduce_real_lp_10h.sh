#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH:-}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export MPLBACKEND="${MPLBACKEND:-Agg}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

TARGET_HOURS="${TARGET_HOURS:-10}"
SEEDS="${SEEDS:-0 1 2 3}"
SUITES="${SUITES:-netlib stochlp misc}"
MAX_INSTANCES_PER_CASE="${MAX_INSTANCES_PER_CASE:-0}"
SOLVE_TIMEOUT_S="${SOLVE_TIMEOUT_S:-1800}"
INSTANCE_TIMEOUT_S="${INSTANCE_TIMEOUT_S:-3600}"
PARSE_TIMEOUT_S="${PARSE_TIMEOUT_S:-300}"
REPEATS="${REPEATS:-1}"
WARMUPS="${WARMUPS:-0}"
TARGET_TOL="${TARGET_TOL:-1e-2}"
BATCH_SIZE="${BATCH_SIZE:-1}"
U_CAP="${U_CAP:-1e8}"
MAX_GPUS="${MAX_GPUS:-0}"
STAMP="$(date +"%Y%m%d_%H%M%S")"
OUT_BASE="${REAL_LP_OUT_DIR:-${ROOT_DIR}/results/real_lp_${STAMP}}"
MANIFEST_PATH="${MANIFEST_PATH:-${ROOT_DIR}/benchmarks/datasets/real_lp_manifest.csv}"
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
  local case_log="${case_dir}/run.log"
  mkdir -p "${case_dir}"

  local now elapsed
  now="$(date +%s)"
  elapsed="$((now - START_TS))"
  if (( elapsed >= TARGET_SECONDS )); then
    echo "[real-lp] time budget reached (${elapsed}s >= ${TARGET_SECONDS}s). stopping."
    return 1
  fi

  echo "[real-lp] case=${case_name} elapsed=${elapsed}s/${TARGET_SECONDS}s"
  echo "[real-lp] output=${case_dir}"

  local case_start status case_end dur
  case_start="$(date +%s)"
  if python "${ROOT_DIR}/benchmarks/reproduce_real_lp.py" \
    --manifest "${MANIFEST_PATH}" \
    --output "${case_dir}" \
    --warmups "${WARMUPS}" \
    --repeats "${REPEATS}" \
    --target-tol "${TARGET_TOL}" \
    --batch-size "${BATCH_SIZE}" \
    --u-cap "${U_CAP}" \
    --parse-timeout-s "${PARSE_TIMEOUT_S}" \
    --solve-timeout-s "${SOLVE_TIMEOUT_S}" \
    --instance-timeout-s "${INSTANCE_TIMEOUT_S}" \
    --max-instances "${MAX_INSTANCES_PER_CASE}" \
    --max-gpus "${MAX_GPUS}" \
    "$@" 2>&1 | tee "${case_log}"; then
    status="ok"
  else
    status="failed"
  fi

  if [[ "${status}" == "ok" ]]; then
    python "${ROOT_DIR}/benchmarks/aggregate_real_lp.py" \
      --input "${case_dir}/results.jsonl" \
      --output "${case_dir}" || true
  fi

  case_end="$(date +%s)"
  dur="$((case_end - case_start))"
  echo "${case_name},${status},${dur},${case_dir},$*" >> "${MANIFEST}"
  if [[ "${status}" != "ok" ]]; then
    echo "[real-lp] case failed: ${case_name}"
  fi
  return 0
}

for suite in ${SUITES}; do
  for seed in ${SEEDS}; do
    case_name="${suite}_s${seed}"
    run_case "${case_name}" --suites "${suite}" --seed "${seed}" || break 2
  done
done

# Build combined JSONL with explicit case-failure markers.
python - "${OUT_BASE}" <<'PY'
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

base = Path(sys.argv[1])
manifest = base / "manifest.csv"
combined = base / "combined_results.jsonl"
data_rows = 0
failure_rows = 0
with combined.open("w", encoding="utf-8") as f_out:
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
                md["real_lp_case"] = case_dir.name
                row["metadata"] = md
                f_out.write(json.dumps(row) + "\n")
                data_rows += 1

    if manifest.exists():
        with manifest.open("r", encoding="utf-8", newline="") as f_m:
            for rec in csv.DictReader(f_m):
                case_name = str(rec.get("case", "")).strip()
                status = str(rec.get("status", "")).strip().lower()
                if not case_name or status == "ok":
                    continue
                try:
                    duration_s = float(rec.get("duration_s", "nan"))
                except Exception:
                    duration_s = float("nan")
                row = {
                    "run_id": f"{case_name}_case_failure",
                    "category": "meta",
                    "problem_id": case_name,
                    "solver": "__case_failure__",
                    "status": status or "failed",
                    "timing": {"median_s": duration_s, "p90_s": duration_s, "repeats": 1},
                    "metrics": {},
                    "metadata": {
                        "real_lp_case": case_name,
                        "case_status": status or "failed",
                        "case_duration_s": duration_s,
                        "case_output_dir": rec.get("output_dir"),
                        "case_args": rec.get("args"),
                        "source": "manifest",
                    },
                    "system": {"kind": "case_status_marker"},
                }
                f_out.write(json.dumps(row) + "\n")
                failure_rows += 1
print(f"combined rows: {data_rows + failure_rows} (data={data_rows}, failure_markers={failure_rows})")
PY

mkdir -p "${OUT_BASE}/combined"
python "${ROOT_DIR}/benchmarks/aggregate_real_lp.py" \
  --input "${OUT_BASE}/combined_results.jsonl" \
  --output "${OUT_BASE}/combined" || true

echo "[real-lp] done"
echo "[real-lp] output base: ${OUT_BASE}"
echo "[real-lp] manifest: ${MANIFEST}"

