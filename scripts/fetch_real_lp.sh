#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH:-}"
read -r -a REAL_LP_SUITES_ARR <<< "${REAL_LP_SUITES:-netlib}"

python "${ROOT_DIR}/benchmarks/datasets/fetch_lp_suites.py" \
  --suites "${REAL_LP_SUITES_ARR[@]}" \
  --manifest "${ROOT_DIR}/benchmarks/datasets/real_lp_manifest.csv" \
  --raw-dir "${ROOT_DIR}/benchmarks/datasets/raw" \
  --expanded-dir "${ROOT_DIR}/benchmarks/datasets/expanded" \
  "$@"

python "${ROOT_DIR}/benchmarks/datasets/load_lp_instances.py" \
  --manifest "${ROOT_DIR}/benchmarks/datasets/real_lp_manifest.csv" \
  --cache-root "${ROOT_DIR}/benchmarks/datasets/cache"
