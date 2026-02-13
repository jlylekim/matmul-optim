#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH:-}"

python "${ROOT_DIR}/benchmarks/reproduce.py" --output "${ROOT_DIR}/artifacts/main" --study all
python "${ROOT_DIR}/benchmarks/plots.py" --input "${ROOT_DIR}/artifacts/main/results.jsonl" --output "${ROOT_DIR}/artifacts/main"

echo "Artifacts generated at ${ROOT_DIR}/artifacts/main"
