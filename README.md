# GEMM-KKT: Factorization-Free GPU KKT Solvers

`GEMM-KKT` is a research-grade benchmarking repository for GPU-first optimization solvers built around a new primitive:

- Newton-Schulz (NS) approximate inverse / inverse-root updates for KKT systems
- iterative refinement using only GEMM-like matrix products
- NS-preconditioned Krylov fallback (CG / MINRES / GMRES)

The core solver (`GEMM-IPM`) avoids sparse/direct factorization and is optimized for tensor-core throughput in dense and batched regimes.

## Why this is novel

1. KKT direction computed by **factorization-free** NS inverse application (plus refinement) instead of Cholesky / LDL.
2. Robust mode that auto-switches to NS-preconditioned Krylov for difficult conditioning.
3. Multi-GPU data parallel path focused on throughput for parametric / multi-RHS workloads.

## Where it is expected to win

- Batched parametric dense QPs (fixed `P, A`, varying `q, l, u`)
- Multi-RHS solves (matrix-matrix solve style)
- Strong/weak scaling over 1,2,4,8 GPUs where throughput matters

## Limitations

- Very ill-conditioned KKT systems can require many NS/Krylov iterations.
- High-accuracy tails may favor direct factorization in low-batch small problems.
- Baseline wrappers depend on external installs/licenses for some solvers.

## Quickstart

Option A: `conda`

```bash
conda env create -f environment.yml
conda activate gemm-kkt
pip install -e .[test]
pytest
```

Option B: Python `venv` (no conda, Python 3.10+)

```bash
bash scripts/setup_venv.sh
source .venv/bin/activate
pytest
```

Manual `venv` setup:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
# CUDA 12.4 wheels (use cpu instead of cu124 for CPU-only):
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
python -m pip install -e ".[test]"
pytest
```

Run reproducible benchmark suite:

```bash
bash scripts/reproduce.sh
```

Run larger GPU-throughput-oriented suite:

```bash
bash scripts/reproduce_large.sh
```

Run overnight discovery sweep (targets ~10 hours by default):

```bash
bash scripts/reproduce_overnight.sh
```

Optional time budget override (hours):

```bash
TARGET_HOURS=8 bash scripts/reproduce_overnight.sh
```

Optional overnight experiment-cap override (per case):

```bash
OVERNIGHT_MAX_EXPERIMENTS=12 TARGET_HOURS=8 bash scripts/reproduce_overnight.sh
```

`OVERNIGHT_MAX_EXPERIMENTS<=0` means "run all planned experiments" for each case (this is the overnight default).

Run a larger 8xA100-oriented comprehensive overnight (all methods/categories, `study=all`):

```bash
bash scripts/reproduce_10h_comprehensive.sh
```

Useful overrides:

```bash
TARGET_HOURS=10 MAX_EXPERIMENTS_PER_CASE=0 REPEATS=7 WARMUPS=2 bash scripts/reproduce_10h_comprehensive.sh
```

`benchmarks/reproduce.py` auto-launches `torchrun` across all visible GPUs by default.
By default, each reproduce invocation is capped at `10` experiments total (`--max-experiments`).
NS-IPM hyperparameter grid search is enabled by default and can be disabled with `--disable-ns-grid-search`.
Recommended discovery presets are `ns_favor`, `mixed`, and `stress`.
Overnight sweep mode is intentionally focused on NS-IPM vs reliable CPU QP baselines (`--skip-splitting --focus-ns-baselines`).
To force single-process behavior:

```bash
python benchmarks/reproduce.py --output results/manual_run_YYYYmmdd_HHMMSS --study all --no-auto-distributed --max-experiments 10 --preset mixed --target-tol 1e-2 --max-iter-scale 2.0
```

Benchmark logs include standardized pass/fail tiers (default `1e-1`, `5e-2`, `1e-2`)
(`metadata.tier_pass.*`) in addition to solver-native stopping status.

Run strong/weak multi-GPU scaling (1,2,4,8 ranks):

```bash
bash scripts/run_scaling.sh
```

Results are written to timestamped directories under `results/` (for example `results/main_20260218_043500`):

- JSONL run logs
- markdown + LaTeX summary table (`summary_table.*`) with accuracy/time only
- PDF plots (`accuracy_vs_time.pdf`) with accuracy/time only
- overnight runs additionally save `manifest.csv` and `combined_results.jsonl` and filter plots to QP NS-vs-baseline solvers

## Repository layout

- `src/gemm_kkt/linalg`: NS kernels, spectral norm estimation, refinement, Krylov, metrics
- `src/gemm_kkt/solvers`: GEMM-IPM, splitting QP, LP first-order, optional batched MIQP BnB wrapper
- `src/gemm_kkt/baselines`: wrappers for external SOTA solvers and CPU references
- `benchmarks/generators`: dense parametric QP, portfolio QP, optional mixed-binary MIQP relaxations
- `benchmarks/harness.py`: unified runner and timing discipline
- `benchmarks/plots.py`: accuracy/time summary + plot generation
- `tests`: kernel and solver smoke tests

## Reproducibility

- Deterministic seeds enabled by default (`torch.use_deterministic_algorithms` where possible)
- For CUDA deterministic GEMM, set `CUBLAS_WORKSPACE_CONFIG=:4096:8` (the provided scripts set this automatically)
- All runs log device, CUDA, driver-reported metadata, tolerances, and stopping metrics
- No hidden tuning: benchmark configs are explicit in `benchmarks/reproduce.py`

## Baseline wrappers

GPU baseline wrappers are included with a uniform result interface for:

- `cuClarabel`, `cuOSQP`, `MPAX`, `cuPDLP-C`, `MadIPM`

Some wrappers are environment stubs by default (`status=not_implemented`) because installation paths differ by cluster image; CPU references include `SciPy trust-constr` (always available with this repo deps) plus optional `OSQP` and `HiGHS` when installed.
