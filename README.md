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

Run strong/weak multi-GPU scaling (1,2,4,8 ranks):

```bash
bash scripts/run_scaling.sh
```

Artifacts are written to `artifacts/`:

- JSONL run logs
- markdown + LaTeX tables
- PDF plots

## Repository layout

- `src/gemm_kkt/linalg`: NS kernels, spectral norm estimation, refinement, Krylov, metrics
- `src/gemm_kkt/solvers`: GEMM-IPM, splitting QP, LP first-order, optional batched MIQP BnB wrapper
- `src/gemm_kkt/baselines`: wrappers for external SOTA solvers and CPU references
- `benchmarks/generators`: dense parametric QP, portfolio QP, optional mixed-binary MIQP relaxations
- `benchmarks/harness.py`: unified runner and timing discipline
- `benchmarks/plots.py`: performance profiles and scaling plots
- `tests`: kernel and solver smoke tests

## Reproducibility

- Deterministic seeds enabled by default (`torch.use_deterministic_algorithms` where possible)
- All runs log device, CUDA, driver-reported metadata, tolerances, and stopping metrics
- No hidden tuning: benchmark configs are explicit in `benchmarks/reproduce.py`

## Baseline wrappers

GPU baseline wrappers are included with a uniform result interface for:

- `cuClarabel`, `cuOSQP`, `MPAX`, `cuPDLP-C`, `MadIPM`

Some wrappers are environment stubs by default (`status=not_implemented`) because installation paths differ by cluster image; CPU references (`OSQP`, `HiGHS`) are runnable when dependencies are installed.
