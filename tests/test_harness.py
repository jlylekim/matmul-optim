from __future__ import annotations

import torch

from benchmarks.generators.dense_parametric_qp import DenseParametricQPConfig, generate_dense_parametric_qp
from benchmarks.harness import run_qp_solver
from gemm_kkt.solvers.gemm_splitting_qp import GEMMSplittingQPConfig, GemmSplittingQPSolver


def test_harness_record_fields() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    problem = generate_dense_parametric_qp(
        DenseParametricQPConfig(batch_size=2, n=12, m=16, seed=5, device=device)
    )

    solver = GemmSplittingQPSolver(GEMMSplittingQPConfig(max_iters=100, tol_p=1e-2, tol_d=1e-2))

    rec = run_qp_solver(
        run_id="test_run",
        problem_id="p0",
        solver_name="split",
        solver=solver.solve,
        problem=problem,
        warmups=0,
        repeats=1,
    )

    assert rec is not None
    assert rec.problem_id == "p0"
    assert rec.timing.median_s >= 0.0
    assert "throughput_prob_per_s" in rec.metadata
