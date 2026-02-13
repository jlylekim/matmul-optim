from __future__ import annotations

import torch

from benchmarks.generators.dense_parametric_qp import DenseParametricQPConfig, generate_dense_parametric_qp
from gemm_kkt.solvers.gemm_splitting_qp import GEMMSplittingQPConfig, GemmSplittingQPSolver


def test_splitting_qp_smoke() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    problem = generate_dense_parametric_qp(
        DenseParametricQPConfig(
            batch_size=3,
            n=20,
            m=28,
            condition_number=1e2,
            tightness=0.8,
            shared_matrices=True,
            seed=4,
            device=device,
        )
    )

    solver = GemmSplittingQPSolver(
        GEMMSplittingQPConfig(
            max_iters=300,
            tol_p=1e-2,
            tol_d=1e-2,
            linear_solver="cg_ns",
        )
    )

    res = solver.solve(problem)
    assert len(res.history) > 0
    assert torch.isfinite(res.x).all()
