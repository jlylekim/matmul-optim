from __future__ import annotations

import torch

from benchmarks.generators.dense_parametric_qp import DenseParametricQPConfig, generate_dense_parametric_qp
from gemm_kkt.solvers.gemm_ipm_qp import GEMMIPMConfig, GemmIPMQPSolver


def test_gemm_ipm_smoke() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    problem = generate_dense_parametric_qp(
        DenseParametricQPConfig(
            batch_size=4,
            n=16,
            m=24,
            condition_number=1e2,
            tightness=0.5,
            shared_matrices=True,
            seed=3,
            device=device,
        )
    )

    solver = GemmIPMQPSolver(
        GEMMIPMConfig(
            max_iters=30,
            tol_p=5e-2,
            tol_d=5e-2,
            tol_g=5e-2,
            kkt_mode="robust",
            precision="bf16" if device == "cuda" else "fp32",
        )
    )

    res = solver.solve(problem)
    assert len(res.history) > 0
    assert torch.isfinite(res.x).all()
    assert res.metrics["primal_residual"] < 1.0
