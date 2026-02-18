from __future__ import annotations

import torch

from gemm_kkt.baselines.base import BaselineRunConfig
from gemm_kkt.baselines.cpu_refs import run_scipy_trust_constr_qp
from gemm_kkt.solvers.types import DenseBatchQP


def test_scipy_trust_constr_qp_smoke() -> None:
    n = 4
    P = torch.eye(n, dtype=torch.float64).unsqueeze(0)
    q = torch.tensor([[0.2, -0.3, 0.1, -0.2]], dtype=torch.float64)
    A = torch.eye(n, dtype=torch.float64).unsqueeze(0)
    l = torch.full((1, n), -10.0, dtype=torch.float64)
    u = torch.full((1, n), 10.0, dtype=torch.float64)

    problem = DenseBatchQP(P=P, q=q, A=A, l=l, u=u)
    res = run_scipy_trust_constr_qp(
        problem,
        BaselineRunConfig(tol_p=1e-8, tol_d=1e-8, tol_g=1e-8, max_iters=300),
    )

    assert res.status in {"solved", "max_iters", "failed"}
    assert "primal_residual" in res.metrics
    assert "dual_residual" in res.metrics
    assert float(res.metrics["primal_residual"]) < 1e-6
