from __future__ import annotations

import torch

from benchmarks.generators.batched_lp import BatchedLPConfig, generate_batched_lp
from gemm_kkt.solvers.lp_first_order_gpu import LPFirstOrderConfig, LPFirstOrderSolver


def test_lp_first_order_smoke() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    lp = generate_batched_lp(BatchedLPConfig(batch_size=4, n=32, m=16, seed=9, device=device))
    solver = LPFirstOrderSolver(LPFirstOrderConfig(max_iters=500, tol_p=5e-2, tol_d=5e-2))
    res = solver.solve(lp)

    assert len(res.history) > 0
    assert torch.isfinite(res.x).all()
