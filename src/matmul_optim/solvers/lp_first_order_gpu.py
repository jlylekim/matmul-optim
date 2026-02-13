from __future__ import annotations

import time
from dataclasses import dataclass

import torch

from gemm_kkt.solvers.common import IterationLog, SolverResult
from gemm_kkt.solvers.types import DenseBatchLP


@dataclass
class LPFirstOrderConfig:
    max_iters: int = 10_000
    tau: float = 0.5
    sigma: float = 0.5
    theta: float = 1.0
    tol_p: float = 1e-4
    tol_d: float = 1e-4


class LPFirstOrderSolver:
    """PDHG/PDLP-style LP solver for batched dense LPs on GPU."""

    def __init__(self, config: LPFirstOrderConfig | None = None):
        self.config = config or LPFirstOrderConfig()

    def _matvec_A(self, A: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if A.dim() == 2:
            return torch.einsum("ij,bj->bi", A, x)
        return torch.bmm(A, x.unsqueeze(-1)).squeeze(-1)

    def _matvec_At(self, A: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if A.dim() == 2:
            return torch.einsum("ji,bj->bi", A, y)
        return torch.bmm(A.transpose(1, 2), y.unsqueeze(-1)).squeeze(-1)

    def solve(self, problem: DenseBatchLP) -> SolverResult:
        p = problem.as_batched().expanded()
        A = p.A.float()
        b = p.b.float()
        c = p.c.float()

        B, n = c.shape
        m = b.shape[-1]
        device = c.device

        x = torch.zeros(B, n, device=device, dtype=torch.float32)
        y = torch.zeros(B, m, device=device, dtype=torch.float32)
        x_bar = x.clone()

        history: list[IterationLog] = []
        t0 = time.perf_counter()
        status = "max_iters"

        for it in range(self.config.max_iters):
            y = y + self.config.sigma * (self._matvec_A(A, x_bar) - b)
            x_next = torch.relu(x - self.config.tau * (self._matvec_At(A, y) + c))
            x_bar = x_next + self.config.theta * (x_next - x)
            x = x_next

            p_res = float((self._matvec_A(A, x) - b).norm(dim=1).mean().item())
            d_res = float(torch.relu(-(self._matvec_At(A, y) + c)).norm(dim=1).mean().item())
            gap = float((c * x).sum(dim=1).mean().item())

            history.append(
                IterationLog(
                    iter=it + 1,
                    primal_residual=p_res,
                    dual_residual=d_res,
                    complementarity=0.0,
                    duality_gap=gap,
                    alpha_primal=1.0,
                    alpha_dual=1.0,
                    solve_residual=0.0,
                    kkt_method="pdhg",
                    iter_time_s=0.0,
                )
            )

            if p_res <= self.config.tol_p and d_res <= self.config.tol_d:
                status = "solved"
                break

        total = time.perf_counter() - t0

        metrics = {
            "primal_residual": history[-1].primal_residual if history else float("inf"),
            "dual_residual": history[-1].dual_residual if history else float("inf"),
            "duality_gap": history[-1].duality_gap if history else float("inf"),
            "certificate_type": "first_order_primal_dual",
        }

        return SolverResult(
            status=status,
            iterations=len(history),
            x=x,
            y=y,
            metrics=metrics,
            history=history,
            timing={"total_s": total},
            metadata={"method": "PDHG"},
        )
