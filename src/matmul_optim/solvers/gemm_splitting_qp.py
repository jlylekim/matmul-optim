from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Literal

import torch

from gemm_kkt.linalg._ops import batch_eye, batch_quad_form
from gemm_kkt.linalg.krylov import batched_cg
from gemm_kkt.linalg.ns_inverse import NSInverseConfig, apply_inverse, newton_schulz_inverse
from gemm_kkt.solvers.common import IterationLog, SolverResult, sync_if_cuda
from gemm_kkt.solvers.types import DenseBatchQP


@dataclass
class GEMMSplittingQPConfig:
    max_iters: int = 3_000
    rho: float = 1.0
    tol_p: float = 1e-4
    tol_d: float = 1e-4
    regularization: float = 1e-7
    linear_solver: Literal["cg_ns", "ns_only"] = "cg_ns"
    cg_max_iters: int = 200
    cg_tol: float = 1e-6
    ns_config: NSInverseConfig = field(default_factory=NSInverseConfig)


class GemmSplittingQPSolver:
    """GPU ADMM/OSQP-style splitting solver with NS-enabled linear solves."""

    def __init__(self, config: GEMMSplittingQPConfig | None = None):
        self.config = config or GEMMSplittingQPConfig()

    def _matvec_A(self, A: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if A.dim() == 2:
            return torch.einsum("ij,bj->bi", A, x)
        return torch.bmm(A, x.unsqueeze(-1)).squeeze(-1)

    def _matvec_At(self, A: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if A.dim() == 2:
            return torch.einsum("ji,bj->bi", A, y)
        return torch.bmm(A.transpose(1, 2), y.unsqueeze(-1)).squeeze(-1)

    def solve(self, problem: DenseBatchQP) -> SolverResult:
        p = problem.as_batched().expanded()
        device = p.q.device

        B, n = p.q.shape
        P = p.P.float()
        A = p.A.float()
        q = p.q.float()
        l = p.l.float()
        u = p.u.float()

        x = torch.zeros(B, n, device=device, dtype=torch.float32)
        z = torch.zeros(B, l.shape[-1], device=device, dtype=torch.float32)
        y = torch.zeros_like(z)

        if A.dim() == 2:
            AtA = batch_quad_form(A.unsqueeze(0).expand(B, -1, -1), torch.ones(B, A.shape[0], device=device))
        else:
            AtA = torch.bmm(A.transpose(1, 2), A)

        I = batch_eye(B, n, device=device, dtype=torch.float32)
        K = P + self.config.rho * AtA + self.config.regularization * I

        inv = None
        if self.config.linear_solver in ("cg_ns", "ns_only"):
            inv, _ = newton_schulz_inverse(K, self.config.ns_config)

        history: list[IterationLog] = []
        t0 = time.perf_counter()
        status = "max_iters"

        for it in range(self.config.max_iters):
            z_prev = z.clone()

            rhs = -q + self.config.rho * self._matvec_At(A, z - y)

            if self.config.linear_solver == "ns_only":
                x = apply_inverse(inv, rhs).float()
            else:
                precond = (lambda v: apply_inverse(inv, v)) if inv is not None else None
                x, _ = batched_cg(
                    K,
                    rhs,
                    x0=x,
                    preconditioner=precond,
                    max_iters=self.config.cg_max_iters,
                    tol=self.config.cg_tol,
                )

            Ax = self._matvec_A(A, x)
            z = torch.clamp(Ax + y, min=l, max=u)
            y = y + Ax - z

            primal_res = float((Ax - z).norm(dim=1).mean().item())
            dual_res = float((self.config.rho * self._matvec_At(A, z - z_prev)).norm(dim=1).mean().item())

            history.append(
                IterationLog(
                    iter=it + 1,
                    primal_residual=primal_res,
                    dual_residual=dual_res,
                    complementarity=0.0,
                    duality_gap=0.0,
                    alpha_primal=1.0,
                    alpha_dual=1.0,
                    solve_residual=0.0,
                    kkt_method=self.config.linear_solver,
                    iter_time_s=0.0,
                )
            )

            if primal_res <= self.config.tol_p and dual_res <= self.config.tol_d:
                status = "solved"
                break

        total = time.perf_counter() - t0
        metrics = {
            "primal_residual": history[-1].primal_residual if history else float("inf"),
            "dual_residual": history[-1].dual_residual if history else float("inf"),
            "duality_gap": None,
            "certificate_type": "primal_dual_residual",
        }

        return SolverResult(
            status=status,
            iterations=len(history),
            x=x,
            y=y,
            metrics=metrics,
            history=history,
            timing={"total_s": total},
            metadata={"linear_solver": self.config.linear_solver},
        )
