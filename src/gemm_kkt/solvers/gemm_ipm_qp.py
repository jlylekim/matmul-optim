from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Literal

import torch

from gemm_kkt.linalg._ops import batch_eye, batch_matmul, batch_quad_form
from gemm_kkt.linalg.krylov import batched_gmres, batched_minres
from gemm_kkt.linalg.metrics import kkt_metrics
from gemm_kkt.linalg.ns_inverse import NSInverseConfig, apply_inverse, newton_schulz_inverse
from gemm_kkt.linalg.refine import iterative_refinement
from gemm_kkt.solvers.common import IterationLog, SolverResult, sync_if_cuda
from gemm_kkt.solvers.types import DenseBatchQP


@dataclass
class GEMMIPMConfig:
    max_iters: int = 60
    tol_p: float = 1e-6
    tol_d: float = 1e-6
    tol_g: float = 1e-6
    mu_sigma: float = 0.1
    regularization: float = 1e-6
    positivity_eps: float = 1e-8
    line_search_backoff: float = 0.99
    kkt_mode: Literal["ns_only", "krylov_only", "robust"] = "robust"
    robust_switch_residual: float = 5e-2
    krylov_solver: Literal["gmres", "minres"] = "minres"
    krylov_max_iters: int = 120
    krylov_tol: float = 1e-6
    refinement_iters: int = 6
    refinement_tol: float = 1e-6
    ns_config: NSInverseConfig = field(default_factory=NSInverseConfig)
    precision: Literal["fp32", "fp16", "bf16"] = "bf16"


class GemmIPMQPSolver:
    """Factorization-free primal-dual IPM using GEMM-heavy KKT solves.

    The KKT linearization is reduced to an SPD normal-equation block solved by:
    1) Newton-Schulz inverse apply + iterative refinement
    2) optional Krylov fallback with NS-derived right preconditioner
    """

    def __init__(self, config: GEMMIPMConfig | None = None):
        self.config = config or GEMMIPMConfig()

    def _compute_dtype(self) -> torch.dtype:
        if self.config.precision == "fp16":
            return torch.float16
        if self.config.precision == "bf16":
            return torch.bfloat16
        return torch.float32

    def _matvec_A(self, A: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if A.dim() == 2:
            return torch.einsum("ij,bj->bi", A, x)
        return torch.bmm(A, x.unsqueeze(-1)).squeeze(-1)

    def _matvec_At(self, A: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if A.dim() == 2:
            return torch.einsum("ji,bj->bi", A, y)
        return torch.bmm(A.transpose(1, 2), y.unsqueeze(-1)).squeeze(-1)

    def _matvec_P(self, P: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if P.dim() == 2:
            return torch.einsum("ij,bj->bi", P, x)
        return torch.bmm(P, x.unsqueeze(-1)).squeeze(-1)

    def _max_step_positive(self, s: torch.Tensor, ds: torch.Tensor) -> float:
        mask = ds < 0
        if not torch.any(mask):
            return 1.0
        alpha = torch.min((-s[mask] / ds[mask]).clamp_min(0.0))
        return float(min(1.0, self.config.line_search_backoff * alpha.item()))

    def _kkt_solve(self, M: torch.Tensor, rhs: torch.Tensor, compute_dtype: torch.dtype) -> tuple[torch.Tensor, float, str, dict[str, float]]:
        cfg = self.config
        timing: dict[str, float] = {"ns_s": 0.0, "refine_s": 0.0, "krylov_s": 0.0}

        Mc = M.to(compute_dtype)
        rhsc = rhs.to(compute_dtype)

        method = "krylov"
        inv = None
        solve_res = float("inf")
        dx = torch.zeros_like(rhs)

        if cfg.kkt_mode in ("ns_only", "robust"):
            t0 = time.perf_counter()
            inv, ns_info = newton_schulz_inverse(Mc, cfg.ns_config)
            sync_if_cuda(M.device)
            timing["ns_s"] += time.perf_counter() - t0

            dx = apply_inverse(inv, rhsc).float()
            if cfg.refinement_iters > 0:
                t0 = time.perf_counter()
                dx, _ = iterative_refinement(
                    M,
                    rhs,
                    dx,
                    inv=inv.float(),
                    max_iters=cfg.refinement_iters,
                    tol=cfg.refinement_tol,
                )
                sync_if_cuda(M.device)
                timing["refine_s"] += time.perf_counter() - t0

            r = rhs - batch_matmul(M, dx.unsqueeze(-1)).squeeze(-1)
            solve_res = float((r.norm(dim=1) / rhs.norm(dim=1).clamp_min(1e-12)).mean().item())
            method = "ns"

            if cfg.kkt_mode == "ns_only" and not ns_info.converged:
                method = "ns_unconverged"

        use_krylov = cfg.kkt_mode == "krylov_only" or (
            cfg.kkt_mode == "robust" and solve_res > cfg.robust_switch_residual
        )

        if use_krylov:
            t0 = time.perf_counter()

            precond = inv.float() if inv is not None else None

            if cfg.krylov_solver == "gmres":
                dx, _ = batched_gmres(
                    M,
                    rhs,
                    x0=dx,
                    preconditioner=precond,
                    max_iters=cfg.krylov_max_iters,
                    tol=cfg.krylov_tol,
                )
            else:
                dx, _ = batched_minres(
                    M,
                    rhs,
                    x0=dx,
                    preconditioner=precond,
                    max_iters=cfg.krylov_max_iters,
                    tol=cfg.krylov_tol,
                )

            sync_if_cuda(M.device)
            timing["krylov_s"] += time.perf_counter() - t0
            method = "krylov"

            r = rhs - batch_matmul(M, dx.unsqueeze(-1)).squeeze(-1)
            solve_res = float((r.norm(dim=1) / rhs.norm(dim=1).clamp_min(1e-12)).mean().item())

        return dx.float(), solve_res, method, timing

    def solve(self, problem: DenseBatchQP) -> SolverResult:
        p = problem.as_batched().expanded()
        device = p.q.device
        compute_dtype = self._compute_dtype()

        B, n = p.q.shape
        m = p.l.shape[-1]

        P = p.P.float()
        A = p.A.float()
        q = p.q.float()
        l = p.l.float()
        u = p.u.float()

        x = torch.zeros(B, n, device=device, dtype=torch.float32)
        lam_l = torch.ones(B, m, device=device, dtype=torch.float32)
        lam_u = torch.ones(B, m, device=device, dtype=torch.float32)

        I = batch_eye(B, n, device=device, dtype=torch.float32)

        history: list[IterationLog] = []
        timing_total = {
            "build_kkt_s": 0.0,
            "ns_s": 0.0,
            "refine_s": 0.0,
            "krylov_s": 0.0,
            "line_search_s": 0.0,
            "total_s": 0.0,
        }

        t_total = time.perf_counter()
        status = "max_iters"

        for it in range(self.config.max_iters):
            sync_if_cuda(device)
            t_it = time.perf_counter()

            Ax = self._matvec_A(A, x)
            g_l = Ax - l
            g_u = u - Ax

            g_l_pos = g_l.clamp_min(self.config.positivity_eps)
            g_u_pos = g_u.clamp_min(self.config.positivity_eps)

            r_dual = self._matvec_P(P, x) + q + self._matvec_At(A, lam_u - lam_l)
            mu = self.config.mu_sigma * (g_l_pos * lam_l + g_u_pos * lam_u).mean(dim=1, keepdim=True)
            r_cent_l = lam_l * g_l_pos - mu
            r_cent_u = lam_u * g_u_pos - mu

            t0 = time.perf_counter()
            w = lam_u / g_u_pos + lam_l / g_l_pos
            c = -r_cent_u / g_u_pos + r_cent_l / g_l_pos

            Atc = self._matvec_At(A, c)
            rhs = -r_dual - Atc

            if A.dim() == 2:
                M = P + batch_quad_form(A.unsqueeze(0).expand(B, -1, -1), w)
            else:
                M = P + batch_quad_form(A, w)
            M = M + self.config.regularization * I

            sync_if_cuda(device)
            timing_total["build_kkt_s"] += time.perf_counter() - t0

            dx, solve_res, method, solve_timing = self._kkt_solve(M, rhs, compute_dtype)
            timing_total["ns_s"] += solve_timing["ns_s"]
            timing_total["refine_s"] += solve_timing["refine_s"]
            timing_total["krylov_s"] += solve_timing["krylov_s"]

            dAx = self._matvec_A(A, dx)
            dlam_l = (-r_cent_l - lam_l * dAx) / g_l_pos
            dlam_u = (-r_cent_u + lam_u * dAx) / g_u_pos

            t0 = time.perf_counter()
            alpha_pri = min(
                self._max_step_positive(g_l_pos, dAx),
                self._max_step_positive(g_u_pos, -dAx),
            )
            alpha_dual = min(
                self._max_step_positive(lam_l, dlam_l),
                self._max_step_positive(lam_u, dlam_u),
            )
            alpha = min(alpha_pri, alpha_dual)

            x = x + alpha * dx
            lam_l = (lam_l + alpha * dlam_l).clamp_min(self.config.positivity_eps)
            lam_u = (lam_u + alpha * dlam_u).clamp_min(self.config.positivity_eps)

            sync_if_cuda(device)
            timing_total["line_search_s"] += time.perf_counter() - t0

            metrics = kkt_metrics(P, q, A, l, u, x, lam_l, lam_u)
            sync_if_cuda(device)
            iter_time = time.perf_counter() - t_it

            history.append(
                IterationLog(
                    iter=it + 1,
                    primal_residual=metrics["primal_residual"],
                    dual_residual=metrics["dual_residual"],
                    complementarity=metrics["complementarity"],
                    duality_gap=metrics["duality_gap"],
                    alpha_primal=alpha_pri,
                    alpha_dual=alpha_dual,
                    solve_residual=solve_res,
                    kkt_method=method,
                    iter_time_s=iter_time,
                )
            )

            if (
                metrics["primal_residual"] <= self.config.tol_p
                and metrics["dual_residual"] <= self.config.tol_d
                and metrics["duality_gap"] <= self.config.tol_g
            ):
                status = "solved"
                break

        sync_if_cuda(device)
        timing_total["total_s"] = time.perf_counter() - t_total

        final_metrics = kkt_metrics(P, q, A, l, u, x, lam_l, lam_u)
        final_metrics["certificate_type"] = "hsde_style_primal_dual_gap"

        return SolverResult(
            status=status,
            iterations=len(history),
            x=x,
            lam_l=lam_l,
            lam_u=lam_u,
            metrics=final_metrics,
            history=history,
            timing=timing_total,
            metadata={
                "kkt_mode": self.config.kkt_mode,
                "precision": self.config.precision,
                "krylov_solver": self.config.krylov_solver,
            },
        )
