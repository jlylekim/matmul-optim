from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import torch

from gemm_kkt.linalg._ops import as_batched_rhs, batch_matmul, restore_rhs_shape

MatVecFn = Callable[[torch.Tensor], torch.Tensor]
PrecondFn = Callable[[torch.Tensor], torch.Tensor]


@dataclass
class KrylovInfo:
    converged: bool
    iters: int
    residual_history: list[float] = field(default_factory=list)


def _as_matvec(M_or_fn: torch.Tensor | MatVecFn) -> MatVecFn:
    if callable(M_or_fn):
        return M_or_fn

    M = M_or_fn

    def _mv(x: torch.Tensor) -> torch.Tensor:
        return batch_matmul(M, x)

    return _mv


def batched_cg(
    M_or_fn: torch.Tensor | MatVecFn,
    b: torch.Tensor,
    *,
    x0: torch.Tensor | None = None,
    preconditioner: PrecondFn | None = None,
    max_iters: int = 200,
    tol: float = 1e-6,
) -> tuple[torch.Tensor, KrylovInfo]:
    """Batched conjugate gradient for SPD operators."""
    mv = _as_matvec(M_or_fn)

    b3, b_was_vec, b_was_mat = as_batched_rhs(b)
    x = torch.zeros_like(b3) if x0 is None else as_batched_rhs(x0)[0]

    r = b3 - mv(x)
    z = preconditioner(r) if preconditioner is not None else r
    p = z.clone()
    rz = (r * z).sum(dim=1, keepdim=True)

    b_norm = b3.norm(dim=(1, 2)).clamp_min(1e-12)
    residual_history: list[float] = []
    converged = False
    iters = max_iters

    for k in range(max_iters):
        Ap = mv(p)
        denom = (p * Ap).sum(dim=1, keepdim=True).clamp_min(1e-20)
        alpha = rz / denom

        x = x + alpha * p
        r = r - alpha * Ap

        rel = (r.norm(dim=(1, 2)) / b_norm).max().item()
        residual_history.append(float(rel))
        if rel <= tol:
            converged = True
            iters = k + 1
            break

        z = preconditioner(r) if preconditioner is not None else r
        rz_new = (r * z).sum(dim=1, keepdim=True)
        beta = rz_new / rz.clamp_min(1e-20)
        p = z + beta * p
        rz = rz_new

    return restore_rhs_shape(x, b_was_vec, b_was_mat), KrylovInfo(
        converged=converged,
        iters=iters,
        residual_history=residual_history,
    )


def batched_gmres(
    M_or_fn: torch.Tensor | MatVecFn,
    b: torch.Tensor,
    *,
    x0: torch.Tensor | None = None,
    preconditioner: PrecondFn | None = None,
    restart: int = 20,
    max_iters: int = 200,
    tol: float = 1e-6,
) -> tuple[torch.Tensor, KrylovInfo]:
    """Batched GMRES for general operators.

    Current implementation solves each (batch, rhs) system independently.
    """
    mv = _as_matvec(M_or_fn)
    b3, b_was_vec, b_was_mat = as_batched_rhs(b)

    B, n, k_rhs = b3.shape
    x = torch.zeros_like(b3) if x0 is None else as_batched_rhs(x0)[0].clone()

    residual_history: list[float] = []
    converged_all = True
    max_used_iters = 0

    def _apply_precond(v: torch.Tensor) -> torch.Tensor:
        if preconditioner is None:
            return v
        return preconditioner(v)

    for bi in range(B):
        for ri in range(k_rhs):
            rhs = b3[bi, :, ri]
            xi = x[bi, :, ri]
            rhs_norm = rhs.norm().clamp_min(1e-12)

            iter_count = 0
            solved = False

            while iter_count < max_iters and not solved:
                r0 = rhs - mv(xi.view(1, n, 1)).view(-1)
                beta = r0.norm()
                rel = (beta / rhs_norm).item()
                residual_history.append(float(rel))
                if rel <= tol:
                    solved = True
                    break

                V = [r0 / beta.clamp_min(1e-20)]
                H = torch.zeros(restart + 1, restart, device=rhs.device, dtype=rhs.dtype)
                g = torch.zeros(restart + 1, device=rhs.device, dtype=rhs.dtype)
                g[0] = beta

                m = 0
                for j in range(restart):
                    vj = V[j].view(1, n, 1)
                    z = _apply_precond(vj).view(1, n, 1)
                    w = mv(z).view(-1)
                    for i in range(j + 1):
                        hij = torch.dot(V[i], w)
                        H[i, j] = hij
                        w = w - hij * V[i]
                    H[j + 1, j] = w.norm()
                    if H[j + 1, j] > 0:
                        V.append(w / H[j + 1, j])
                    m = j + 1

                    Hj = H[: m + 1, :m]
                    gj = g[: m + 1]
                    y = torch.linalg.lstsq(Hj, gj).solution
                    Vm = torch.stack(V[:m], dim=1)
                    z_corr = _apply_precond((Vm @ y).view(1, n, 1)).view(-1)
                    x_trial = xi + z_corr
                    r_trial = rhs - mv(x_trial.view(1, n, 1)).view(-1)
                    rel_trial = (r_trial.norm() / rhs_norm).item()
                    residual_history.append(float(rel_trial))
                    if rel_trial <= tol:
                        xi = x_trial
                        solved = True
                        iter_count += m
                        break

                if not solved:
                    Hj = H[: m + 1, :m]
                    gj = g[: m + 1]
                    y = torch.linalg.lstsq(Hj, gj).solution
                    Vm = torch.stack(V[:m], dim=1)
                    z_corr = _apply_precond((Vm @ y).view(1, n, 1)).view(-1)
                    xi = xi + z_corr
                    iter_count += m

            x[bi, :, ri] = xi
            converged_all = converged_all and solved
            max_used_iters = max(max_used_iters, iter_count)

    out = restore_rhs_shape(x, b_was_vec, b_was_mat)
    return out, KrylovInfo(converged=converged_all, iters=max_used_iters, residual_history=residual_history)


def batched_minres(
    M_or_fn: torch.Tensor | MatVecFn,
    b: torch.Tensor,
    *,
    x0: torch.Tensor | None = None,
    preconditioner: PrecondFn | None = None,
    max_iters: int = 200,
    tol: float = 1e-6,
) -> tuple[torch.Tensor, KrylovInfo]:
    """MINRES-style interface using GMRES backend for symmetric/indefinite robustness.

    A full batched Lanczos MINRES is substantial; this path preserves the API and
    robust residual-minimization behavior while staying GPU-compatible.
    """
    return batched_gmres(
        M_or_fn,
        b,
        x0=x0,
        preconditioner=preconditioner,
        restart=min(20, max_iters),
        max_iters=max_iters,
        tol=tol,
    )
