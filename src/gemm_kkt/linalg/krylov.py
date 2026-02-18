from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import torch

from gemm_kkt.linalg._ops import as_batched_rhs, batch_matmul, restore_rhs_shape

MatVecFn = Callable[[torch.Tensor], torch.Tensor]
PrecondFn = Callable[[torch.Tensor], torch.Tensor]
PrecondArg = PrecondFn | torch.Tensor | None


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


def _apply_preconditioner(preconditioner: PrecondArg, v: torch.Tensor, *, batch_index: int | None = None) -> torch.Tensor:
    if preconditioner is None:
        return v

    if callable(preconditioner):
        return preconditioner(v)

    P = preconditioner
    work_dtype = torch.promote_types(P.dtype, v.dtype)
    P_work = P.to(dtype=work_dtype)
    v_work = v.to(dtype=work_dtype)
    if P.dim() == 2:
        if v_work.dim() == 3:
            return torch.einsum("ij,bjk->bik", P_work, v_work)
        return P_work @ v_work

    if P.dim() == 3:
        if v_work.dim() == 3 and v_work.shape[0] == P_work.shape[0]:
            return torch.bmm(P_work, v_work)
        if batch_index is None:
            raise ValueError("batch_index required for single-system apply with batched preconditioner")
        Pb = P_work[batch_index]
        if v_work.dim() == 3:
            return (Pb @ v_work.squeeze(0)).unsqueeze(0)
        return Pb @ v_work

    raise ValueError(f"Unsupported preconditioner tensor dimension: {P.dim()}")


def batched_cg(
    M_or_fn: torch.Tensor | MatVecFn,
    b: torch.Tensor,
    *,
    x0: torch.Tensor | None = None,
    preconditioner: PrecondArg = None,
    max_iters: int = 200,
    tol: float = 1e-6,
) -> tuple[torch.Tensor, KrylovInfo]:
    """Batched conjugate gradient for SPD operators."""
    mv = _as_matvec(M_or_fn)

    b3, b_was_vec, b_was_mat = as_batched_rhs(b)
    x = torch.zeros_like(b3) if x0 is None else as_batched_rhs(x0)[0]

    r = b3 - mv(x)
    z = _apply_preconditioner(preconditioner, r).to(dtype=r.dtype)
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

        z = _apply_preconditioner(preconditioner, r).to(dtype=r.dtype)
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
    preconditioner: PrecondArg = None,
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

    for bi in range(B):
        if isinstance(M_or_fn, torch.Tensor):
            M_tensor = M_or_fn

            def _mv_vec(vec: torch.Tensor) -> torch.Tensor:
                if M_tensor.dim() == 2:
                    return M_tensor @ vec
                return M_tensor[bi] @ vec

        else:
            def _mv_vec(vec: torch.Tensor) -> torch.Tensor:
                return mv(vec.view(1, n, 1)).view(-1)

        def _apply_precond_vec(vec: torch.Tensor) -> torch.Tensor:
            out = _apply_preconditioner(preconditioner, vec.view(1, n, 1), batch_index=bi)
            return out.view(-1).to(dtype=rhs.dtype)

        for ri in range(k_rhs):
            rhs = b3[bi, :, ri]
            xi = x[bi, :, ri]
            rhs_norm = rhs.norm().clamp_min(1e-12)

            iter_count = 0
            solved = False

            while iter_count < max_iters and not solved:
                r0 = rhs - _mv_vec(xi)
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
                    z = _apply_precond_vec(V[j])
                    w = _mv_vec(z)
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
                    z_corr = _apply_precond_vec(Vm @ y)
                    x_trial = xi + z_corr
                    r_trial = rhs - _mv_vec(x_trial)
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
                    z_corr = _apply_precond_vec(Vm @ y)
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
    preconditioner: PrecondArg = None,
    max_iters: int = 200,
    tol: float = 1e-6,
) -> tuple[torch.Tensor, KrylovInfo]:
    """MINRES-style interface.

    For the current KKT normal-equation path (SPD), use CG backend.
    """
    return batched_cg(
        M_or_fn,
        b,
        x0=x0,
        preconditioner=preconditioner,
        max_iters=max_iters,
        tol=tol,
    )
