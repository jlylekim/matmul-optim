from __future__ import annotations

from dataclasses import dataclass, field

import torch

from gemm_kkt.linalg._ops import as_batched_matrix, batch_eye
from gemm_kkt.linalg.spectral_norm import estimate_spectral_norm


@dataclass
class NSInverseConfig:
    max_iters: int = 30
    tol: float = 1e-5
    damping: float = 1.0
    eps: float = 1e-7
    spectral_iters: int = 20
    residual_check_interval: int = 1


@dataclass
class NSInverseInfo:
    converged: bool
    iters: int
    residual_history: list[float] = field(default_factory=list)
    scale_mean: float = 0.0


def _initial_guess(A: torch.Tensor, eps: float) -> torch.Tensor:
    # X0 = A^T / (||A||_1 ||A||_inf) gives a practical contraction-start for many matrices.
    norm1 = A.abs().sum(dim=1).max(dim=1).values
    norminf = A.abs().sum(dim=2).max(dim=1).values
    beta = (norm1 * norminf + eps).unsqueeze(-1).unsqueeze(-1)
    return A.transpose(1, 2) / beta


def newton_schulz_inverse(
    M: torch.Tensor,
    config: NSInverseConfig | None = None,
) -> tuple[torch.Tensor, NSInverseInfo]:
    """Approximate inverse via stabilized Newton-Schulz iterations.

    Update: X <- X (2I - A X), where A = M / alpha and alpha ~= ||M||_2.
    """
    cfg = config or NSInverseConfig()
    Mb, squeezed = as_batched_matrix(M)
    B, n, _ = Mb.shape

    work = Mb.float()
    alpha = estimate_spectral_norm(work, num_iters=cfg.spectral_iters).clamp_min(cfg.eps)
    A = work / alpha.view(B, 1, 1)

    X = _initial_guess(A, cfg.eps)
    I = batch_eye(B, n, device=A.device, dtype=A.dtype)

    residual_history: list[float] = []
    converged = False
    iters = cfg.max_iters

    for k in range(cfg.max_iters):
        AX = torch.bmm(A, X)
        R = I - AX

        if (k + 1) % cfg.residual_check_interval == 0 or k == cfg.max_iters - 1:
            rel = (R.norm(dim=(1, 2)) / (n**0.5)).max().item()
            residual_history.append(float(rel))
            if rel <= cfg.tol:
                converged = True
                iters = k + 1
                break

        update = torch.bmm(X, 2.0 * I - AX)
        if cfg.damping < 1.0:
            X = (1.0 - cfg.damping) * X + cfg.damping * update
        else:
            X = update

    X = X / alpha.view(B, 1, 1)

    info = NSInverseInfo(
        converged=converged,
        iters=iters,
        residual_history=residual_history,
        scale_mean=float(alpha.mean().item()),
    )

    if squeezed:
        return X.squeeze(0).to(M.dtype), info
    return X.to(M.dtype), info


def apply_inverse(inv: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    """Apply inverse matrix to rhs.

    inv: (n,n) or (B,n,n)
    rhs: (n,), (B,n), or (B,n,k)
    """
    rhs_was_vec = rhs.dim() == 1
    rhs_was_mat = rhs.dim() == 2

    if rhs_was_vec:
        rhs3 = rhs.unsqueeze(0).unsqueeze(-1)
    elif rhs_was_mat:
        rhs3 = rhs.unsqueeze(-1)
    else:
        rhs3 = rhs

    work_dtype = torch.promote_types(inv.dtype, rhs3.dtype)
    inv_work = inv.to(dtype=work_dtype)
    rhs_work = rhs3.to(dtype=work_dtype)

    if inv.dim() == 2:
        out = torch.einsum("ij,bjk->bik", inv_work, rhs_work)
    else:
        out = torch.bmm(inv_work, rhs_work)

    if rhs_was_vec:
        return out.squeeze(0).squeeze(-1)
    if rhs_was_mat:
        return out.squeeze(-1)
    return out
