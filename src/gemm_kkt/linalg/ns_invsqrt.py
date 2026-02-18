from __future__ import annotations

from dataclasses import dataclass, field

import torch

from gemm_kkt.linalg._ops import as_batched_matrix, batch_eye
from gemm_kkt.linalg.spectral_norm import estimate_spectral_norm


@dataclass
class NSInvSqrtConfig:
    max_iters: int = 30
    tol: float = 1e-5
    eps: float = 1e-7
    spectral_iters: int = 20


@dataclass
class NSInvSqrtInfo:
    converged: bool
    iters: int
    residual_history: list[float] = field(default_factory=list)


def newton_schulz_invsqrt(
    M: torch.Tensor,
    config: NSInvSqrtConfig | None = None,
) -> tuple[torch.Tensor, NSInvSqrtInfo]:
    """Coupled Newton-Schulz inverse square-root for SPD matrices.

    Y_{k+1} = 0.5 Y_k (3I - Z_k Y_k)
    Z_{k+1} = 0.5 (3I - Z_k Y_k) Z_k
    with Y_0 = M/alpha, Z_0 = I.
    """
    cfg = config or NSInvSqrtConfig()
    Mb, squeezed = as_batched_matrix(M)
    B, n, _ = Mb.shape

    work = Mb.float()
    alpha = estimate_spectral_norm(work, num_iters=cfg.spectral_iters).clamp_min(cfg.eps)
    Y = work / alpha.view(B, 1, 1)
    Z = batch_eye(B, n, device=work.device, dtype=work.dtype)
    I = batch_eye(B, n, device=work.device, dtype=work.dtype)

    residual_history: list[float] = []
    converged = False
    iters = cfg.max_iters

    for k in range(cfg.max_iters):
        ZY = torch.bmm(Z, Y)
        T = 0.5 * (3.0 * I - ZY)
        Y = torch.bmm(Y, T)
        Z = torch.bmm(T, Z)

        R = I - torch.bmm(Y, Y)
        rel = (R.norm(dim=(1, 2)) / (n**0.5)).max().item()
        residual_history.append(float(rel))
        if rel <= cfg.tol:
            converged = True
            iters = k + 1
            break

    invsqrt = Z / torch.sqrt(alpha).view(B, 1, 1)
    info = NSInvSqrtInfo(converged=converged, iters=iters, residual_history=residual_history)

    if squeezed:
        return invsqrt.squeeze(0).to(M.dtype), info
    return invsqrt.to(M.dtype), info
