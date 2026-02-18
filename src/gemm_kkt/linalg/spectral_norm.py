from __future__ import annotations

import torch

from gemm_kkt.linalg._ops import as_batched_matrix


def estimate_spectral_norm(
    M: torch.Tensor,
    num_iters: int = 20,
    tol: float = 1e-6,
    seed: int = 0,
) -> torch.Tensor:
    """Estimate ||M||_2 for (n,n) or (B,n,n) using batched power iteration on M^T M."""
    Mb, squeezed = as_batched_matrix(M)
    B, n, _ = Mb.shape

    work = Mb.float()
    gen = torch.Generator(device=work.device)
    gen.manual_seed(seed)

    v = torch.randn(B, n, 1, device=work.device, dtype=work.dtype, generator=gen)
    v = v / (v.norm(dim=1, keepdim=True) + 1e-12)

    sigma_prev = torch.zeros(B, device=work.device, dtype=work.dtype)

    for _ in range(num_iters):
        w = torch.bmm(work, v)
        sigma = w.norm(dim=1).squeeze(-1)

        z = torch.bmm(work.transpose(1, 2), w)
        z_norm = z.norm(dim=1, keepdim=True)
        v = z / (z_norm + 1e-12)

        if torch.max(torch.abs(sigma - sigma_prev)) < tol:
            sigma_prev = sigma
            break
        sigma_prev = sigma

    if squeezed:
        return sigma_prev.squeeze(0).to(M.dtype)
    return sigma_prev.to(M.dtype)
