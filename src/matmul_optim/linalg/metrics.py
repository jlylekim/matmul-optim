from __future__ import annotations

from typing import Any

import torch


def primal_violation(Ax: torch.Tensor, l: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    lower = torch.relu(l - Ax)
    upper = torch.relu(Ax - u)
    return lower + upper


def dual_residual(P: torch.Tensor, q: torch.Tensor, A: torch.Tensor, lam_l: torch.Tensor, lam_u: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    if P.dim() == 2:
        Px = torch.einsum("ij,bj->bi", P, x)
    else:
        Px = torch.bmm(P, x.unsqueeze(-1)).squeeze(-1)

    if A.dim() == 2:
        Atlam = torch.einsum("ji,bj->bi", A, lam_u - lam_l)
    else:
        Atlam = torch.bmm(A.transpose(1, 2), (lam_u - lam_l).unsqueeze(-1)).squeeze(-1)

    return Px + q + Atlam


def complementarity(g_l: torch.Tensor, g_u: torch.Tensor, lam_l: torch.Tensor, lam_u: torch.Tensor) -> torch.Tensor:
    return (g_l * lam_l + g_u * lam_u).mean(dim=-1)


def kkt_metrics(
    P: torch.Tensor,
    q: torch.Tensor,
    A: torch.Tensor,
    l: torch.Tensor,
    u: torch.Tensor,
    x: torch.Tensor,
    lam_l: torch.Tensor,
    lam_u: torch.Tensor,
) -> dict[str, Any]:
    if A.dim() == 2:
        Ax = torch.einsum("ij,bj->bi", A, x)
    else:
        Ax = torch.bmm(A, x.unsqueeze(-1)).squeeze(-1)

    pviol = primal_violation(Ax, l, u)
    dres = dual_residual(P, q, A, lam_l, lam_u, x)

    g_l = (Ax - l).clamp_min(1e-12)
    g_u = (u - Ax).clamp_min(1e-12)
    gap = complementarity(g_l, g_u, lam_l, lam_u)

    return {
        "primal_residual": float(pviol.norm(dim=-1).mean().item()),
        "dual_residual": float(dres.norm(dim=-1).mean().item()),
        "complementarity": float(gap.mean().item()),
        "duality_gap": float(gap.mean().item()),
    }
