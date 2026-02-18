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

    # Report normalized residuals to reduce sensitivity to raw problem scaling.
    p_abs = pviol.norm(dim=-1)
    p_scale = 1.0 + torch.maximum(torch.maximum(l.norm(dim=-1), u.norm(dim=-1)), Ax.norm(dim=-1))
    p_rel = p_abs / p_scale.clamp_min(1e-12)

    if P.dim() == 2:
        Px = torch.einsum("ij,bj->bi", P, x)
    else:
        Px = torch.bmm(P, x.unsqueeze(-1)).squeeze(-1)
    if A.dim() == 2:
        Atlam = torch.einsum("ji,bj->bi", A, lam_u - lam_l)
    else:
        Atlam = torch.bmm(A.transpose(1, 2), (lam_u - lam_l).unsqueeze(-1)).squeeze(-1)

    d_abs = dres.norm(dim=-1)
    d_scale = 1.0 + torch.maximum(torch.maximum(q.norm(dim=-1), Px.norm(dim=-1)), Atlam.norm(dim=-1))
    d_rel = d_abs / d_scale.clamp_min(1e-12)

    return {
        "primal_residual": float(p_rel.mean().item()),
        "dual_residual": float(d_rel.mean().item()),
        "primal_residual_abs": float(p_abs.mean().item()),
        "dual_residual_abs": float(d_abs.mean().item()),
        "complementarity": float(gap.mean().item()),
        "duality_gap": float(gap.mean().item()),
    }
