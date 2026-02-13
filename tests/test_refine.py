from __future__ import annotations

import torch

from gemm_kkt.linalg._ops import batch_matmul
from gemm_kkt.linalg.ns_inverse import NSInverseConfig, apply_inverse, newton_schulz_inverse
from gemm_kkt.linalg.refine import iterative_refinement


def test_iterative_refinement_reduces_residual() -> None:
    torch.manual_seed(0)
    B, n = 4, 10
    A = torch.randn(B, n, n)
    M = torch.bmm(A.transpose(1, 2), A) + 0.2 * torch.eye(n).unsqueeze(0)

    b = torch.randn(B, n)
    inv, _ = newton_schulz_inverse(M, NSInverseConfig(max_iters=6, tol=1e-2))
    x0 = apply_inverse(inv, b)

    r0 = (b - batch_matmul(M, x0.unsqueeze(-1)).squeeze(-1)).norm(dim=1).mean().item()
    x1, _ = iterative_refinement(M, b, x0, inv=inv, max_iters=8, tol=1e-7)
    r1 = (b - batch_matmul(M, x1.unsqueeze(-1)).squeeze(-1)).norm(dim=1).mean().item()

    assert r1 <= r0
