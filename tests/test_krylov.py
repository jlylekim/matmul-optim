from __future__ import annotations

import torch

from gemm_kkt.linalg._ops import batch_matmul
from gemm_kkt.linalg.krylov import batched_cg


def test_batched_cg_solves_spd() -> None:
    torch.manual_seed(1)
    B, n = 2, 20
    A = torch.randn(B, n, n)
    M = torch.bmm(A.transpose(1, 2), A) + 0.1 * torch.eye(n).unsqueeze(0)
    b = torch.randn(B, n)

    x, info = batched_cg(M, b, max_iters=300, tol=1e-7)
    r = b - batch_matmul(M, x.unsqueeze(-1)).squeeze(-1)
    rel = (r.norm(dim=1) / b.norm(dim=1).clamp_min(1e-12)).mean().item()

    assert info.iters > 0
    assert rel < 1e-4
