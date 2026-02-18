from __future__ import annotations

import torch

from gemm_kkt.linalg._ops import batch_matmul
from gemm_kkt.linalg.krylov import batched_gmres


def test_batched_gmres_handles_batched_matrix_and_preconditioner() -> None:
    torch.manual_seed(0)
    B, n = 3, 12

    A = torch.randn(B, n, n)
    M = torch.bmm(A.transpose(1, 2), A) + 0.2 * torch.eye(n).unsqueeze(0)
    b = torch.randn(B, n)

    # Simple right preconditioner using diagonal inverse.
    diag = torch.diagonal(M, dim1=1, dim2=2).clamp_min(1e-4)
    Pinv = torch.diag_embed(1.0 / diag)

    x, info = batched_gmres(M, b, preconditioner=Pinv, max_iters=80, restart=10, tol=1e-6)
    r = b - batch_matmul(M, x.unsqueeze(-1)).squeeze(-1)
    rel = (r.norm(dim=1) / b.norm(dim=1).clamp_min(1e-12)).mean().item()

    assert info.iters >= 0
    assert rel < 1e-3
