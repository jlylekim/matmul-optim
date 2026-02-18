from __future__ import annotations

import torch

from gemm_kkt.linalg._ops import batch_matmul
from gemm_kkt.linalg.krylov import batched_cg
from gemm_kkt.linalg.ns_inverse import apply_inverse


def test_apply_inverse_mixed_dtype() -> None:
    torch.manual_seed(0)
    B, n = 2, 8
    A = torch.randn(B, n, n)
    M = torch.bmm(A.transpose(1, 2), A) + 0.1 * torch.eye(n).unsqueeze(0)
    inv = torch.linalg.inv(M).to(torch.bfloat16)

    b = torch.randn(B, n, dtype=torch.float32)
    x = apply_inverse(inv, b)

    assert x.dtype == torch.float32
    r = b - batch_matmul(M.float(), x.unsqueeze(-1)).squeeze(-1)
    rel = (r.norm(dim=1) / b.norm(dim=1).clamp_min(1e-12)).mean().item()
    # bf16-quantized inverse is approximate; this test validates stable mixed-dtype
    # execution and a reasonable residual, not high-accuracy solve quality.
    assert rel < 5e-2


def test_cg_callable_preconditioner_mixed_dtype() -> None:
    torch.manual_seed(1)
    B, n = 2, 16
    A = torch.randn(B, n, n)
    M = torch.bmm(A.transpose(1, 2), A) + 0.5 * torch.eye(n).unsqueeze(0)
    b = torch.randn(B, n, dtype=torch.float32)

    # Simulate NS inverse in bf16 while Krylov state is fp32.
    inv_bf16 = torch.linalg.inv(M).to(torch.bfloat16)

    x, info = batched_cg(
        M.float(),
        b,
        preconditioner=lambda v: apply_inverse(inv_bf16, v),
        max_iters=60,
        tol=1e-8,
    )

    r = b - batch_matmul(M.float(), x.unsqueeze(-1)).squeeze(-1)
    rel = (r.norm(dim=1) / b.norm(dim=1).clamp_min(1e-12)).mean().item()

    assert info.iters > 0
    assert rel < 1e-4
