from __future__ import annotations

import torch

from gemm_kkt.linalg.ns_inverse import NSInverseConfig, newton_schulz_inverse


def test_newton_schulz_inverse_spd_accuracy() -> None:
    torch.manual_seed(0)
    B, n = 3, 12
    X = torch.randn(B, n, n)
    M = torch.bmm(X.transpose(1, 2), X) + 0.5 * torch.eye(n).unsqueeze(0)

    inv, info = newton_schulz_inverse(M, NSInverseConfig(max_iters=40, tol=1e-5))

    I_hat = torch.bmm(M, inv)
    I = torch.eye(n).unsqueeze(0).expand(B, -1, -1)
    err = (I_hat - I).norm(dim=(1, 2)).mean().item()

    assert info.iters > 0
    assert err < 0.2
