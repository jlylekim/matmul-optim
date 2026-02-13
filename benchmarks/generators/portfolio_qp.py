from __future__ import annotations

from dataclasses import dataclass

import torch

from gemm_kkt.solvers.types import DenseBatchQP


@dataclass
class PortfolioQPConfig:
    batch_size: int = 32
    n_assets: int = 128
    n_factors: int = 8
    max_weight: float = 0.2
    risk_scale: float = 1.0
    expected_return_scale: float = 0.05
    seed: int = 0
    device: str = "cuda"


def generate_portfolio_qp(config: PortfolioQPConfig) -> DenseBatchQP:
    device = torch.device(config.device if torch.cuda.is_available() and config.device.startswith("cuda") else "cpu")
    gen = torch.Generator(device=device)
    gen.manual_seed(config.seed)

    B = config.batch_size
    n = config.n_assets

    F = torch.randn(n, config.n_factors, device=device, generator=gen)
    D = torch.rand(n, device=device, generator=gen) * 0.2 + 0.05
    Sigma = F @ F.transpose(0, 1) + torch.diag(D)
    P = config.risk_scale * Sigma + 1e-4 * torch.eye(n, device=device)

    mu = config.expected_return_scale * torch.randn(B, n, device=device, generator=gen)
    q = -mu

    # Constraints:
    # 1) budget equality: sum(x)=1
    # 2) box: 0 <= x_i <= max_weight
    A = torch.cat(
        [
            torch.ones(1, n, device=device),
            torch.eye(n, device=device),
        ],
        dim=0,
    )

    l = torch.cat(
        [
            torch.ones(B, 1, device=device),
            torch.zeros(B, n, device=device),
        ],
        dim=1,
    )
    u = torch.cat(
        [
            torch.ones(B, 1, device=device),
            config.max_weight * torch.ones(B, n, device=device),
        ],
        dim=1,
    )

    return DenseBatchQP(P=P, q=q, A=A, l=l, u=u)
