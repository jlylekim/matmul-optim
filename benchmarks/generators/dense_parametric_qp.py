from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from gemm_kkt.solvers.types import DenseBatchQP


@dataclass
class DenseParametricQPConfig:
    batch_size: int = 32
    n: int = 256
    m: int = 512
    condition_number: float = 1e4
    tightness: float = 0.2
    rhs_count: int = 1
    shared_matrices: bool = True
    seed: int = 0
    device: str = "cuda"
    dtype: torch.dtype = torch.float32


def _make_spd(n: int, cond: float, *, device: torch.device, dtype: torch.dtype, gen: torch.Generator) -> torch.Tensor:
    Q, _ = torch.linalg.qr(torch.randn(n, n, device=device, dtype=dtype, generator=gen))
    vals = torch.logspace(0.0, math.log10(cond), n, device=device, dtype=torch.float32)
    D = torch.diag(vals.to(dtype))
    return Q @ D @ Q.transpose(0, 1)


def generate_dense_parametric_qp(config: DenseParametricQPConfig) -> DenseBatchQP:
    device = torch.device(config.device if torch.cuda.is_available() and config.device.startswith("cuda") else "cpu")
    gen = torch.Generator(device=device)
    gen.manual_seed(config.seed)

    total_batch = config.batch_size * config.rhs_count

    if config.shared_matrices:
        P = _make_spd(config.n, config.condition_number, device=device, dtype=config.dtype, gen=gen)
        A = torch.randn(config.m, config.n, device=device, dtype=config.dtype, generator=gen) / (config.n**0.5)
    else:
        P = torch.stack(
            [_make_spd(config.n, config.condition_number, device=device, dtype=config.dtype, gen=gen) for _ in range(total_batch)],
            dim=0,
        )
        A = torch.randn(total_batch, config.m, config.n, device=device, dtype=config.dtype, generator=gen) / (config.n**0.5)

    x_ref = torch.randn(total_batch, config.n, device=device, dtype=config.dtype, generator=gen)

    if config.shared_matrices:
        Ax_ref = torch.einsum("mn,bn->bm", A, x_ref)
        Px_ref = torch.einsum("ij,bj->bi", P, x_ref)
    else:
        Ax_ref = torch.bmm(A, x_ref.unsqueeze(-1)).squeeze(-1)
        Px_ref = torch.bmm(P, x_ref.unsqueeze(-1)).squeeze(-1)

    q_noise = torch.randn(total_batch, config.n, device=device, dtype=config.dtype, generator=gen)
    q = -Px_ref + 0.05 * q_noise

    slack = config.tightness * (1.0 + torch.rand(total_batch, config.m, device=device, dtype=config.dtype, generator=gen))
    l = Ax_ref - slack
    u = Ax_ref + slack

    return DenseBatchQP(P=P, q=q, A=A, l=l, u=u)
