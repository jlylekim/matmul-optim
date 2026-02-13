from __future__ import annotations

from dataclasses import dataclass

import torch

from gemm_kkt.solvers.types import DenseBatchLP


@dataclass
class BatchedLPConfig:
    batch_size: int = 64
    n: int = 256
    m: int = 128
    seed: int = 0
    device: str = "cuda"


def generate_batched_lp(config: BatchedLPConfig) -> DenseBatchLP:
    device = torch.device(config.device if torch.cuda.is_available() and config.device.startswith("cuda") else "cpu")
    gen = torch.Generator(device=device)
    gen.manual_seed(config.seed)

    B, n, m = config.batch_size, config.n, config.m

    A = torch.randn(m, n, device=device, generator=gen) / (n**0.5)
    x_ref = torch.rand(B, n, device=device, generator=gen)
    b = torch.einsum("mn,bn->bm", A, x_ref)
    c = torch.randn(B, n, device=device, generator=gen)

    return DenseBatchLP(A=A, b=b, c=c)
