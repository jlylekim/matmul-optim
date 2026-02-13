from __future__ import annotations

from dataclasses import dataclass

import torch

from gemm_kkt.solvers.types import DenseBatchQP


@dataclass
class SyntheticMIQPConfig:
    horizon: int = 12
    state_dim: int = 8
    control_dim: int = 4
    binary_dim: int = 8
    batch_nodes: int = 32
    seed: int = 0
    device: str = "cuda"


@dataclass
class SyntheticMIQPRelaxationBatch:
    qp: DenseBatchQP
    fixed_binary: torch.Tensor


def generate_miqp_relaxation_batch(config: SyntheticMIQPConfig) -> SyntheticMIQPRelaxationBatch:
    """Generate node relaxations where each node corresponds to fixed binaries.

    This utility is for optional GPU batched branch-and-bound experiments.
    """
    device = torch.device(config.device if torch.cuda.is_available() and config.device.startswith("cuda") else "cpu")
    gen = torch.Generator(device=device)
    gen.manual_seed(config.seed)

    n = config.horizon * (config.state_dim + config.control_dim)
    m = 2 * n

    Q = torch.randn(n, n, device=device, generator=gen)
    P = Q.transpose(0, 1) @ Q + 1e-2 * torch.eye(n, device=device)
    A = torch.randn(m, n, device=device, generator=gen) / (n**0.5)

    batch = config.batch_nodes
    q = torch.randn(batch, n, device=device, generator=gen)

    x_ref = torch.randn(batch, n, device=device, generator=gen)
    Ax = torch.einsum("mn,bn->bm", A, x_ref)

    l = Ax - 0.5
    u = Ax + 0.5

    fixed_binary = torch.randint(0, 2, (batch, config.binary_dim), device=device, generator=gen).float()

    qp = DenseBatchQP(P=P, q=q, A=A, l=l, u=u)
    return SyntheticMIQPRelaxationBatch(qp=qp, fixed_binary=fixed_binary)
