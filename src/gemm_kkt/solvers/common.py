from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class IterationLog:
    iter: int
    primal_residual: float
    dual_residual: float
    complementarity: float
    duality_gap: float
    alpha_primal: float
    alpha_dual: float
    solve_residual: float
    kkt_method: str
    iter_time_s: float


@dataclass
class SolverResult:
    status: str
    iterations: int
    x: torch.Tensor
    y: torch.Tensor | None = None
    lam_l: torch.Tensor | None = None
    lam_u: torch.Tensor | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    history: list[IterationLog] = field(default_factory=list)
    timing: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


def sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device=device)
