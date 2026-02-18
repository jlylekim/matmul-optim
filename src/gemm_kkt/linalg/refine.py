from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import torch

from gemm_kkt.linalg._ops import as_batched_rhs, batch_matmul, restore_rhs_shape
from gemm_kkt.linalg.ns_inverse import apply_inverse


@dataclass
class RefinementInfo:
    converged: bool
    iters: int
    residual_history: list[float] = field(default_factory=list)


def iterative_refinement(
    M: torch.Tensor,
    b: torch.Tensor,
    x0: torch.Tensor,
    *,
    inv: torch.Tensor | None = None,
    inv_apply: Callable[[torch.Tensor], torch.Tensor] | None = None,
    max_iters: int = 8,
    tol: float = 1e-6,
) -> tuple[torch.Tensor, RefinementInfo]:
    """Iterative refinement in fp32 residual space.

    Update: x <- x + M_tilde^{-1}(b - Mx)
    """
    if inv is None and inv_apply is None:
        raise ValueError("Either `inv` or `inv_apply` must be provided")

    b3, b_was_vec, b_was_mat = as_batched_rhs(b)
    x3, _, _ = as_batched_rhs(x0)

    M_work = M.float()
    b_work = b3.float()
    x_work = x3.float()

    residual_history: list[float] = []
    converged = False
    iters = max_iters

    def _apply(v: torch.Tensor) -> torch.Tensor:
        if inv_apply is not None:
            return inv_apply(v)
        assert inv is not None
        return apply_inverse(inv.float(), v)

    b_norm = b_work.norm(dim=(1, 2)).clamp_min(1e-12)

    for k in range(max_iters):
        r = b_work - batch_matmul(M_work, x_work)
        rel = (r.norm(dim=(1, 2)) / b_norm).max().item()
        residual_history.append(float(rel))

        if rel <= tol:
            converged = True
            iters = k
            break

        dx = _apply(r)
        x_work = x_work + dx.float()

    out = restore_rhs_shape(x_work.to(x0.dtype), b_was_vec, b_was_mat)
    info = RefinementInfo(converged=converged, iters=iters, residual_history=residual_history)
    return out, info
