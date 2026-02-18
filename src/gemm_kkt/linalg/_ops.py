from __future__ import annotations

import torch


def as_batched_matrix(M: torch.Tensor) -> tuple[torch.Tensor, bool]:
    if M.dim() == 2:
        return M.unsqueeze(0), True
    return M, False


def as_batched_rhs(B: torch.Tensor) -> tuple[torch.Tensor, bool, bool]:
    was_vector = False
    was_matrix = False

    if B.dim() == 1:
        B = B.unsqueeze(0).unsqueeze(-1)
        was_vector = True
    elif B.dim() == 2:
        B = B.unsqueeze(-1)
        was_matrix = True
    elif B.dim() != 3:
        raise ValueError(f"Expected RHS dim in {{1,2,3}}, got {B.dim()}")

    return B, was_vector, was_matrix


def restore_rhs_shape(X: torch.Tensor, was_vector: bool, was_matrix: bool) -> torch.Tensor:
    if was_vector:
        return X.squeeze(0).squeeze(-1)
    if was_matrix:
        return X.squeeze(-1)
    return X


def batch_eye(batch: int, n: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    I = torch.eye(n, device=device, dtype=dtype)
    return I.unsqueeze(0).expand(batch, -1, -1).clone()


def batch_matmul(M: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
    """Compute M @ X for shared or batched M.

    M: (n,n) or (B,n,n)
    X: (B,n,k)
    """
    if M.dim() == 2:
        return torch.einsum("ij,bjk->bik", M, X)
    if M.dim() == 3:
        return torch.bmm(M, X)
    raise ValueError(f"Expected M dim in {{2,3}}, got {M.dim()}")


def batch_quad_form(A: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Compute A^T diag(w) A batched.

    A: (B,m,n), w: (B,m)
    returns (B,n,n)
    """
    Aw = A * w.unsqueeze(-1)
    return torch.bmm(A.transpose(1, 2), Aw)
