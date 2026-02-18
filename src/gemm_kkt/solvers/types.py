from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class DenseBatchQP:
    """Dense batched QP with two-sided linear constraints.

    minimize    0.5 x^T P x + q^T x
    subject to  l <= A x <= u
    """

    P: torch.Tensor
    q: torch.Tensor
    A: torch.Tensor
    l: torch.Tensor
    u: torch.Tensor

    @property
    def batch_size(self) -> int:
        if self.q.dim() == 1:
            return 1
        return int(self.q.shape[0])

    @property
    def n(self) -> int:
        if self.q.dim() == 1:
            return int(self.q.shape[0])
        return int(self.q.shape[-1])

    @property
    def m(self) -> int:
        if self.l.dim() == 1:
            return int(self.l.shape[0])
        return int(self.l.shape[-1])

    def to(self, device: torch.device | str) -> "DenseBatchQP":
        return DenseBatchQP(
            P=self.P.to(device),
            q=self.q.to(device),
            A=self.A.to(device),
            l=self.l.to(device),
            u=self.u.to(device),
        )

    def cast(self, dtype: torch.dtype) -> "DenseBatchQP":
        return DenseBatchQP(
            P=self.P.to(dtype),
            q=self.q.to(dtype),
            A=self.A.to(dtype),
            l=self.l.to(dtype),
            u=self.u.to(dtype),
        )

    def as_batched(self) -> "DenseBatchQP":
        if self.batch_size > 1:
            return self
        P = self.P.unsqueeze(0) if self.P.dim() == 2 else self.P
        q = self.q.unsqueeze(0) if self.q.dim() == 1 else self.q
        A = self.A.unsqueeze(0) if self.A.dim() == 2 else self.A
        l = self.l.unsqueeze(0) if self.l.dim() == 1 else self.l
        u = self.u.unsqueeze(0) if self.u.dim() == 1 else self.u
        return DenseBatchQP(P=P, q=q, A=A, l=l, u=u)

    def expanded(self) -> "DenseBatchQP":
        """Broadcast shared P/A/l/u to match q batch dimension."""
        batch = self.batch_size
        if batch == 1:
            return self.as_batched()

        P = self.P
        A = self.A
        l = self.l
        u = self.u

        if P.dim() == 2:
            P = P.unsqueeze(0).expand(batch, -1, -1)
        if A.dim() == 2:
            A = A.unsqueeze(0).expand(batch, -1, -1)
        if l.dim() == 1:
            l = l.unsqueeze(0).expand(batch, -1)
        if u.dim() == 1:
            u = u.unsqueeze(0).expand(batch, -1)

        return DenseBatchQP(P=P, q=self.q, A=A, l=l, u=u)

    def slice_batch(self, start: int, end: int) -> "DenseBatchQP":
        q = self.q[start:end] if self.q.dim() == 2 else self.q

        if self.P.dim() == 3:
            P = self.P[start:end]
        else:
            P = self.P

        if self.A.dim() == 3:
            A = self.A[start:end]
        else:
            A = self.A

        if self.l.dim() == 2:
            l = self.l[start:end]
        else:
            l = self.l

        if self.u.dim() == 2:
            u = self.u[start:end]
        else:
            u = self.u

        return DenseBatchQP(P=P, q=q, A=A, l=l, u=u)


@dataclass
class DenseBatchLP:
    """Dense batched LP for first-order comparisons.

    minimize    c^T x
    subject to  A x = b, x >= 0
    """

    A: torch.Tensor
    b: torch.Tensor
    c: torch.Tensor

    @property
    def batch_size(self) -> int:
        if self.b.dim() == 1:
            return 1
        return int(self.b.shape[0])

    @property
    def m(self) -> int:
        if self.b.dim() == 1:
            return int(self.b.shape[0])
        return int(self.b.shape[-1])

    @property
    def n(self) -> int:
        if self.c.dim() == 1:
            return int(self.c.shape[0])
        return int(self.c.shape[-1])

    def to(self, device: torch.device | str) -> "DenseBatchLP":
        return DenseBatchLP(A=self.A.to(device), b=self.b.to(device), c=self.c.to(device))

    def as_batched(self) -> "DenseBatchLP":
        if self.batch_size > 1:
            return self
        A = self.A.unsqueeze(0) if self.A.dim() == 2 else self.A
        b = self.b.unsqueeze(0) if self.b.dim() == 1 else self.b
        c = self.c.unsqueeze(0) if self.c.dim() == 1 else self.c
        return DenseBatchLP(A=A, b=b, c=c)

    def expanded(self) -> "DenseBatchLP":
        batch = self.batch_size
        if batch == 1:
            return self.as_batched()
        A = self.A
        if A.dim() == 2:
            A = A.unsqueeze(0).expand(batch, -1, -1)
        return DenseBatchLP(A=A, b=self.b, c=self.c)

    def slice_batch(self, start: int, end: int) -> "DenseBatchLP":
        b = self.b[start:end] if self.b.dim() == 2 else self.b
        c = self.c[start:end] if self.c.dim() == 2 else self.c
        if self.A.dim() == 3:
            A = self.A[start:end]
        else:
            A = self.A
        return DenseBatchLP(A=A, b=b, c=c)
