from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class BaselineRunConfig:
    tol_p: float = 1e-4
    tol_d: float = 1e-4
    tol_g: float = 1e-4
    max_iters: int = 10_000
    time_limit_s: float | None = None
    presolve: bool = True
    warm_start: bool = False


@dataclass
class BaselineResult:
    name: str
    status: str
    solve_time_s: float
    metrics: dict[str, Any] = field(default_factory=dict)
    version: str | None = None
    device: str = "cpu"
    metadata: dict[str, Any] = field(default_factory=dict)


def unavailable_result(name: str, reason: str, *, device: str = "cpu") -> BaselineResult:
    return BaselineResult(
        name=name,
        status="unavailable",
        solve_time_s=0.0,
        metrics={},
        device=device,
        metadata={"reason": reason},
    )
