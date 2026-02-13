from __future__ import annotations

import importlib.metadata

from gemm_kkt.baselines.base import BaselineResult, BaselineRunConfig, unavailable_result
from gemm_kkt.solvers.types import DenseBatchLP, DenseBatchQP


def run_mpax_qp(problem: DenseBatchQP, config: BaselineRunConfig | None = None) -> BaselineResult:
    _ = problem
    _ = config or BaselineRunConfig()
    try:
        version = importlib.metadata.version("mpax")
    except Exception:
        return unavailable_result("MPAX", "MPAX not installed.", device="cuda")

    return BaselineResult(
        name="MPAX",
        status="not_implemented",
        solve_time_s=0.0,
        version=version,
        device="cuda",
        metadata={"note": "Add JAX device-sharding benchmark glue."},
    )


def run_mpax_lp(problem: DenseBatchLP, config: BaselineRunConfig | None = None) -> BaselineResult:
    _ = problem
    _ = config or BaselineRunConfig()
    try:
        version = importlib.metadata.version("mpax")
    except Exception:
        return unavailable_result("MPAX", "MPAX not installed.", device="cuda")

    return BaselineResult(
        name="MPAX",
        status="not_implemented",
        solve_time_s=0.0,
        version=version,
        device="cuda",
        metadata={"note": "Add batched LP path via JAX."},
    )
