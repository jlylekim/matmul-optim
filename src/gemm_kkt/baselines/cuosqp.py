from __future__ import annotations

import importlib.metadata

from gemm_kkt.baselines.base import BaselineResult, BaselineRunConfig, unavailable_result
from gemm_kkt.solvers.types import DenseBatchQP


def run_cuosqp_qp(problem: DenseBatchQP, config: BaselineRunConfig | None = None) -> BaselineResult:
    _ = problem
    _ = config or BaselineRunConfig()
    try:
        version = importlib.metadata.version("cuosqp")
    except Exception:
        return unavailable_result(
            "cuOSQP",
            "cuOSQP bindings not installed in this environment.",
            device="cuda",
        )

    return BaselineResult(
        name="cuOSQP",
        status="not_implemented",
        solve_time_s=0.0,
        version=version,
        device="cuda",
        metadata={"note": "Implement direct binding call for your cuOSQP build."},
    )
