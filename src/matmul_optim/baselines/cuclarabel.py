from __future__ import annotations

import importlib.metadata

from gemm_kkt.baselines.base import BaselineResult, BaselineRunConfig, unavailable_result
from gemm_kkt.solvers.types import DenseBatchQP


def run_cuclarabel_qp(problem: DenseBatchQP, config: BaselineRunConfig | None = None) -> BaselineResult:
    _ = problem
    _ = config or BaselineRunConfig()
    try:
        version = importlib.metadata.version("cuclarabel")
    except Exception:
        return unavailable_result(
            "cuClarabel",
            "cuClarabel Python bindings not detected. Use JuliaCall/subprocess integration in your environment.",
            device="cuda",
        )

    return BaselineResult(
        name="cuClarabel",
        status="not_implemented",
        solve_time_s=0.0,
        version=version,
        device="cuda",
        metadata={"note": "Add environment-specific subprocess or JuliaCall wrapper."},
    )
