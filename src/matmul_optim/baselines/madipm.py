from __future__ import annotations

import importlib.metadata

from gemm_kkt.baselines.base import BaselineResult, BaselineRunConfig, unavailable_result
from gemm_kkt.solvers.types import DenseBatchLP


def run_madipm_lp(problem: DenseBatchLP, config: BaselineRunConfig | None = None) -> BaselineResult:
    _ = problem
    _ = config or BaselineRunConfig()
    try:
        version = importlib.metadata.version("madipm")
    except Exception:
        return unavailable_result("MadIPM", "MadIPM package not found.", device="cuda")

    return BaselineResult(
        name="MadIPM",
        status="not_implemented",
        solve_time_s=0.0,
        version=version,
        device="cuda",
        metadata={"note": "Add environment-specific MadIPM execution wrapper."},
    )
