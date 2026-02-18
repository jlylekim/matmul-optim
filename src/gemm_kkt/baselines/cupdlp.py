from __future__ import annotations

import importlib.metadata

from gemm_kkt.baselines.base import BaselineResult, BaselineRunConfig, unavailable_result
from gemm_kkt.solvers.types import DenseBatchLP


def run_cupdlp_lp(problem: DenseBatchLP, config: BaselineRunConfig | None = None) -> BaselineResult:
    _ = problem
    _ = config or BaselineRunConfig()
    try:
        version = importlib.metadata.version("cupdlp")
    except Exception:
        return unavailable_result("cuPDLP-C", "cuPDLP bindings not detected.", device="cuda")

    return BaselineResult(
        name="cuPDLP-C",
        status="not_implemented",
        solve_time_s=0.0,
        version=version,
        device="cuda",
        metadata={"note": "Hook C/CUDA executable or Python binding in your environment."},
    )
