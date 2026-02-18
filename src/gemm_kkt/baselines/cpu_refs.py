from __future__ import annotations

import time
from typing import Any

import numpy as np

from gemm_kkt.baselines.base import BaselineResult, BaselineRunConfig, unavailable_result
from gemm_kkt.solvers.types import DenseBatchLP, DenseBatchQP


def run_osqp_qp(problem: DenseBatchQP, config: BaselineRunConfig | None = None) -> BaselineResult:
    cfg = config or BaselineRunConfig()
    try:
        import osqp
        import scipy.sparse as sp
    except Exception as exc:
        return unavailable_result("OSQP", f"OSQP dependencies missing: {exc}")

    p = problem.as_batched().expanded()
    if p.batch_size != 1:
        return unavailable_result("OSQP", "Wrapper currently supports batch_size=1 for CPU baseline")

    P = p.P.squeeze(0).detach().cpu().numpy().astype(np.float64)
    q = p.q.squeeze(0).detach().cpu().numpy().astype(np.float64)
    A = p.A.squeeze(0).detach().cpu().numpy().astype(np.float64)
    l = p.l.squeeze(0).detach().cpu().numpy().astype(np.float64)
    u = p.u.squeeze(0).detach().cpu().numpy().astype(np.float64)

    solver = osqp.OSQP()
    t0 = time.perf_counter()
    solver.setup(
        P=sp.csc_matrix(P),
        q=q,
        A=sp.csc_matrix(A),
        l=l,
        u=u,
        eps_abs=min(cfg.tol_p, cfg.tol_d),
        eps_rel=min(cfg.tol_p, cfg.tol_d),
        max_iter=cfg.max_iters,
        warm_start=cfg.warm_start,
        verbose=False,
        polish=False,
    )
    result = solver.solve()
    solve_time = time.perf_counter() - t0

    info: Any = result.info
    status = str(info.status)

    metrics = {
        "primal_residual": float(getattr(info, "prim_res", np.nan)),
        "dual_residual": float(getattr(info, "dual_res", np.nan)),
        "duality_gap": float(getattr(info, "duality_gap", np.nan)),
    }

    return BaselineResult(
        name="OSQP",
        status=status,
        solve_time_s=solve_time,
        metrics=metrics,
        version=getattr(osqp, "__version__", None),
        device="cpu",
        metadata={
            "presolve": cfg.presolve,
            "warm_start": cfg.warm_start,
            "max_iters": cfg.max_iters,
            "tol_absrel": min(cfg.tol_p, cfg.tol_d),
        },
    )


def run_highs_lp(problem: DenseBatchLP, config: BaselineRunConfig | None = None) -> BaselineResult:
    cfg = config or BaselineRunConfig()
    try:
        import highspy
    except Exception as exc:
        return unavailable_result("HiGHS", f"highspy missing: {exc}")

    p = problem.as_batched().expanded()
    if p.batch_size != 1:
        return unavailable_result("HiGHS", "Wrapper currently supports batch_size=1 for CPU baseline")

    A = p.A.squeeze(0).detach().cpu().numpy().astype(np.float64)
    b = p.b.squeeze(0).detach().cpu().numpy().astype(np.float64)
    c = p.c.squeeze(0).detach().cpu().numpy().astype(np.float64)

    m, n = A.shape

    lp = highspy.Highs()
    lp.setOptionValue("output_flag", False)
    lp.setOptionValue("presolve", "on" if cfg.presolve else "off")
    lp.setOptionValue("simplex_iteration_limit", cfg.max_iters)

    inf = highspy.kHighsInf
    lp.addCols(
        n,
        c.tolist(),
        [0.0] * n,
        [inf] * n,
        0,
        [],
        [],
        [],
    )

    starts = [0]
    indices: list[int] = []
    values: list[float] = []
    for i in range(m):
        nz = np.nonzero(A[i])[0]
        indices.extend(int(j) for j in nz)
        values.extend(float(A[i, j]) for j in nz)
        starts.append(len(indices))

    lp.addRows(
        m,
        b.tolist(),
        b.tolist(),
        len(indices),
        starts,
        indices,
        values,
    )

    t0 = time.perf_counter()
    lp.run()
    solve_time = time.perf_counter() - t0

    status = lp.modelStatusToString(lp.getModelStatus())
    info = lp.getInfo()

    metrics = {
        "primal_residual": float(getattr(info, "max_primal_infeasibility", np.nan)),
        "dual_residual": float(getattr(info, "max_dual_infeasibility", np.nan)),
        "duality_gap": float(getattr(info, "objective_function_value", np.nan)),
    }

    return BaselineResult(
        name="HiGHS",
        status=status,
        solve_time_s=solve_time,
        metrics=metrics,
        version=getattr(highspy, "HIGHS_VERSION_MAJOR", None),
        device="cpu",
        metadata={
            "presolve": cfg.presolve,
            "max_iters": cfg.max_iters,
            "warm_start": cfg.warm_start,
        },
    )
