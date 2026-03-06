from __future__ import annotations

import time
from typing import Any

import numpy as np

from gemm_kkt.baselines.base import BaselineResult, BaselineRunConfig, unavailable_result
from gemm_kkt.solvers.types import DenseBatchLP, DenseBatchQP


def _qp_primal_residual(A: np.ndarray, l: np.ndarray, u: np.ndarray, x: np.ndarray) -> float:
    Ax = A @ x
    viol = np.maximum(l - Ax, 0.0) + np.maximum(Ax - u, 0.0)
    numer = float(np.linalg.norm(viol))
    denom = 1.0 + max(float(np.linalg.norm(l)), float(np.linalg.norm(u)), float(np.linalg.norm(Ax)))
    return numer / max(denom, 1e-12)


def _failed_result(name: str, error: Exception, *, solve_time_s: float = 0.0) -> BaselineResult:
    return BaselineResult(
        name=name,
        status="failed",
        solve_time_s=float(max(0.0, solve_time_s)),
        metrics={"primal_residual": float("nan"), "dual_residual": float("nan"), "duality_gap": float("nan")},
        device="cpu",
        metadata={"error_type": type(error).__name__, "error_message": str(error)},
    )


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
    try:
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
    except Exception as exc:
        return _failed_result("OSQP", exc, solve_time_s=time.perf_counter() - t0)

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


def run_scipy_trust_constr_qp(problem: DenseBatchQP, config: BaselineRunConfig | None = None) -> BaselineResult:
    cfg = config or BaselineRunConfig()
    try:
        import scipy
        from scipy.optimize import LinearConstraint, minimize
    except Exception as exc:
        return unavailable_result("SciPyTrustConstr", f"SciPy optimize dependencies missing: {exc}")

    p = problem.as_batched().expanded()
    if p.batch_size != 1:
        return unavailable_result("SciPyTrustConstr", "Wrapper currently supports batch_size=1 for CPU baseline")

    P = p.P.squeeze(0).detach().cpu().numpy().astype(np.float64)
    q = p.q.squeeze(0).detach().cpu().numpy().astype(np.float64)
    A = p.A.squeeze(0).detach().cpu().numpy().astype(np.float64)
    l = p.l.squeeze(0).detach().cpu().numpy().astype(np.float64)
    u = p.u.squeeze(0).detach().cpu().numpy().astype(np.float64)

    x0 = np.zeros_like(q)
    linear = LinearConstraint(A, l, u)

    def fun(x: np.ndarray) -> float:
        return float(0.5 * x.dot(P).dot(x) + q.dot(x))

    def jac(x: np.ndarray) -> np.ndarray:
        return P.dot(x) + q

    def hess(_: np.ndarray) -> np.ndarray:
        return P

    t0 = time.perf_counter()
    try:
        result = minimize(
            fun=fun,
            x0=x0,
            method="trust-constr",
            jac=jac,
            hess=hess,
            constraints=[linear],
            options={
                "maxiter": int(cfg.max_iters),
                "gtol": float(min(cfg.tol_p, cfg.tol_d)),
                "xtol": float(min(cfg.tol_p, cfg.tol_d)),
                "barrier_tol": float(cfg.tol_g),
                "verbose": 0,
            },
        )
        solve_time = time.perf_counter() - t0
    except Exception as exc:
        return _failed_result("SciPyTrustConstr", exc, solve_time_s=time.perf_counter() - t0)

    msg = str(result.message).lower()
    if bool(result.success):
        status = "solved"
    elif "max" in msg and "iter" in msg:
        status = "max_iters"
    else:
        status = "failed"

    primal_res = _qp_primal_residual(A=A, l=l, u=u, x=result.x)
    metrics = {
        "primal_residual": float(primal_res),
        "dual_residual": float(getattr(result, "optimality", np.nan)),
        "duality_gap": float("nan"),
        "objective_value": float(result.fun),
    }

    return BaselineResult(
        name="SciPyTrustConstr",
        status=status,
        solve_time_s=solve_time,
        metrics=metrics,
        version=getattr(scipy, "__version__", None),
        device="cpu",
        metadata={
            "presolve": cfg.presolve,
            "warm_start": cfg.warm_start,
            "max_iters": cfg.max_iters,
            "tol_p": cfg.tol_p,
            "tol_d": cfg.tol_d,
            "tol_g": cfg.tol_g,
            "raw_message": str(result.message),
            "success": bool(result.success),
            "nit": int(getattr(result, "nit", 0)),
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
    try:
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
    except Exception as exc:
        return _failed_result("HiGHS", exc)

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
