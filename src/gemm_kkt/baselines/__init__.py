"""Baseline solver wrappers."""

from gemm_kkt.baselines.base import BaselineResult, BaselineRunConfig
from gemm_kkt.baselines.cpu_refs import run_highs_lp, run_osqp_qp, run_scipy_trust_constr_qp

__all__ = ["BaselineResult", "BaselineRunConfig", "run_osqp_qp", "run_highs_lp", "run_scipy_trust_constr_qp"]
