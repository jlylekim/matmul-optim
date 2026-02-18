"""Linear algebra primitives for GEMM-first KKT solving."""

from gemm_kkt.linalg.metrics import kkt_metrics
from gemm_kkt.linalg.ns_inverse import NSInverseConfig, NSInverseInfo, newton_schulz_inverse
from gemm_kkt.linalg.refine import RefinementInfo, iterative_refinement
from gemm_kkt.linalg.spectral_norm import estimate_spectral_norm

__all__ = [
    "estimate_spectral_norm",
    "NSInverseConfig",
    "NSInverseInfo",
    "newton_schulz_inverse",
    "RefinementInfo",
    "iterative_refinement",
    "kkt_metrics",
]
