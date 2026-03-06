from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog

from benchmarks.datasets.load_lp_instances import CanonicalLP
from benchmarks.datasets.lp_adapters import canonical_to_standard_form


def test_standard_form_conversion_preserves_optimum() -> None:
    # Original LP:
    #   min 2 x1 - x2
    #   s.t. 1 <= x1 + x2 <= 2
    #        0 <= x1 <= 2
    #        x2 free
    A = sp.csr_matrix(np.array([[1.0, 1.0]], dtype=np.float64))
    c = np.array([2.0, -1.0], dtype=np.float64)
    row_l = np.array([1.0], dtype=np.float64)
    row_u = np.array([2.0], dtype=np.float64)
    col_l = np.array([0.0, -np.inf], dtype=np.float64)
    col_u = np.array([2.0, np.inf], dtype=np.float64)

    canonical = CanonicalLP(
        name="toy",
        suite="unit",
        sense="min",
        A_csr=A,
        c=c,
        row_l=row_l,
        row_u=row_u,
        col_l=col_l,
        col_u=col_u,
        meta={},
    )
    std = canonical_to_standard_form(canonical)

    orig = linprog(
        c=c,
        A_ub=np.array([[1.0, 1.0], [-1.0, -1.0]], dtype=np.float64),
        b_ub=np.array([2.0, -1.0], dtype=np.float64),
        bounds=[(0.0, 2.0), (None, None)],
        method="highs",
    )
    assert orig.success, f"Original LP solve failed: {orig.message}"

    conv = linprog(
        c=std.c,
        A_eq=std.A_eq.toarray(),
        b_eq=std.b_eq,
        bounds=[(0.0, None)] * std.c.shape[0],
        method="highs",
    )
    assert conv.success, f"Converted LP solve failed: {conv.message}"

    z = conv.x
    x = std.x_map @ z[: std.n_core] + std.x_shift
    orig_obj = float(c @ orig.x)
    conv_obj = float(std.c @ z + std.objective_offset)

    assert abs(orig_obj - conv_obj) <= 1e-6
    assert np.all(A @ x <= row_u + 1e-7)
    assert np.all(A @ x >= row_l - 1e-7)
    assert np.all(x >= col_l - 1e-7)
    assert np.all(x <= col_u + 1e-7)
    assert std.diagnostics["added_splits"] >= 1

