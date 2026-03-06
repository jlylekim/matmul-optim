from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

from benchmarks.datasets.load_lp_instances import (
    CanonicalLP,
    load_canonical_lp_npz,
    parse_lp_file,
    save_canonical_lp_npz,
)


def test_canonical_lp_cache_roundtrip(tmp_path: Path) -> None:
    A = sp.csr_matrix(np.array([[1.0, 2.0], [0.0, 1.0]], dtype=np.float64))
    lp = CanonicalLP(
        name="tiny",
        suite="unit",
        sense="min",
        A_csr=A,
        c=np.array([1.0, -1.0], dtype=np.float64),
        row_l=np.array([1.0, 0.0], dtype=np.float64),
        row_u=np.array([1.0, np.inf], dtype=np.float64),
        col_l=np.array([0.0, -np.inf], dtype=np.float64),
        col_u=np.array([np.inf, np.inf], dtype=np.float64),
        meta={"k": "v"},
    )
    cache = tmp_path / "tiny.npz"
    save_canonical_lp_npz(lp, cache)
    loaded = load_canonical_lp_npz(cache)
    assert loaded.name == "tiny"
    assert loaded.suite == "unit"
    assert loaded.sense == "min"
    assert np.allclose(loaded.c, lp.c)
    assert np.allclose(loaded.row_l, lp.row_l, equal_nan=True)
    assert np.allclose(loaded.row_u, lp.row_u, equal_nan=True)
    assert np.allclose(loaded.col_l, lp.col_l, equal_nan=True)
    assert np.allclose(loaded.col_u, lp.col_u, equal_nan=True)
    assert loaded.A_csr.shape == lp.A_csr.shape
    assert np.allclose(loaded.A_csr.toarray(), lp.A_csr.toarray())


@pytest.mark.skipif(importlib.util.find_spec("highspy") is None, reason="highspy not installed")
def test_parse_tiny_mps_with_highspy(tmp_path: Path) -> None:
    mps = """NAME          TINY
ROWS
 N  COST
 E  R1
COLUMNS
    X1        COST      1
    X1        R1        1
RHS
    RHS1      R1        1
BOUNDS
 LO BND1      X1        0
ENDATA
"""
    path = tmp_path / "tiny.mps"
    path.write_text(mps, encoding="utf-8")
    lp = parse_lp_file(path, suite="unit", name="tiny")
    assert lp.n == 1
    assert lp.m == 1
    assert lp.nnz == 1
    assert lp.sense in ("min", "max")

