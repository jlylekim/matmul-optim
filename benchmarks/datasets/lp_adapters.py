from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import scipy.sparse as sp
import torch

from benchmarks.datasets.load_lp_instances import CanonicalLP
from gemm_kkt.solvers.types import DenseBatchLP, DenseBatchQP


@dataclass
class LPStandardForm:
    A_eq: sp.csr_matrix
    b_eq: np.ndarray
    c: np.ndarray
    x_map: sp.csr_matrix
    x_shift: np.ndarray
    n_core: int
    objective_offset: float
    diagnostics: dict[str, Any] = field(default_factory=dict)


def _is_finite(x: float) -> bool:
    return bool(np.isfinite(x))


def canonical_to_standard_form(canonical_lp: CanonicalLP) -> LPStandardForm:
    """Convert a general bounded LP to equality + nonnegative variable standard form.

    Original:
        min/max c^T x
        row_l <= A x <= row_u
        col_l <= x <= col_u

    Standard:
        min c_std^T z + objective_offset
        A_eq z = b_eq
        z >= 0
    """
    A = canonical_lp.A_csr.tocsr()
    c = canonical_lp.c.astype(np.float64).copy()
    row_l = canonical_lp.row_l.astype(np.float64).copy()
    row_u = canonical_lp.row_u.astype(np.float64).copy()
    col_l = canonical_lp.col_l.astype(np.float64).copy()
    col_u = canonical_lp.col_u.astype(np.float64).copy()
    m, n = A.shape

    if canonical_lp.sense == "max":
        c = -c

    x_shift = np.zeros(n, dtype=np.float64)
    map_rows: list[int] = []
    map_cols: list[int] = []
    map_vals: list[float] = []
    c_core: list[float] = []
    core_upper: list[float] = []
    added_splits = 0
    objective_offset = 0.0

    def add_core_var(orig_idx: int, coeff_in_x: float, coeff_in_obj: float, upper: float) -> None:
        j = len(c_core)
        map_rows.append(orig_idx)
        map_cols.append(j)
        map_vals.append(coeff_in_x)
        c_core.append(float(coeff_in_obj))
        core_upper.append(float(upper))

    for i in range(n):
        lb = float(col_l[i])
        ub = float(col_u[i])
        ci = float(c[i])
        if _is_finite(lb):
            # x = lb + y, y>=0
            x_shift[i] = lb
            objective_offset += ci * lb
            add_core_var(i, coeff_in_x=1.0, coeff_in_obj=ci, upper=(ub - lb if _is_finite(ub) else np.inf))
        elif _is_finite(ub):
            # x = ub - y, y>=0
            x_shift[i] = ub
            objective_offset += ci * ub
            add_core_var(i, coeff_in_x=-1.0, coeff_in_obj=-ci, upper=np.inf)
        else:
            # free variable split: x = y_pos - y_neg
            add_core_var(i, coeff_in_x=1.0, coeff_in_obj=ci, upper=np.inf)
            add_core_var(i, coeff_in_x=-1.0, coeff_in_obj=-ci, upper=np.inf)
            added_splits += 1

    n_core = len(c_core)
    x_map = sp.csr_matrix((map_vals, (map_rows, map_cols)), shape=(n, n_core), dtype=np.float64)
    A_core = (A @ x_map).tocsr()
    shift_term = A @ x_shift
    row_l_eff = row_l - shift_term
    row_u_eff = row_u - shift_term

    eq_rows: list[sp.csr_matrix] = []
    eq_rhs: list[float] = []
    ineq_rows: list[sp.csr_matrix] = []
    ineq_rhs: list[float] = []
    for i in range(m):
        ai = A_core.getrow(i)
        li = float(row_l_eff[i])
        ui = float(row_u_eff[i])
        li_fin = _is_finite(li)
        ui_fin = _is_finite(ui)
        if li_fin and ui_fin and abs(ui - li) <= 1e-12:
            eq_rows.append(ai)
            eq_rhs.append(li)
            continue
        if ui_fin:
            ineq_rows.append(ai)
            ineq_rhs.append(ui)
        if li_fin:
            ineq_rows.append(-ai)
            ineq_rhs.append(-li)

    for j, ub in enumerate(core_upper):
        if _is_finite(ub):
            ineq_rows.append(sp.csr_matrix(([1.0], ([0], [j])), shape=(1, n_core), dtype=np.float64))
            ineq_rhs.append(float(ub))

    E = sp.vstack(eq_rows, format="csr") if eq_rows else sp.csr_matrix((0, n_core), dtype=np.float64)
    f = np.asarray(eq_rhs, dtype=np.float64) if eq_rhs else np.zeros(0, dtype=np.float64)
    G = sp.vstack(ineq_rows, format="csr") if ineq_rows else sp.csr_matrix((0, n_core), dtype=np.float64)
    h = np.asarray(ineq_rhs, dtype=np.float64) if ineq_rhs else np.zeros(0, dtype=np.float64)

    if G.shape[0] > 0:
        q = int(G.shape[0])
        zeros_eq_slack = sp.csr_matrix((E.shape[0], q), dtype=np.float64)
        top = sp.hstack([E, zeros_eq_slack], format="csr")
        bottom = sp.hstack([G, sp.eye(q, format="csr", dtype=np.float64)], format="csr")
        A_eq = sp.vstack([top, bottom], format="csr")
        b_eq = np.concatenate([f, h], axis=0)
        c_std = np.concatenate([np.asarray(c_core, dtype=np.float64), np.zeros(q, dtype=np.float64)], axis=0)
    else:
        A_eq = E
        b_eq = f
        c_std = np.asarray(c_core, dtype=np.float64)

    n_std = int(A_eq.shape[1])
    m_std = int(A_eq.shape[0])
    density_std = float(A_eq.nnz / float(max(1, m_std * max(1, n_std))))
    diagnostics = {
        "n_orig": int(n),
        "m_orig": int(m),
        "nnz_orig": int(A.nnz),
        "n_std": n_std,
        "m_std": m_std,
        "added_slacks": int(G.shape[0]),
        "added_splits": int(added_splits),
        "density_std": density_std,
        "n_core": int(n_core),
        "m_eq": int(E.shape[0]),
        "m_ineq": int(G.shape[0]),
        "objective_offset": float(objective_offset),
    }
    return LPStandardForm(
        A_eq=A_eq,
        b_eq=b_eq,
        c=c_std,
        x_map=x_map,
        x_shift=x_shift,
        n_core=n_core,
        objective_offset=float(objective_offset),
        diagnostics=diagnostics,
    )


def canonical_to_densebatch_lp(
    canonical_lp: CanonicalLP,
    *,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
    batch_size: int = 1,
) -> tuple[DenseBatchLP, dict[str, Any]]:
    std = canonical_to_standard_form(canonical_lp)
    A_dense = torch.tensor(std.A_eq.toarray(), dtype=dtype, device=device)
    b = torch.tensor(std.b_eq, dtype=dtype, device=device)
    c = torch.tensor(std.c, dtype=dtype, device=device)

    if batch_size > 1:
        b = b.unsqueeze(0).repeat(batch_size, 1)
        c = c.unsqueeze(0).repeat(batch_size, 1)

    out = DenseBatchLP(A=A_dense, b=b, c=c)
    diag = dict(std.diagnostics)
    diag.update(
        {
            "suite": canonical_lp.suite,
            "instance_name": canonical_lp.name,
            "objective_sense_original": canonical_lp.sense,
            "objective_sense_converted": "min",
        }
    )
    return out, diag


def canonical_to_densebatch_qp(
    canonical_lp: CanonicalLP,
    *,
    u_cap: float = 1e8,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
    batch_size: int = 1,
) -> tuple[DenseBatchQP, dict[str, Any]]:
    lp_problem, diag = canonical_to_densebatch_lp(
        canonical_lp,
        device=device,
        dtype=dtype,
        batch_size=batch_size,
    )

    if batch_size > 1:
        A_lp = lp_problem.A if lp_problem.A.dim() == 2 else lp_problem.A[0]
        b_lp = lp_problem.b[0]
        c_lp = lp_problem.c
    else:
        A_lp = lp_problem.A if lp_problem.A.dim() == 2 else lp_problem.A.squeeze(0)
        b_lp = lp_problem.b if lp_problem.b.dim() == 1 else lp_problem.b.squeeze(0)
        c_lp = lp_problem.c if lp_problem.c.dim() == 1 else lp_problem.c.squeeze(0)

    n = int(A_lp.shape[1])
    I = torch.eye(n, dtype=dtype, device=device)
    A_qp = torch.cat([A_lp, I], dim=0)
    l = torch.cat([b_lp, torch.zeros(n, dtype=dtype, device=device)], dim=0)
    u = torch.cat([b_lp, torch.full((n,), float(u_cap), dtype=dtype, device=device)], dim=0)
    P = torch.zeros((n, n), dtype=dtype, device=device)

    if batch_size > 1:
        q = c_lp if c_lp.dim() == 2 else c_lp.unsqueeze(0).repeat(batch_size, 1)
    else:
        q = c_lp

    out = DenseBatchQP(P=P, q=q, A=A_qp, l=l, u=u)
    qp_diag = dict(diag)
    qp_diag.update({"u_cap": float(u_cap), "qp_n": int(n), "qp_m": int(A_qp.shape[0])})
    return out, qp_diag

