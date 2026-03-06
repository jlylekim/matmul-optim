from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import scipy.sparse as sp


@dataclass
class CanonicalLP:
    name: str
    suite: str
    sense: Literal["min", "max"]
    A_csr: sp.csr_matrix
    c: np.ndarray
    row_l: np.ndarray
    row_u: np.ndarray
    col_l: np.ndarray
    col_u: np.ndarray
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def n(self) -> int:
        return int(self.c.shape[0])

    @property
    def m(self) -> int:
        return int(self.row_l.shape[0])

    @property
    def nnz(self) -> int:
        return int(self.A_csr.nnz)

    @property
    def density(self) -> float:
        if self.m == 0 or self.n == 0:
            return 0.0
        return float(self.nnz / float(self.m * self.n))


def _require_highspy():
    try:
        import highspy  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "highspy is required for parsing MPS/SIF. Install with `pip install highspy`."
        ) from exc
    return highspy


def _to_np_f64(seq: Any) -> np.ndarray:
    return np.asarray(seq, dtype=np.float64)


def _extract_lp_matrix_csr(lp: Any, m: int, n: int) -> sp.csr_matrix:
    mat = getattr(lp, "a_matrix_", None)
    if mat is None:
        raise RuntimeError("Could not access Highs LP sparse matrix `a_matrix_`")
    start = np.asarray(getattr(mat, "start_", []), dtype=np.int64)
    index = np.asarray(getattr(mat, "index_", []), dtype=np.int64)
    value = _to_np_f64(getattr(mat, "value_", []))
    if start.size != n + 1:
        raise RuntimeError(
            f"Unexpected Highs matrix start_ length: got={start.size}, expected={n + 1}"
        )
    csc = sp.csc_matrix((value, index, start), shape=(m, n))
    return csc.tocsr()


def _extract_objective_sense(highs: Any, lp: Any) -> Literal["min", "max"]:
    try:
        obj_sense = highs.getObjectiveSense()
        if "max" in str(obj_sense).lower():
            return "max"
    except Exception:
        pass
    try:
        lp_sense = getattr(lp, "sense_", None)
        if lp_sense is not None and "max" in str(lp_sense).lower():
            return "max"
    except Exception:
        pass
    return "min"


def parse_lp_file(path: str | Path, *, suite: str = "unknown", name: str | None = None) -> CanonicalLP:
    highspy = _require_highspy()
    file_path = Path(path)

    highs = highspy.Highs()
    status = highs.readModel(str(file_path))
    if "ok" not in str(status).lower():
        raise RuntimeError(f"HiGHS failed to read model `{file_path}` with status `{status}`")

    lp = highs.getLp()
    n = int(getattr(lp, "num_col_", len(getattr(lp, "col_cost_", []))))
    m = int(getattr(lp, "num_row_", len(getattr(lp, "row_lower_", []))))

    c = _to_np_f64(getattr(lp, "col_cost_", []))
    row_l = _to_np_f64(getattr(lp, "row_lower_", []))
    row_u = _to_np_f64(getattr(lp, "row_upper_", []))
    col_l = _to_np_f64(getattr(lp, "col_lower_", []))
    col_u = _to_np_f64(getattr(lp, "col_upper_", []))
    A_csr = _extract_lp_matrix_csr(lp, m=m, n=n)
    sense = _extract_objective_sense(highs, lp)

    lp_name = name or file_path.stem
    out = CanonicalLP(
        name=lp_name,
        suite=suite,
        sense=sense,
        A_csr=A_csr,
        c=c,
        row_l=row_l,
        row_u=row_u,
        col_l=col_l,
        col_u=col_u,
        meta={"source_path": str(file_path), "parser": "highspy"},
    )
    validate_canonical_lp(out)
    return out


def validate_canonical_lp(lp: CanonicalLP) -> None:
    m, n = lp.A_csr.shape
    if n != lp.c.shape[0]:
        raise ValueError(f"c has length {lp.c.shape[0]} but A has {n} columns")
    if m != lp.row_l.shape[0] or m != lp.row_u.shape[0]:
        raise ValueError(f"row bounds shape mismatch: m={m}, row_l={lp.row_l.shape}, row_u={lp.row_u.shape}")
    if n != lp.col_l.shape[0] or n != lp.col_u.shape[0]:
        raise ValueError(f"col bounds shape mismatch: n={n}, col_l={lp.col_l.shape}, col_u={lp.col_u.shape}")
    if lp.A_csr.indptr.size != m + 1:
        raise ValueError("CSR indptr length mismatch")
    if int(lp.A_csr.indptr[-1]) != lp.A_csr.nnz:
        raise ValueError("CSR indptr terminal nnz mismatch")
    if not np.all(np.isfinite(lp.c)):
        raise ValueError("objective vector c contains non-finite values")
    if not np.all(np.isfinite(lp.A_csr.data)):
        raise ValueError("constraint matrix A contains non-finite values")
    # Bounds may contain infinities but should not contain NaNs.
    for name, arr in [("row_l", lp.row_l), ("row_u", lp.row_u), ("col_l", lp.col_l), ("col_u", lp.col_u)]:
        if np.isnan(arr).any():
            raise ValueError(f"{name} contains NaN values")


def save_canonical_lp_npz(lp: CanonicalLP, path: str | Path) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "name": lp.name,
        "suite": lp.suite,
        "sense": lp.sense,
        "A_data": lp.A_csr.data.astype(np.float64),
        "A_indices": lp.A_csr.indices.astype(np.int64),
        "A_indptr": lp.A_csr.indptr.astype(np.int64),
        "A_shape": np.asarray(lp.A_csr.shape, dtype=np.int64),
        "c": lp.c.astype(np.float64),
        "row_l": lp.row_l.astype(np.float64),
        "row_u": lp.row_u.astype(np.float64),
        "col_l": lp.col_l.astype(np.float64),
        "col_u": lp.col_u.astype(np.float64),
        "meta_json": json.dumps(lp.meta, sort_keys=True),
    }
    np.savez_compressed(out_path, **payload)


def load_canonical_lp_npz(path: str | Path) -> CanonicalLP:
    in_path = Path(path)
    with np.load(in_path, allow_pickle=False) as npz:
        shape_arr = np.asarray(npz["A_shape"], dtype=np.int64)
        shape = (int(shape_arr[0]), int(shape_arr[1]))
        A = sp.csr_matrix(
            (
                np.asarray(npz["A_data"], dtype=np.float64),
                np.asarray(npz["A_indices"], dtype=np.int64),
                np.asarray(npz["A_indptr"], dtype=np.int64),
            ),
            shape=shape,
        )
        lp = CanonicalLP(
            name=str(npz["name"]),
            suite=str(npz["suite"]),
            sense=str(npz["sense"]).lower(),
            A_csr=A,
            c=np.asarray(npz["c"], dtype=np.float64),
            row_l=np.asarray(npz["row_l"], dtype=np.float64),
            row_u=np.asarray(npz["row_u"], dtype=np.float64),
            col_l=np.asarray(npz["col_l"], dtype=np.float64),
            col_u=np.asarray(npz["col_u"], dtype=np.float64),
            meta=json.loads(str(npz["meta_json"])) if "meta_json" in npz else {},
        )
    validate_canonical_lp(lp)
    return lp


def cache_path_for(
    *,
    cache_root: str | Path,
    suite: str,
    instance_name: str,
) -> Path:
    safe_name = instance_name.replace("/", "_")
    return Path(cache_root) / suite / f"{safe_name}.npz"


def load_canonical_lp(
    *,
    expanded_path: str | Path,
    suite: str,
    instance_name: str,
    cache_root: str | Path = "benchmarks/datasets/cache",
    force_reparse: bool = False,
) -> CanonicalLP:
    cpath = cache_path_for(cache_root=cache_root, suite=suite, instance_name=instance_name)
    if cpath.exists() and not force_reparse:
        return load_canonical_lp_npz(cpath)
    lp = parse_lp_file(expanded_path, suite=suite, name=instance_name)
    save_canonical_lp_npz(lp, cpath)
    return lp


def load_manifest_rows(
    manifest_path: str | Path,
    *,
    suites: list[str] | None = None,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    suites_set = set(suites or [])
    with Path(manifest_path).open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if suites_set and row.get("suite", "") not in suites_set:
                continue
            rows.append(dict(row))
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parse LP manifest instances into cached canonical format")
    parser.add_argument("--manifest", type=str, default="benchmarks/datasets/real_lp_manifest.csv")
    parser.add_argument("--suites", nargs="*", default=None, choices=["netlib", "stochlp", "misc"])
    parser.add_argument("--cache-root", type=str, default="benchmarks/datasets/cache")
    parser.add_argument("--force-reparse", action="store_true")
    parser.add_argument("--max-instances", type=int, default=0, help="0 means all")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_manifest_rows(args.manifest, suites=args.suites)
    if int(args.max_instances) > 0:
        rows = rows[: int(args.max_instances)]
    print(f"[cache] instances={len(rows)}")

    ok = 0
    failed = 0
    for row in rows:
        suite = str(row["suite"])
        name = str(row["instance_name"])
        expanded_path = str(row["expanded_path"])
        try:
            lp = load_canonical_lp(
                expanded_path=expanded_path,
                suite=suite,
                instance_name=name,
                cache_root=args.cache_root,
                force_reparse=bool(args.force_reparse),
            )
            ok += 1
            print(
                f"[cache] ok suite={suite} instance={name} m={lp.m} n={lp.n} nnz={lp.nnz} density={lp.density:.3e}"
            )
        except Exception as exc:
            failed += 1
            print(f"[cache] failed suite={suite} instance={name} err={type(exc).__name__}: {exc}")

    print(f"[cache] done ok={ok} failed={failed}")


if __name__ == "__main__":
    main()
