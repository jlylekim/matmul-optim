from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

SOLVER_LABELS = {
    "gemm_ipm_ns": "ns_ipm",
    "gemm_ipm_robust": "ns_ipm_rb",
    "scipy_trust_constr_cpu": "scipy_trust_cpu",
    "osqp_cpu": "osqp_cpu",
    "gemm_splitting_qp": "split_qp",
    "lp_first_order_gpu": "lp_pdhg_gpu",
    "highs_cpu": "highs_cpu",
}

SOLVER_STYLE = {
    "gemm_ipm_ns": {"color": "#1f77b4", "marker": "o"},
    "gemm_ipm_robust": {"color": "#d62728", "marker": "s"},
    "gemm_splitting_qp": {"color": "#2ca02c", "marker": "^"},
    "lp_first_order_gpu": {"color": "#ff7f0e", "marker": "D"},
    "scipy_trust_constr_cpu": {"color": "#9467bd", "marker": "P"},
    "osqp_cpu": {"color": "#8c564b", "marker": "X"},
    "highs_cpu": {"color": "#17becf", "marker": "v"},
}


def _load_df(path: str | Path) -> pd.DataFrame:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
    if not rows:
        return pd.DataFrame()

    df = pd.json_normalize(rows)
    if "timing.median_s" in df.columns:
        df["median_s"] = pd.to_numeric(df["timing.median_s"], errors="coerce")
    return df


def _numeric_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def _solver_label(solver: str) -> str:
    return SOLVER_LABELS.get(solver, solver)


def _solver_style(solver: str) -> dict[str, str]:
    return SOLVER_STYLE.get(solver, {"color": "#7f7f7f", "marker": "o"})


def _filter_df(
    df: pd.DataFrame,
    *,
    category: str,
    include_solvers: list[str] | None,
    exclude_solvers: list[str] | None,
) -> pd.DataFrame:
    out = df
    if category != "all" and "category" in out.columns:
        out = out[out["category"] == category]
    if include_solvers:
        out = out[out["solver"].isin(include_solvers)]
    if exclude_solvers:
        out = out[~out["solver"].isin(exclude_solvers)]
    return out


def write_accuracy_time_summary(df: pd.DataFrame, out_dir: Path) -> None:
    if df.empty:
        return

    rows = []
    for solver, g in df.groupby("solver"):
        times = _numeric_series(g, "median_s").dropna()
        primal = _numeric_series(g, "metrics.primal_residual").dropna()
        dual = _numeric_series(g, "metrics.dual_residual").dropna()
        gap = _numeric_series(g, "metrics.duality_gap").dropna()
        solved = g["status"].astype(str).str.lower().eq("solved") if "status" in g.columns else pd.Series(dtype=bool)

        rows.append(
            {
                "solver": _solver_label(solver),
                "n": int(len(g)),
                "solved": int(solved.sum()) if not solved.empty else 0,
                "t_med_s": float(np.median(times)) if len(times) > 0 else np.nan,
                "t_mean_s": float(times.mean()) if len(times) > 0 else np.nan,
                "t_std_s": float(times.std(ddof=0)) if len(times) > 1 else 0.0,
                "p_mean": float(primal.mean()) if len(primal) > 0 else np.nan,
                "p_std": float(primal.std(ddof=0)) if len(primal) > 1 else 0.0,
                "d_mean": float(dual.mean()) if len(dual) > 0 else np.nan,
                "d_std": float(dual.std(ddof=0)) if len(dual) > 1 else 0.0,
                "g_mean": float(gap.mean()) if len(gap) > 0 else np.nan,
                "g_std": float(gap.std(ddof=0)) if len(gap) > 1 else 0.0,
            }
        )

    sdf = pd.DataFrame(rows).sort_values("t_med_s")
    sdf.to_markdown(out_dir / "summary_table.md", index=False, floatfmt=".4g")
    (out_dir / "summary_table.tex").write_text(
        sdf.to_latex(index=False, float_format=lambda x: f"{x:.4g}"),
        encoding="utf-8",
    )


def plot_accuracy_vs_time(df: pd.DataFrame, out_dir: Path) -> None:
    if df.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    metrics = [
        ("metrics.primal_residual", "Primal residual"),
        ("metrics.dual_residual", "Dual residual"),
    ]

    solver_order = sorted(df["solver"].dropna().unique().tolist())
    for ax, (metric_col, metric_label) in zip(axes, metrics):
        for solver in solver_order:
            sdf = df[df["solver"] == solver]
            x = _numeric_series(sdf, "median_s").to_numpy()
            y = _numeric_series(sdf, metric_col).to_numpy()
            valid = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
            if not np.any(valid):
                continue
            xv = x[valid]
            yv = y[valid]
            order = np.argsort(xv)
            style = _solver_style(solver)
            ax.plot(
                xv[order],
                yv[order],
                marker=style["marker"],
                linestyle="-",
                linewidth=1.5,
                markersize=5,
                color=style["color"],
                label=_solver_label(solver),
            )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(metric_label)
        ax.grid(True, alpha=0.25)

    axes[0].set_title("Primal Accuracy vs Time")
    axes[1].set_title("Dual Accuracy vs Time")
    legend_handles: list[Line2D] = []
    for solver in solver_order:
        style = _solver_style(solver)
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=style["color"],
                marker=style["marker"],
                linestyle="-",
                linewidth=1.5,
                markersize=5,
                label=_solver_label(solver),
            )
        )
    if legend_handles:
        fig.legend(
            handles=legend_handles,
            loc="upper center",
            ncol=min(4, len(legend_handles)),
            title="Method",
            frameon=True,
            framealpha=0.9,
        )
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(out_dir / "accuracy_vs_time.pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create benchmark plots")
    parser.add_argument("--input", type=str, required=True, help="Path to results.jsonl")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument(
        "--category",
        type=str,
        choices=["all", "qp_conic", "lp"],
        default="all",
        help="Optional category filter before plotting",
    )
    parser.add_argument(
        "--include-solvers",
        nargs="*",
        default=None,
        help="Only include these solver ids",
    )
    parser.add_argument(
        "--exclude-solvers",
        nargs="*",
        default=None,
        help="Exclude these solver ids",
    )
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = _load_df(args.input)
    if df.empty:
        print("No rows found in results file")
        return
    df = _filter_df(
        df,
        category=args.category,
        include_solvers=args.include_solvers,
        exclude_solvers=args.exclude_solvers,
    )
    if df.empty:
        print("No rows left after filters")
        return

    write_accuracy_time_summary(df, out_dir)
    plot_accuracy_vs_time(df, out_dir)


if __name__ == "__main__":
    main()
