from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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
                "solver": solver,
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

    for ax, (metric_col, metric_label) in zip(axes, metrics):
        for solver, sdf in df.groupby("solver"):
            x = _numeric_series(sdf, "median_s").to_numpy()
            y = _numeric_series(sdf, metric_col).to_numpy()
            valid = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
            if not np.any(valid):
                continue
            xv = x[valid]
            yv = y[valid]
            order = np.argsort(xv)
            ax.plot(xv[order], yv[order], marker="o", linestyle="-", label=solver)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(metric_label)
        ax.grid(True, alpha=0.25)

    axes[0].set_title("Primal Accuracy vs Time")
    axes[1].set_title("Dual Accuracy vs Time")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(4, len(labels)))
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out_dir / "accuracy_vs_time.pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create benchmark plots")
    parser.add_argument("--input", type=str, required=True, help="Path to results.jsonl")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = _load_df(args.input)
    if df.empty:
        print("No rows found in results file")
        return

    write_accuracy_time_summary(df, out_dir)
    plot_accuracy_vs_time(df, out_dir)


if __name__ == "__main__":
    main()
