from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _load_df(path: str | Path) -> pd.DataFrame:
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
    if not rows:
        return pd.DataFrame()
    return pd.json_normalize(rows)


def _numeric(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def _write_tables(df: pd.DataFrame, out_dir: Path) -> None:
    rows = []
    for solver, g in df.groupby("solver"):
        status_series = g["status"].astype(str).str.lower() if "status" in g.columns else pd.Series(dtype=str)
        times = _numeric(g, "timing.median_s").dropna()
        primal = _numeric(g, "metrics.primal_residual").dropna()
        dual = _numeric(g, "metrics.dual_residual").dropna()
        gap = _numeric(g, "metrics.duality_gap").dropna()
        rows.append(
            {
                "solver": solver,
                "count": int(len(g)),
                "solved": int((status_series == "solved").sum()),
                "max_iters": int((status_series == "max_iters").sum()),
                "failed": int((status_series == "failed").sum()),
                "timeout": int((status_series == "timeout").sum()),
                "median_time_s": float(times.median()) if len(times) > 0 else np.nan,
                "mean_time_s": float(times.mean()) if len(times) > 0 else np.nan,
                "std_time_s": float(times.std(ddof=0)) if len(times) > 1 else 0.0,
                "mean_primal_residual": float(primal.mean()) if len(primal) > 0 else np.nan,
                "mean_dual_residual": float(dual.mean()) if len(dual) > 0 else np.nan,
                "mean_duality_gap": float(gap.mean()) if len(gap) > 0 else np.nan,
            }
        )

    summary = pd.DataFrame(rows).sort_values(["solved", "median_time_s"], ascending=[False, True])
    summary.to_markdown(out_dir / "summary_table.md", index=False, floatfmt=".6g")
    (out_dir / "summary_table.tex").write_text(
        summary.to_latex(index=False, float_format=lambda x: f"{x:.6g}"),
        encoding="utf-8",
    )

    if "metadata.suite" in df.columns:
        per_suite_rows = []
        for (suite, solver), g in df.groupby(["metadata.suite", "solver"]):
            status_series = g["status"].astype(str).str.lower()
            times = _numeric(g, "timing.median_s").dropna()
            per_suite_rows.append(
                {
                    "suite": suite,
                    "solver": solver,
                    "count": int(len(g)),
                    "solved": int((status_series == "solved").sum()),
                    "failed": int((status_series == "failed").sum()),
                    "timeout": int((status_series == "timeout").sum()),
                    "median_time_s": float(times.median()) if len(times) > 0 else np.nan,
                }
            )
        per_suite = pd.DataFrame(per_suite_rows).sort_values(["suite", "solved", "median_time_s"], ascending=[True, False, True])
        per_suite.to_markdown(out_dir / "summary_by_suite.md", index=False, floatfmt=".6g")


def _plot_time_vs_residual(df: pd.DataFrame, out_dir: Path) -> None:
    if "metadata.suite" not in df.columns:
        return
    suites = sorted(df["metadata.suite"].dropna().astype(str).unique().tolist())
    if not suites:
        return
    for suite in suites:
        sdf = df[df["metadata.suite"] == suite]
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for solver, g in sdf.groupby("solver"):
            t = _numeric(g, "timing.median_s").to_numpy()
            p = _numeric(g, "metrics.primal_residual").to_numpy()
            d = _numeric(g, "metrics.dual_residual").to_numpy()
            vp = np.isfinite(t) & np.isfinite(p) & (t > 0) & (p > 0)
            vd = np.isfinite(t) & np.isfinite(d) & (t > 0) & (d > 0)
            if np.any(vp):
                axes[0].scatter(t[vp], p[vp], label=solver, s=20, alpha=0.8)
            if np.any(vd):
                axes[1].scatter(t[vd], d[vd], label=solver, s=20, alpha=0.8)
        axes[0].set_xscale("log")
        axes[0].set_yscale("log")
        axes[1].set_xscale("log")
        axes[1].set_yscale("log")
        axes[0].set_xlabel("time (s)")
        axes[0].set_ylabel("primal residual")
        axes[1].set_xlabel("time (s)")
        axes[1].set_ylabel("dual residual")
        axes[0].set_title(f"{suite}: primal vs time")
        axes[1].set_title(f"{suite}: dual vs time")
        handles, labels = axes[1].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper center", ncol=min(4, len(labels)), title="solver")
        fig.tight_layout(rect=(0, 0, 1, 0.9))
        fig.savefig(out_dir / f"accuracy_vs_time_{suite}.pdf")
        plt.close(fig)


def _plot_solved_rate(df: pd.DataFrame, out_dir: Path) -> None:
    if "metadata.suite" not in df.columns:
        return
    plot_df = df.copy()
    plot_df["solved_flag"] = plot_df["status"].astype(str).str.lower().eq("solved").astype(float)
    pivot = plot_df.pivot_table(index="solver", columns="metadata.suite", values="solved_flag", aggfunc="mean")
    if pivot.empty:
        return
    ax = pivot.plot(kind="bar", figsize=(10, 5))
    ax.set_ylabel("solved rate")
    ax.set_xlabel("solver")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Solved rate by suite")
    ax.legend(title="suite")
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_dir / "solved_rate_by_suite.pdf")
    plt.close()


def _plot_size_bucket_runtime(df: pd.DataFrame, out_dir: Path) -> None:
    if "metadata.n" not in df.columns:
        return
    plot_df = df.copy()
    plot_df["n"] = _numeric(plot_df, "metadata.n")
    plot_df["time"] = _numeric(plot_df, "timing.median_s")
    plot_df = plot_df[np.isfinite(plot_df["n"]) & np.isfinite(plot_df["time"]) & (plot_df["time"] > 0)]
    if plot_df.empty:
        return
    bins = [0, 100, 250, 500, 1000, 2500, 5000, np.inf]
    labels = ["<=100", "101-250", "251-500", "501-1k", "1k-2.5k", "2.5k-5k", ">5k"]
    plot_df["size_bucket"] = pd.cut(plot_df["n"], bins=bins, labels=labels, include_lowest=True)
    agg = (
        plot_df.groupby(["size_bucket", "solver"], dropna=False)["time"]
        .median()
        .reset_index()
        .pivot(index="size_bucket", columns="solver", values="time")
    )
    if agg.empty:
        return
    ax = agg.plot(marker="o", figsize=(11, 5))
    ax.set_yscale("log")
    ax.set_ylabel("median time (s)")
    ax.set_xlabel("problem size bucket (n)")
    ax.set_title("Runtime by problem size bucket")
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_dir / "runtime_by_size_bucket.pdf")
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate real LP benchmark JSONL outputs")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = _load_df(args.input)
    if df.empty:
        print("No rows found in results file")
        return
    # Exclude case-level markers from numerical summaries.
    df = df[df["solver"].astype(str) != "__case_failure__"]
    if df.empty:
        print("No solver rows left after filtering case markers")
        return
    _write_tables(df, out_dir)
    _plot_time_vs_residual(df, out_dir)
    _plot_solved_rate(df, out_dir)
    _plot_size_bucket_runtime(df, out_dir)
    print(f"Wrote aggregate outputs to {out_dir}")


if __name__ == "__main__":
    main()

