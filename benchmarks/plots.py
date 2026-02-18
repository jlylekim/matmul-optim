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
        df["median_s"] = df["timing.median_s"].astype(float)
    return df


def _shifted_geo_mean(values: np.ndarray, shift: float = 1e-3) -> float:
    return float(np.exp(np.mean(np.log(values + shift))) - shift)


def _numeric_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def plot_performance_profile(df: pd.DataFrame, out_dir: Path) -> None:
    if df.empty:
        return

    pivot = df.pivot_table(index="problem_id", columns="solver", values="median_s", aggfunc="min")
    pivot = pivot.dropna(axis=0, how="any")
    if pivot.empty:
        return

    best = pivot.min(axis=1)
    ratios = pivot.div(best, axis=0)

    taus = np.linspace(1.0, max(3.0, float(ratios.max().max())), 200)

    plt.figure(figsize=(8, 5))
    for solver in ratios.columns:
        y = [(ratios[solver] <= t).mean() for t in taus]
        plt.plot(taus, y, label=solver)
    plt.xlabel("Performance ratio")
    plt.ylabel("Solved fraction")
    plt.title("Performance Profile")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "performance_profile.pdf")
    plt.close()


def plot_accuracy_vs_time(df: pd.DataFrame, out_dir: Path) -> None:
    if df.empty or "metrics.primal_residual" not in df.columns:
        return

    plt.figure(figsize=(8, 5))
    for solver, sdf in df.groupby("solver"):
        x = sdf["median_s"].to_numpy()
        y = sdf["metrics.primal_residual"].astype(float).to_numpy()
        order = np.argsort(x)
        plt.plot(x[order], y[order], marker="o", linestyle="-", label=solver)
    plt.yscale("log")
    plt.xscale("log")
    plt.xlabel("Time (s)")
    plt.ylabel("Primal residual")
    plt.title("Accuracy vs Time")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "accuracy_vs_time.pdf")
    plt.close()


def plot_scaling(df: pd.DataFrame, out_dir: Path) -> None:
    if df.empty or "metadata.world_size" not in df.columns:
        return

    sdf = df[df["metadata.world_size"].notna()]
    if sdf.empty:
        return

    plt.figure(figsize=(8, 5))
    for solver, g in sdf.groupby("solver"):
        world = g["metadata.world_size"].astype(int)
        thr = g["metadata.throughput_prob_per_s"].astype(float)
        agg = g.assign(world=world, thr=thr).groupby("world")["thr"].mean().reset_index()
        plt.plot(agg["world"], agg["thr"], marker="o", label=solver)

    plt.xlabel("#GPUs")
    plt.ylabel("Throughput (problems/s)")
    plt.title("Scaling Curve")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "scaling_curve.pdf")
    plt.close()


def plot_workload_scaling(df: pd.DataFrame, out_dir: Path) -> None:
    if df.empty:
        return

    dimensions = [
        ("metadata.config.n", "n"),
        ("metadata.config.batch_size", "batch_size"),
        ("metadata.config.rhs_count", "rhs_count"),
        ("metadata.world_size", "num_gpus"),
    ]

    for col, label in dimensions:
        if col not in df.columns:
            continue
        sdf = df[df[col].notna()].copy()
        if sdf.empty:
            continue

        sdf["x"] = pd.to_numeric(sdf[col], errors="coerce")
        sdf["thr"] = pd.to_numeric(sdf["metadata.throughput_prob_per_s"], errors="coerce")
        sdf = sdf.dropna(subset=["x", "thr"])
        if sdf.empty:
            continue

        plt.figure(figsize=(8, 5))
        for solver, g in sdf.groupby("solver"):
            agg = g.groupby("x")["thr"].mean().reset_index().sort_values("x")
            plt.plot(agg["x"], agg["thr"], marker="o", label=solver)
        plt.xlabel(label)
        plt.ylabel("Throughput (problems/s)")
        plt.title(f"Throughput vs {label}")
        plt.grid(True, alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"scaling_{label}.pdf")
        plt.close()


def plot_memory_vs_throughput(df: pd.DataFrame, out_dir: Path) -> None:
    if df.empty:
        return

    if "system.gpu_memory_gb" not in df.columns:
        return

    plt.figure(figsize=(8, 5))
    for solver, g in df.groupby("solver"):
        mem = g["system.gpu_memory_gb"].astype(float)
        thr = g["metadata.throughput_prob_per_s"].astype(float)
        plt.scatter(mem, thr, label=solver, alpha=0.7)

    plt.xlabel("GPU memory (GB)")
    plt.ylabel("Throughput (problems/s)")
    plt.title("Memory vs Throughput")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "memory_vs_throughput.pdf")
    plt.close()


def write_summary_tables(df: pd.DataFrame, out_dir: Path) -> None:
    if df.empty:
        return

    rows = []
    tier_cols = ["metadata.tier_pass.1e-2", "metadata.tier_pass.1e-3", "metadata.tier_pass.1e-4"]
    for solver, g in df.groupby("solver"):
        times = _numeric_series(g, "median_s").dropna()
        primal = _numeric_series(g, "metrics.primal_residual").dropna()
        dual = _numeric_series(g, "metrics.dual_residual").dropna()
        gap = _numeric_series(g, "metrics.duality_gap").dropna()
        iters = _numeric_series(g, "metadata.solver_iterations_mean").dropna()
        world = _numeric_series(g, "metadata.world_size").dropna()
        stopping_pass = _numeric_series(g, "metadata.stopping_pass").fillna(0).astype(bool)
        solved = g["status"] == "solved" if "status" in g.columns else pd.Series(dtype=bool)
        max_iters = g["status"] == "max_iters" if "status" in g.columns else pd.Series(dtype=bool)

        tier_counts: dict[str, int] = {}
        for c in tier_cols:
            key = c.split(".")[-1]
            if c in g.columns:
                tier_counts[f"pass_{key}"] = int(pd.to_numeric(g[c], errors="coerce").fillna(0).astype(bool).sum())
            else:
                tier_counts[f"pass_{key}"] = 0
        rows.append(
            {
                "solver": solver,
                "count": int(len(g)),
                "solved_status_count": int(solved.sum()) if not solved.empty else 0,
                "max_iters_status_count": int(max_iters.sum()) if not max_iters.empty else 0,
                "stopping_pass_count": int(stopping_pass.sum()) if not stopping_pass.empty else 0,
                "median_time_s": float(np.median(times)) if len(times) > 0 else np.nan,
                "mean_time_s": float(times.mean()) if len(times) > 0 else np.nan,
                "std_time_s": float(times.std(ddof=0)) if len(times) > 1 else 0.0,
                "shifted_geo_mean_time_s": _shifted_geo_mean(times.to_numpy()) if len(times) > 0 else np.nan,
                "mean_iterations": float(iters.mean()) if len(iters) > 0 else np.nan,
                "std_iterations": float(iters.std(ddof=0)) if len(iters) > 1 else 0.0,
                "mean_world_size": float(world.mean()) if len(world) > 0 else np.nan,
                "mean_primal_residual": float(primal.mean()) if len(primal) > 0 else np.nan,
                "std_primal_residual": float(primal.std(ddof=0)) if len(primal) > 1 else 0.0,
                "mean_dual_residual": float(dual.mean()) if len(dual) > 0 else np.nan,
                "std_dual_residual": float(dual.std(ddof=0)) if len(dual) > 1 else 0.0,
                "mean_duality_gap": float(gap.mean()) if len(gap) > 0 else np.nan,
                "std_duality_gap": float(gap.std(ddof=0)) if len(gap) > 1 else 0.0,
                **tier_counts,
            }
        )

    sdf = pd.DataFrame(rows).sort_values("median_time_s")
    sdf.to_markdown(out_dir / "summary_table.md", index=False)
    (out_dir / "summary_table.tex").write_text(
        sdf.to_latex(index=False, float_format=lambda x: f"{x:.4g}"),
        encoding="utf-8",
    )


def write_accuracy_iterations_table(df: pd.DataFrame, out_dir: Path) -> None:
    if df.empty:
        return

    rows = []
    for solver, g in df.groupby("solver"):
        primal = _numeric_series(g, "metrics.primal_residual")
        dual = _numeric_series(g, "metrics.dual_residual")
        gap = _numeric_series(g, "metrics.duality_gap")
        iters = _numeric_series(g, "metadata.solver_iterations_mean")

        rows.append(
            {
                "solver": solver,
                "primal_mean": float(primal.mean()) if not primal.dropna().empty else np.nan,
                "primal_std": float(primal.std(ddof=0)) if not primal.dropna().empty else np.nan,
                "dual_mean": float(dual.mean()) if not dual.dropna().empty else np.nan,
                "dual_std": float(dual.std(ddof=0)) if not dual.dropna().empty else np.nan,
                "gap_mean": float(gap.mean()) if not gap.dropna().empty else np.nan,
                "gap_std": float(gap.std(ddof=0)) if not gap.dropna().empty else np.nan,
                "iter_mean": float(iters.mean()) if not iters.dropna().empty else np.nan,
                "iter_std": float(iters.std(ddof=0)) if not iters.dropna().empty else np.nan,
            }
        )

    sdf = pd.DataFrame(rows).sort_values("solver")
    sdf.to_markdown(out_dir / "accuracy_iterations_table.md", index=False)
    (out_dir / "accuracy_iterations_table.tex").write_text(
        sdf.to_latex(index=False, float_format=lambda x: f"{x:.4g}"),
        encoding="utf-8",
    )


def plot_accuracy_iterations_errorbars(df: pd.DataFrame, out_dir: Path) -> None:
    if df.empty:
        return

    summary_rows = []
    for solver, g in df.groupby("solver"):
        primal = _numeric_series(g, "metrics.primal_residual").dropna()
        dual = _numeric_series(g, "metrics.dual_residual").dropna()
        gap = _numeric_series(g, "metrics.duality_gap").dropna()
        iters = _numeric_series(g, "metadata.solver_iterations_mean").dropna()
        summary_rows.append(
            {
                "solver": solver,
                "primal_mean": float(primal.mean()) if not primal.empty else np.nan,
                "primal_std": float(primal.std(ddof=0)) if len(primal) > 1 else 0.0,
                "dual_mean": float(dual.mean()) if not dual.empty else np.nan,
                "dual_std": float(dual.std(ddof=0)) if len(dual) > 1 else 0.0,
                "gap_mean": float(gap.mean()) if not gap.empty else np.nan,
                "gap_std": float(gap.std(ddof=0)) if len(gap) > 1 else 0.0,
                "iter_mean": float(iters.mean()) if not iters.empty else np.nan,
                "iter_std": float(iters.std(ddof=0)) if len(iters) > 1 else 0.0,
            }
        )

    sdf = pd.DataFrame(summary_rows).dropna(subset=["solver"])
    if sdf.empty:
        return

    x = np.arange(len(sdf))
    labels = sdf["solver"].tolist()

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    plots = [
        ("primal_mean", "primal_std", "Primal Residual"),
        ("dual_mean", "dual_std", "Dual Residual"),
        ("gap_mean", "gap_std", "Duality Gap"),
        ("iter_mean", "iter_std", "Iterations"),
    ]
    for ax, (mcol, ecol, title) in zip(axes, plots):
        y = pd.to_numeric(sdf[mcol], errors="coerce").to_numpy()
        yerr = pd.to_numeric(sdf[ecol], errors="coerce").to_numpy()
        ax.errorbar(x, y, yerr=yerr, fmt="o", capsize=4)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
        if "Residual" in title or "Gap" in title:
            positive = y[y > 0]
            if len(positive) > 0:
                ax.set_yscale("log")

    fig.tight_layout()
    fig.savefig(out_dir / "accuracy_iterations_errorbars.pdf")
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

    plot_performance_profile(df, out_dir)
    plot_accuracy_vs_time(df, out_dir)
    plot_scaling(df, out_dir)
    plot_workload_scaling(df, out_dir)
    plot_memory_vs_throughput(df, out_dir)
    write_summary_tables(df, out_dir)
    write_accuracy_iterations_table(df, out_dir)
    plot_accuracy_iterations_errorbars(df, out_dir)


if __name__ == "__main__":
    main()
