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
    for solver, g in df.groupby("solver"):
        times = g["median_s"].astype(float).to_numpy()
        rows.append(
            {
                "solver": solver,
                "count": int(len(g)),
                "median_time_s": float(np.median(times)),
                "shifted_geo_mean_time_s": _shifted_geo_mean(times),
                "mean_primal_residual": float(g["metrics.primal_residual"].astype(float).mean())
                if "metrics.primal_residual" in g.columns
                else np.nan,
                "mean_dual_residual": float(g["metrics.dual_residual"].astype(float).mean())
                if "metrics.dual_residual" in g.columns
                else np.nan,
            }
        )

    sdf = pd.DataFrame(rows).sort_values("median_time_s")
    sdf.to_markdown(out_dir / "summary_table.md", index=False)
    (out_dir / "summary_table.tex").write_text(
        sdf.to_latex(index=False, float_format=lambda x: f"{x:.4g}"),
        encoding="utf-8",
    )


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


if __name__ == "__main__":
    main()
