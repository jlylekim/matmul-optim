from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Callable

import torch

from gemm_kkt.solvers.common import SolverResult, sync_if_cuda
from gemm_kkt.solvers.types import DenseBatchLP, DenseBatchQP
from gemm_kkt.utils.distributed import (
    DistInfo,
    barrier_if_distributed,
    gather_objects,
    init_distributed,
    shard_lp,
    shard_qp,
)
from gemm_kkt.utils.system import system_info

SolveQPFn = Callable[[DenseBatchQP], SolverResult]
SolveLPFn = Callable[[DenseBatchLP], SolverResult]


@dataclass
class TimingSummary:
    median_s: float
    p90_s: float
    repeats: int


@dataclass
class BenchmarkRecord:
    run_id: str
    category: str
    problem_id: str
    solver: str
    status: str
    timing: TimingSummary
    metrics: dict[str, Any]
    metadata: dict[str, Any]
    system: dict[str, Any]


DEFAULT_TIERS = [1e-2, 1e-3, 1e-4]


def _tier_name(tol: float) -> str:
    return f"{tol:.0e}".replace("e-0", "e-").replace("e+0", "e+")


def _as_float_or_inf(v: Any) -> float:
    try:
        x = float(v)
        if math.isfinite(x):
            return x
        return float("inf")
    except Exception:
        return float("inf")


def _compute_qp_tier_pass(metrics: dict[str, Any], tiers: list[float] | None = None) -> tuple[dict[str, bool], str | None]:
    ts = tiers or DEFAULT_TIERS
    p = _as_float_or_inf(metrics.get("primal_residual", float("inf")))
    d = _as_float_or_inf(metrics.get("dual_residual", float("inf")))
    g = _as_float_or_inf(metrics.get("duality_gap", float("inf")))
    use_gap = math.isfinite(g)

    out: dict[str, bool] = {}
    for t in ts:
        ok = p <= t and d <= t and ((g <= t) if use_gap else True)
        out[_tier_name(t)] = bool(ok)

    best = None
    for t in sorted(ts):
        if out[_tier_name(t)]:
            best = _tier_name(t)
            break
    return out, best


def _compute_lp_tier_pass(metrics: dict[str, Any], tiers: list[float] | None = None) -> tuple[dict[str, bool], str | None]:
    ts = tiers or DEFAULT_TIERS
    p = _as_float_or_inf(metrics.get("primal_residual", float("inf")))
    d = _as_float_or_inf(metrics.get("dual_residual", float("inf")))

    out: dict[str, bool] = {}
    for t in ts:
        out[_tier_name(t)] = bool(p <= t and d <= t)

    best = None
    for t in sorted(ts):
        if out[_tier_name(t)]:
            best = _tier_name(t)
            break
    return out, best


def _quantile(values: list[float], q: float) -> float:
    if len(values) == 1:
        return values[0]
    sorted_vals = sorted(values)
    idx = q * (len(sorted_vals) - 1)
    lo = int(idx)
    hi = min(lo + 1, len(sorted_vals) - 1)
    w = idx - lo
    return sorted_vals[lo] * (1 - w) + sorted_vals[hi] * w


def _timed_runs(
    solve_once: Callable[[], SolverResult],
    *,
    device: torch.device,
    warmups: int,
    repeats: int,
) -> tuple[list[float], list[dict[str, Any]], list[int], list[str], SolverResult]:
    last_result: SolverResult | None = None

    for _ in range(warmups):
        last_result = solve_once()

    times: list[float] = []
    metrics_seq: list[dict[str, Any]] = []
    iterations_seq: list[int] = []
    status_seq: list[str] = []
    for _ in range(repeats):
        sync_if_cuda(device)
        t0 = time.perf_counter()
        last_result = solve_once()
        sync_if_cuda(device)
        times.append(time.perf_counter() - t0)
        metrics_seq.append(dict(last_result.metrics))
        iterations_seq.append(int(last_result.iterations))
        status_seq.append(str(last_result.status))

    assert last_result is not None
    return times, metrics_seq, iterations_seq, status_seq, last_result


def _mean_std_sem(values: list[float]) -> tuple[float, float, float]:
    if not values:
        return float("nan"), float("nan"), float("nan")
    mean = float(sum(values) / len(values))
    if len(values) == 1:
        return mean, 0.0, 0.0
    std = float(statistics.pstdev(values))
    sem = float(std / math.sqrt(len(values)))
    return mean, std, sem


def _summarize_metric_seq(metrics_seq: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, float], dict[str, float]]:
    if not metrics_seq:
        return {}, {}, {}
    keys = set().union(*(m.keys() for m in metrics_seq))
    mean_metrics: dict[str, Any] = {}
    std_metrics: dict[str, float] = {}
    sem_metrics: dict[str, float] = {}
    for k in keys:
        vals = [_as_float_or_inf(m.get(k)) for m in metrics_seq]
        finite = [v for v in vals if math.isfinite(v)]
        if not finite:
            mean_metrics[k] = metrics_seq[-1].get(k)
            continue
        mean, std, sem = _mean_std_sem(finite)
        mean_metrics[k] = mean
        std_metrics[k] = std
        sem_metrics[k] = sem
    return mean_metrics, std_metrics, sem_metrics


def _aggregate_rank_results(
    info: DistInfo,
    times: list[float],
    local_batch: int,
    metrics_seq: list[dict[str, Any]],
    iterations_seq: list[int],
    status_seq: list[str],
) -> tuple[list[float], int, dict[str, Any], dict[str, float], dict[str, float], list[float], list[str]]:
    payload = {
        "times": times,
        "local_batch": local_batch,
        "metrics_seq": metrics_seq,
        "iterations_seq": iterations_seq,
        "status_seq": status_seq,
    }
    gathered = gather_objects(info, payload)

    if not info.is_distributed:
        mean_metrics, std_metrics, sem_metrics = _summarize_metric_seq(metrics_seq)
        iters = [float(v) for v in iterations_seq]
        return times, local_batch, mean_metrics, std_metrics, sem_metrics, iters, status_seq

    repeats = len(times)
    wall_times: list[float] = []
    for i in range(repeats):
        wall_times.append(max(float(g["times"][i]) for g in gathered))

    total_batch = int(sum(int(g["local_batch"]) for g in gathered))
    total_weight = float(sum(float(g["local_batch"]) for g in gathered))

    per_repeat_metrics: list[dict[str, Any]] = []
    per_repeat_iters: list[float] = []
    per_repeat_status: list[str] = []

    for i in range(repeats):
        metric_keys = set().union(*(g["metrics_seq"][i].keys() for g in gathered))
        row: dict[str, Any] = {}
        for k in metric_keys:
            weighted_vals = []
            for g in gathered:
                v = _as_float_or_inf(g["metrics_seq"][i].get(k))
                w = float(g["local_batch"])
                if math.isfinite(v):
                    weighted_vals.append((v, w))
            if weighted_vals:
                row[k] = float(sum(v * w for v, w in weighted_vals) / max(sum(w for _, w in weighted_vals), 1e-12))
            else:
                row[k] = gathered[0]["metrics_seq"][i].get(k)
        per_repeat_metrics.append(row)

        weighted_iter = 0.0
        for g in gathered:
            weighted_iter += float(g["iterations_seq"][i]) * float(g["local_batch"])
        per_repeat_iters.append(weighted_iter / max(total_weight, 1e-12))

        statuses = [str(g["status_seq"][i]) for g in gathered]
        per_repeat_status.append("solved" if all(s == "solved" for s in statuses) else statuses[0])

    mean_metrics, std_metrics, sem_metrics = _summarize_metric_seq(per_repeat_metrics)
    return wall_times, total_batch, mean_metrics, std_metrics, sem_metrics, per_repeat_iters, per_repeat_status


def run_qp_solver(
    *,
    run_id: str,
    problem_id: str,
    solver_name: str,
    solver: SolveQPFn,
    problem: DenseBatchQP,
    warmups: int = 1,
    repeats: int = 5,
    dist_backend: str = "nccl",
    tol_p: float | None = None,
    tol_d: float | None = None,
    tol_g: float | None = None,
    tiers: list[float] | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> BenchmarkRecord | None:
    dist = init_distributed(backend=dist_backend)
    shard = shard_qp(problem, dist)

    if torch.cuda.is_available():
        if dist.is_distributed:
            target_device = torch.device(f"cuda:{dist.local_rank}")
            shard = shard.to(target_device)
        elif shard.q.device.type != "cuda":
            shard = shard.to("cuda")

    device = shard.q.device
    local_batch = shard.batch_size

    times, metrics_seq, iterations_seq, status_seq, result = _timed_runs(
        lambda: solver(shard), device=device, warmups=warmups, repeats=repeats
    )
    barrier_if_distributed(dist)

    (
        times_wall,
        total_batch,
        metrics,
        metrics_std,
        metrics_sem,
        iterations_vals,
        status_vals,
    ) = _aggregate_rank_results(dist, times, local_batch, metrics_seq, iterations_seq, status_seq)

    if dist.is_distributed and dist.rank != 0:
        return None

    timing = TimingSummary(
        median_s=statistics.median(times_wall),
        p90_s=_quantile(times_wall, 0.90),
        repeats=len(times_wall),
    )
    time_mean, time_std, time_sem = _mean_std_sem([float(t) for t in times_wall])
    iter_mean, iter_std, iter_sem = _mean_std_sem([float(v) for v in iterations_vals])

    md = {
        "batch_total": total_batch,
        "batch_per_rank": local_batch,
        "distributed": dist.is_distributed,
        "world_size": dist.world_size,
        "throughput_prob_per_s": float(total_batch / max(timing.median_s, 1e-12)),
        "solver_status": result.status,
        "solver_iterations": result.iterations,
        "solver_iterations_mean": iter_mean,
        "solver_iterations_std": iter_std,
        "solver_iterations_sem": iter_sem,
        "timing_mean_s": time_mean,
        "timing_std_s": time_std,
        "timing_sem_s": time_sem,
        "metrics_std": metrics_std,
        "metrics_sem": metrics_sem,
        "repeat_statuses": status_vals,
        "solved_repeats": int(sum(1 for s in status_vals if s == "solved")),
        "solver_timing": result.timing,
        "solver_metadata": result.metadata,
        "requested_tolerances": {"tol_p": tol_p, "tol_d": tol_d, "tol_g": tol_g},
    }
    tier_pass, best_tier = _compute_qp_tier_pass(metrics, tiers=tiers)
    md["tier_pass"] = tier_pass
    md["best_tier"] = best_tier
    if tol_p is not None and tol_d is not None and tol_g is not None:
        pass_flag = (
            float(metrics.get("primal_residual", float("inf"))) <= tol_p
            and float(metrics.get("dual_residual", float("inf"))) <= tol_d
            and float(metrics.get("duality_gap", float("inf"))) <= tol_g
        )
        md["stopping_pass"] = pass_flag
    if extra_metadata:
        md.update(extra_metadata)

    return BenchmarkRecord(
        run_id=run_id,
        category="qp_conic",
        problem_id=problem_id,
        solver=solver_name,
        status=result.status,
        timing=timing,
        metrics=metrics,
        metadata=md,
        system=system_info(),
    )


def run_lp_solver(
    *,
    run_id: str,
    problem_id: str,
    solver_name: str,
    solver: SolveLPFn,
    problem: DenseBatchLP,
    warmups: int = 1,
    repeats: int = 5,
    dist_backend: str = "nccl",
    tol_p: float | None = None,
    tol_d: float | None = None,
    tiers: list[float] | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> BenchmarkRecord | None:
    dist = init_distributed(backend=dist_backend)
    shard = shard_lp(problem, dist)

    if torch.cuda.is_available():
        if dist.is_distributed:
            target_device = torch.device(f"cuda:{dist.local_rank}")
            shard = shard.to(target_device)
        elif shard.b.device.type != "cuda":
            shard = shard.to("cuda")

    device = shard.b.device
    local_batch = shard.batch_size

    times, metrics_seq, iterations_seq, status_seq, result = _timed_runs(
        lambda: solver(shard), device=device, warmups=warmups, repeats=repeats
    )
    barrier_if_distributed(dist)

    (
        times_wall,
        total_batch,
        metrics,
        metrics_std,
        metrics_sem,
        iterations_vals,
        status_vals,
    ) = _aggregate_rank_results(dist, times, local_batch, metrics_seq, iterations_seq, status_seq)

    if dist.is_distributed and dist.rank != 0:
        return None

    timing = TimingSummary(
        median_s=statistics.median(times_wall),
        p90_s=_quantile(times_wall, 0.90),
        repeats=len(times_wall),
    )
    time_mean, time_std, time_sem = _mean_std_sem([float(t) for t in times_wall])
    iter_mean, iter_std, iter_sem = _mean_std_sem([float(v) for v in iterations_vals])

    md = {
        "batch_total": total_batch,
        "batch_per_rank": local_batch,
        "distributed": dist.is_distributed,
        "world_size": dist.world_size,
        "throughput_prob_per_s": float(total_batch / max(timing.median_s, 1e-12)),
        "solver_status": result.status,
        "solver_iterations": result.iterations,
        "solver_iterations_mean": iter_mean,
        "solver_iterations_std": iter_std,
        "solver_iterations_sem": iter_sem,
        "timing_mean_s": time_mean,
        "timing_std_s": time_std,
        "timing_sem_s": time_sem,
        "metrics_std": metrics_std,
        "metrics_sem": metrics_sem,
        "repeat_statuses": status_vals,
        "solved_repeats": int(sum(1 for s in status_vals if s == "solved")),
        "solver_timing": result.timing,
        "solver_metadata": result.metadata,
        "requested_tolerances": {"tol_p": tol_p, "tol_d": tol_d},
    }
    tier_pass, best_tier = _compute_lp_tier_pass(metrics, tiers=tiers)
    md["tier_pass"] = tier_pass
    md["best_tier"] = best_tier
    if tol_p is not None and tol_d is not None:
        pass_flag = (
            float(metrics.get("primal_residual", float("inf"))) <= tol_p
            and float(metrics.get("dual_residual", float("inf"))) <= tol_d
        )
        md["stopping_pass"] = pass_flag
    if extra_metadata:
        md.update(extra_metadata)

    return BenchmarkRecord(
        run_id=run_id,
        category="lp",
        problem_id=problem_id,
        solver=solver_name,
        status=result.status,
        timing=timing,
        metrics=metrics,
        metadata=md,
        system=system_info(),
    )


def _to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return str(value)


def write_jsonl(records: list[Any], path: str | Path) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for rec in records:
            if is_dataclass(rec):
                payload = asdict(rec)
            elif isinstance(rec, dict):
                payload = rec
            elif hasattr(rec, "__dict__"):
                payload = dict(rec.__dict__)
            else:
                raise TypeError(f"Unsupported record type: {type(rec)}")
            f.write(json.dumps(_to_jsonable(payload)) + "\n")


def cli() -> None:
    parser = argparse.ArgumentParser(description="Benchmark harness helper")
    parser.add_argument("--input", type=str, required=True, help="Input JSON payload")
    parser.add_argument("--output", type=str, required=True, help="Output JSONL path")
    args = parser.parse_args()

    # This CLI is a minimal adapter for external orchestration.
    with open(args.input, "r", encoding="utf-8") as f:
        payload = json.load(f)

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")


if __name__ == "__main__":
    cli()
