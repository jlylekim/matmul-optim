from __future__ import annotations

import argparse
import json
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
) -> tuple[list[float], SolverResult]:
    last_result: SolverResult | None = None

    for _ in range(warmups):
        last_result = solve_once()

    times: list[float] = []
    for _ in range(repeats):
        sync_if_cuda(device)
        t0 = time.perf_counter()
        last_result = solve_once()
        sync_if_cuda(device)
        times.append(time.perf_counter() - t0)

    assert last_result is not None
    return times, last_result


def _aggregate_rank_results(info: DistInfo, times: list[float], local_batch: int, metrics: dict[str, Any]) -> tuple[list[float], int, dict[str, Any]]:
    payload = {
        "times": times,
        "local_batch": local_batch,
        "metrics": metrics,
    }
    gathered = gather_objects(info, payload)

    if not info.is_distributed:
        return times, local_batch, metrics

    repeats = len(times)
    wall_times: list[float] = []
    for i in range(repeats):
        wall_times.append(max(float(g["times"][i]) for g in gathered))

    total_batch = int(sum(int(g["local_batch"]) for g in gathered))

    metric_keys = set().union(*(g["metrics"].keys() for g in gathered))
    aggregated: dict[str, Any] = {}
    for key in metric_keys:
        vals = [g["metrics"].get(key) for g in gathered]
        numeric = [float(v) for v in vals if isinstance(v, (int, float))]
        aggregated[key] = float(sum(numeric) / len(numeric)) if numeric else vals[0]

    return wall_times, total_batch, aggregated


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
    extra_metadata: dict[str, Any] | None = None,
) -> BenchmarkRecord | None:
    dist = init_distributed(backend=dist_backend)
    shard = shard_qp(problem, dist)

    device = shard.q.device
    local_batch = shard.batch_size

    times, result = _timed_runs(lambda: solver(shard), device=device, warmups=warmups, repeats=repeats)
    barrier_if_distributed(dist)

    times_wall, total_batch, metrics = _aggregate_rank_results(dist, times, local_batch, result.metrics)

    if dist.is_distributed and dist.rank != 0:
        return None

    timing = TimingSummary(
        median_s=statistics.median(times_wall),
        p90_s=_quantile(times_wall, 0.90),
        repeats=len(times_wall),
    )

    md = {
        "batch_total": total_batch,
        "batch_per_rank": local_batch,
        "distributed": dist.is_distributed,
        "world_size": dist.world_size,
        "throughput_prob_per_s": float(total_batch / max(timing.median_s, 1e-12)),
        "solver_status": result.status,
        "solver_iterations": result.iterations,
        "solver_timing": result.timing,
        "solver_metadata": result.metadata,
        "requested_tolerances": {"tol_p": tol_p, "tol_d": tol_d, "tol_g": tol_g},
    }
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
    extra_metadata: dict[str, Any] | None = None,
) -> BenchmarkRecord | None:
    dist = init_distributed(backend=dist_backend)
    shard = shard_lp(problem, dist)

    device = shard.b.device
    local_batch = shard.batch_size

    times, result = _timed_runs(lambda: solver(shard), device=device, warmups=warmups, repeats=repeats)
    barrier_if_distributed(dist)

    times_wall, total_batch, metrics = _aggregate_rank_results(dist, times, local_batch, result.metrics)

    if dist.is_distributed and dist.rank != 0:
        return None

    timing = TimingSummary(
        median_s=statistics.median(times_wall),
        p90_s=_quantile(times_wall, 0.90),
        repeats=len(times_wall),
    )

    md = {
        "batch_total": total_batch,
        "batch_per_rank": local_batch,
        "distributed": dist.is_distributed,
        "world_size": dist.world_size,
        "throughput_prob_per_s": float(total_batch / max(timing.median_s, 1e-12)),
        "solver_status": result.status,
        "solver_iterations": result.iterations,
        "solver_timing": result.timing,
        "solver_metadata": result.metadata,
        "requested_tolerances": {"tol_p": tol_p, "tol_d": tol_d},
    }
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
