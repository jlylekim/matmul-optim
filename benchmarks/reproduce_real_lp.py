from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import torch

from benchmarks.datasets.load_lp_instances import CanonicalLP, load_canonical_lp, load_manifest_rows
from benchmarks.datasets.lp_adapters import canonical_to_densebatch_lp, canonical_to_densebatch_qp
from benchmarks.harness import write_jsonl
from gemm_kkt.baselines.base import BaselineResult, BaselineRunConfig
from gemm_kkt.baselines.cpu_refs import run_highs_lp, run_osqp_qp, run_scipy_trust_constr_qp
from gemm_kkt.solvers.common import SolverResult, sync_if_cuda
from gemm_kkt.solvers.gemm_ipm_qp import GEMMIPMConfig, GemmIPMQPSolver
from gemm_kkt.solvers.gemm_splitting_qp import GEMMSplittingQPConfig, GemmSplittingQPSolver
from gemm_kkt.solvers.lp_first_order_gpu import LPFirstOrderConfig, LPFirstOrderSolver
from gemm_kkt.utils.distributed import barrier_if_distributed, init_distributed
from gemm_kkt.utils.random import set_deterministic_seed
from gemm_kkt.utils.system import system_info

DEFAULT_SOLVERS = [
    "lp_first_order_gpu",
    "gemm_ipm_ns",
    "gemm_ipm_robust",
    "gemm_splitting_qp",
    "highs_cpu",
    "scipy_trust_constr_cpu",
    "osqp_cpu",
]


def _as_float_or_inf(value: Any) -> float:
    try:
        x = float(value)
        if math.isfinite(x):
            return x
    except Exception:
        pass
    return float("inf")


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return str(value)


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(_jsonable(row)) + "\n")


def _status_from_baseline(raw_status: str) -> str:
    s = raw_status.lower()
    if "solved" in s or "optimal" in s:
        return "solved"
    if "max" in s and "iter" in s:
        return "max_iters"
    if "infeasible" in s:
        return "infeasible"
    if "unbounded" in s:
        return "unbounded"
    if "unavailable" in s:
        return "unavailable"
    return "failed"


def _failure_row(
    *,
    run_id: str,
    problem_id: str,
    solver: str,
    category: str,
    suite: str,
    instance_name: str,
    rank: int,
    world_size: int,
    base_meta: dict[str, Any],
    stage: str,
    status: str,
    error: Exception | None = None,
    elapsed_s: float | None = None,
) -> dict[str, Any]:
    md = dict(base_meta)
    md.update(
        {
            "suite": suite,
            "instance_name": instance_name,
            "rank": rank,
            "world_size": world_size,
            "stage": stage,
        }
    )
    if error is not None:
        md["error_type"] = type(error).__name__
        md["error_message"] = str(error)
    return {
        "run_id": run_id,
        "category": category,
        "problem_id": problem_id,
        "solver": solver,
        "status": status,
        "timing": {"median_s": float(elapsed_s or 0.0), "p90_s": float(elapsed_s or 0.0), "repeats": 1},
        "metrics": {"primal_residual": float("nan"), "dual_residual": float("nan"), "duality_gap": float("nan")},
        "metadata": md,
        "system": system_info(),
    }


def _run_solver_repeats(
    *,
    solve_once: Any,
    device: torch.device,
    warmups: int,
    repeats: int,
    solve_timeout_s: float,
) -> tuple[str, dict[str, Any], dict[str, float], dict[str, Any], list[str], list[int]]:
    for _ in range(max(0, warmups)):
        solve_once()

    times: list[float] = []
    metrics_seq: list[dict[str, Any]] = []
    statuses: list[str] = []
    iters: list[int] = []
    timed_out = False
    last_result: SolverResult | None = None
    for _ in range(max(1, repeats)):
        sync_if_cuda(device)
        t0 = time.perf_counter()
        last_result = solve_once()
        sync_if_cuda(device)
        elapsed = time.perf_counter() - t0
        times.append(float(elapsed))
        metrics_seq.append(dict(last_result.metrics))
        statuses.append(str(last_result.status))
        iters.append(int(last_result.iterations))
        if elapsed > solve_timeout_s > 0:
            timed_out = True
            break

    assert last_result is not None
    metric_keys = set().union(*(m.keys() for m in metrics_seq))
    metrics_mean: dict[str, Any] = {}
    for k in metric_keys:
        vals = [_as_float_or_inf(m.get(k)) for m in metrics_seq]
        finite = [v for v in vals if math.isfinite(v)]
        metrics_mean[k] = float(sum(finite) / len(finite)) if finite else metrics_seq[-1].get(k)

    if timed_out:
        status = "timeout"
    else:
        status = "solved" if all(s == "solved" for s in statuses) else statuses[-1]
    timing = {
        "median_s": float(sorted(times)[len(times) // 2]),
        "p90_s": float(sorted(times)[min(len(times) - 1, int(math.ceil(0.9 * len(times))) - 1)]),
        "repeats": len(times),
    }
    meta = {
        "repeat_statuses": statuses,
        "solver_iterations_mean": float(sum(float(i) for i in iters) / len(iters)),
        "solver_iterations_max": int(max(iters)),
        "timing_mean_s": float(sum(times) / len(times)),
        "timing_max_s": float(max(times)),
    }
    return status, metrics_mean, timing, meta, statuses, iters


def _make_baseline_row(
    *,
    run_id: str,
    problem_id: str,
    solver: str,
    suite: str,
    instance_name: str,
    rank: int,
    world_size: int,
    base_meta: dict[str, Any],
    baseline: BaselineResult,
) -> dict[str, Any]:
    raw_status = str(baseline.status)
    status = _status_from_baseline(raw_status)
    md = dict(base_meta)
    md.update(
        {
            "suite": suite,
            "instance_name": instance_name,
            "rank": rank,
            "world_size": world_size,
            "raw_solver_status": raw_status,
            "baseline_version": baseline.version,
            "solver_metadata": baseline.metadata,
        }
    )
    return {
        "run_id": run_id,
        "category": "lp",
        "problem_id": problem_id,
        "solver": solver,
        "status": status,
        "timing": {"median_s": float(baseline.solve_time_s), "p90_s": float(baseline.solve_time_s), "repeats": 1},
        "metrics": dict(baseline.metrics),
        "metadata": md,
        "system": system_info(),
    }


def _slice_for_rank(rows: list[dict[str, str]], rank: int, world_size: int) -> list[dict[str, str]]:
    if world_size <= 1:
        return rows
    return [row for i, row in enumerate(rows) if (i % world_size) == rank]


def _maybe_autolaunch_torchrun(args: argparse.Namespace) -> None:
    if args.no_auto_distributed:
        return
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        return
    if not torch.cuda.is_available():
        return
    visible_gpus = torch.cuda.device_count()
    if visible_gpus <= 1:
        return
    if shutil.which("torchrun") is None:
        print("[warn] Multiple GPUs detected but torchrun is unavailable; running single process.")
        return

    nproc = visible_gpus if args.max_gpus <= 0 else min(max(1, args.max_gpus), visible_gpus)
    if nproc <= 1:
        return
    cmd = [
        "torchrun",
        "--standalone",
        f"--nproc_per_node={nproc}",
        str(Path(__file__).resolve()),
        "--manifest",
        str(Path(args.manifest).resolve()),
        "--output",
        str(Path(args.output).resolve()),
        "--cache-root",
        str(Path(args.cache_root).resolve()),
        "--u-cap",
        str(args.u_cap),
        "--warmups",
        str(args.warmups),
        "--repeats",
        str(args.repeats),
        "--target-tol",
        str(args.target_tol),
        "--parse-timeout-s",
        str(args.parse_timeout_s),
        "--solve-timeout-s",
        str(args.solve_timeout_s),
        "--instance-timeout-s",
        str(args.instance_timeout_s),
        "--seed",
        str(args.seed),
        "--batch-size",
        str(args.batch_size),
        "--precision",
        str(args.precision),
        "--ipm-max-iters",
        str(args.ipm_max_iters),
        "--robust-max-iters",
        str(args.robust_max_iters),
        "--split-max-iters",
        str(args.split_max_iters),
        "--lp-max-iters",
        str(args.lp_max_iters),
        "--max-instances",
        str(args.max_instances),
        "--max-gpus",
        str(args.max_gpus),
        "--no-auto-distributed",
    ]
    if args.suites:
        cmd.extend(["--suites", *args.suites])
    if args.solvers:
        cmd.extend(["--solvers", *args.solvers])
    if args.force_reparse:
        cmd.append("--force-reparse")
    if args.skip_existing:
        cmd.append("--skip-existing")
    print(f"[auto-distributed] launching on {nproc} GPUs: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    raise SystemExit(0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run real LP suite benchmarks with robust failure capture")
    parser.add_argument("--manifest", type=str, default="benchmarks/datasets/real_lp_manifest.csv")
    parser.add_argument("--cache-root", type=str, default="benchmarks/datasets/cache")
    parser.add_argument("--suites", nargs="*", default=["netlib", "stochlp", "misc"])
    parser.add_argument("--solvers", nargs="*", default=DEFAULT_SOLVERS)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--u-cap", type=float, default=1e8)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--target-tol", type=float, default=1e-2)
    parser.add_argument("--parse-timeout-s", type=float, default=300.0)
    parser.add_argument("--solve-timeout-s", type=float, default=1200.0)
    parser.add_argument("--instance-timeout-s", type=float, default=3600.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--precision", type=str, choices=["fp32", "fp16", "bf16"], default="bf16")
    parser.add_argument("--ipm-max-iters", type=int, default=80)
    parser.add_argument("--robust-max-iters", type=int, default=120)
    parser.add_argument("--split-max-iters", type=int, default=20000)
    parser.add_argument("--lp-max-iters", type=int, default=8000)
    parser.add_argument("--max-instances", type=int, default=0, help="0 means all")
    parser.add_argument("--force-reparse", action="store_true")
    parser.add_argument("--skip-existing", action="store_true", help="Skip (instance,solver) already in rank output file")
    parser.add_argument("--max-gpus", type=int, default=0, help="Auto-distributed GPU cap (0 means all visible)")
    parser.add_argument("--no-auto-distributed", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _maybe_autolaunch_torchrun(args)
    set_deterministic_seed(int(args.seed))

    dist = init_distributed()
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    rank_file = out_dir / f"results_rank{dist.rank}.jsonl"
    if rank_file.exists() and not args.skip_existing:
        rank_file.unlink()
    rank_file.touch(exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{dist.local_rank}" if dist.is_distributed else "cuda")
    else:
        device = torch.device("cpu")
    print(
        f"[rank-bind] rank={dist.rank} local_rank={dist.local_rank} world_size={dist.world_size} device={device}",
        flush=True,
    )

    rows = load_manifest_rows(args.manifest, suites=args.suites)
    rows = sorted(rows, key=lambda r: (str(r.get("suite", "")), str(r.get("instance_name", ""))))
    if int(args.max_instances) > 0:
        rows = rows[: int(args.max_instances)]
    shard = _slice_for_rank(rows, dist.rank, dist.world_size)
    print(f"[plan] total_instances={len(rows)} local_instances={len(shard)} solvers={args.solvers}", flush=True)

    completed_keys: set[str] = set()
    if args.skip_existing and rank_file.exists():
        with rank_file.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                completed_keys.add(f"{rec.get('problem_id')}::{rec.get('solver')}")

    ipm_ns = GemmIPMQPSolver(
        GEMMIPMConfig(
            kkt_mode="ns_only",
            max_iters=int(args.ipm_max_iters),
            tol_p=float(args.target_tol),
            tol_d=float(args.target_tol),
            tol_g=float(args.target_tol),
            precision=str(args.precision),
        )
    )
    ipm_rb = GemmIPMQPSolver(
        GEMMIPMConfig(
            kkt_mode="robust",
            max_iters=int(args.robust_max_iters),
            tol_p=float(args.target_tol),
            tol_d=float(args.target_tol),
            tol_g=float(args.target_tol),
            precision=str(args.precision),
        )
    )
    split = GemmSplittingQPSolver(
        GEMMSplittingQPConfig(
            max_iters=int(args.split_max_iters),
            tol_p=float(args.target_tol),
            tol_d=float(args.target_tol),
            linear_solver="cg_ns",
        )
    )
    lp_solver = LPFirstOrderSolver(
        LPFirstOrderConfig(
            max_iters=int(args.lp_max_iters),
            tol_p=float(args.target_tol),
            tol_d=float(args.target_tol),
        )
    )
    baseline_cfg = BaselineRunConfig(
        tol_p=float(args.target_tol),
        tol_d=float(args.target_tol),
        tol_g=float(args.target_tol),
        max_iters=max(int(args.lp_max_iters), int(args.robust_max_iters), 10000),
        warm_start=False,
        presolve=True,
    )

    run_counter = 0
    for entry in shard:
        suite = str(entry["suite"])
        instance_name = str(entry["instance_name"])
        expanded_path = str(entry["expanded_path"])
        problem_id = f"{suite}:{instance_name}"
        instance_t0 = time.perf_counter()
        base_meta = {
            "suite": suite,
            "instance_name": instance_name,
            "rank": dist.rank,
            "world_size": dist.world_size,
        }

        parse_elapsed = 0.0
        canonical: CanonicalLP | None = None
        parse_error: Exception | None = None
        try:
            t0 = time.perf_counter()
            canonical = load_canonical_lp(
                expanded_path=expanded_path,
                suite=suite,
                instance_name=instance_name,
                cache_root=args.cache_root,
                force_reparse=bool(args.force_reparse),
            )
            parse_elapsed = time.perf_counter() - t0
            if parse_elapsed > float(args.parse_timeout_s) > 0:
                parse_error = RuntimeError(
                    f"parse_timeout: elapsed={parse_elapsed:.3f}s exceeds {float(args.parse_timeout_s):.3f}s"
                )
        except Exception as exc:
            parse_error = exc

        lp_problem = None
        qp_problem = None
        lp_diag: dict[str, Any] = {}
        qp_diag: dict[str, Any] = {}
        lp_convert_error: Exception | None = None
        qp_convert_error: Exception | None = None
        if parse_error is None and canonical is not None:
            base_meta.update(
                {
                    "n": canonical.n,
                    "m": canonical.m,
                    "nnz": canonical.nnz,
                    "density": canonical.density,
                    "parse_elapsed_s": parse_elapsed,
                }
            )
            try:
                lp_problem, lp_diag = canonical_to_densebatch_lp(
                    canonical,
                    device=device,
                    dtype=torch.float32,
                    batch_size=max(1, int(args.batch_size)),
                )
            except Exception as exc:
                lp_convert_error = exc
            try:
                qp_problem, qp_diag = canonical_to_densebatch_qp(
                    canonical,
                    u_cap=float(args.u_cap),
                    device=device,
                    dtype=torch.float32,
                    batch_size=max(1, int(args.batch_size)),
                )
            except Exception as exc:
                qp_convert_error = exc

        for solver_name in args.solvers:
            run_counter += 1
            key = f"{problem_id}::{solver_name}"
            if key in completed_keys:
                continue

            run_id = f"real_lp_{dist.rank}_{run_counter}_{suite}_{instance_name}_{solver_name}"
            elapsed_instance = time.perf_counter() - instance_t0
            if float(args.instance_timeout_s) > 0 and elapsed_instance > float(args.instance_timeout_s):
                row = _failure_row(
                    run_id=run_id,
                    problem_id=problem_id,
                    solver=solver_name,
                    category="lp",
                    suite=suite,
                    instance_name=instance_name,
                    rank=dist.rank,
                    world_size=dist.world_size,
                    base_meta=base_meta,
                    stage="instance_timeout",
                    status="timeout",
                    error=RuntimeError(
                        f"instance_timeout: elapsed={elapsed_instance:.3f}s exceeds {float(args.instance_timeout_s):.3f}s"
                    ),
                    elapsed_s=elapsed_instance,
                )
                _append_jsonl(rank_file, row)
                continue

            if parse_error is not None:
                row = _failure_row(
                    run_id=run_id,
                    problem_id=problem_id,
                    solver=solver_name,
                    category="lp",
                    suite=suite,
                    instance_name=instance_name,
                    rank=dist.rank,
                    world_size=dist.world_size,
                    base_meta=base_meta,
                    stage="parse",
                    status="failed" if "timeout" not in str(parse_error).lower() else "timeout",
                    error=parse_error,
                    elapsed_s=parse_elapsed,
                )
                _append_jsonl(rank_file, row)
                continue

            solver_meta = dict(base_meta)
            solver_meta["conversion_stats"] = lp_diag if solver_name in ("lp_first_order_gpu", "highs_cpu") else qp_diag

            try:
                if solver_name == "lp_first_order_gpu":
                    if lp_problem is None:
                        raise lp_convert_error or RuntimeError("LP conversion did not produce problem")
                    status, metrics, timing, smeta, _, _ = _run_solver_repeats(
                        solve_once=lambda: lp_solver.solve(lp_problem),
                        device=device,
                        warmups=int(args.warmups),
                        repeats=int(args.repeats),
                        solve_timeout_s=float(args.solve_timeout_s),
                    )
                    solver_meta.update(smeta)
                    row = {
                        "run_id": run_id,
                        "category": "lp",
                        "problem_id": problem_id,
                        "solver": solver_name,
                        "status": status,
                        "timing": timing,
                        "metrics": metrics,
                        "metadata": solver_meta,
                        "system": system_info(),
                    }
                elif solver_name in ("gemm_ipm_ns", "gemm_ipm_robust", "gemm_splitting_qp"):
                    if qp_problem is None:
                        raise qp_convert_error or RuntimeError("QP conversion did not produce problem")
                    if solver_name == "gemm_ipm_ns":
                        solve_once = lambda: ipm_ns.solve(qp_problem)
                    elif solver_name == "gemm_ipm_robust":
                        solve_once = lambda: ipm_rb.solve(qp_problem)
                    else:
                        solve_once = lambda: split.solve(qp_problem)
                    status, metrics, timing, smeta, _, _ = _run_solver_repeats(
                        solve_once=solve_once,
                        device=device,
                        warmups=int(args.warmups),
                        repeats=int(args.repeats),
                        solve_timeout_s=float(args.solve_timeout_s),
                    )
                    solver_meta.update(smeta)
                    row = {
                        "run_id": run_id,
                        "category": "lp",
                        "problem_id": problem_id,
                        "solver": solver_name,
                        "status": status,
                        "timing": timing,
                        "metrics": metrics,
                        "metadata": solver_meta,
                        "system": system_info(),
                    }
                elif solver_name == "highs_cpu":
                    if lp_problem is None:
                        raise lp_convert_error or RuntimeError("LP conversion did not produce problem")
                    baseline = run_highs_lp(lp_problem.slice_batch(0, 1).to("cpu"), baseline_cfg)
                    row = _make_baseline_row(
                        run_id=run_id,
                        problem_id=problem_id,
                        solver=solver_name,
                        suite=suite,
                        instance_name=instance_name,
                        rank=dist.rank,
                        world_size=dist.world_size,
                        base_meta=solver_meta,
                        baseline=baseline,
                    )
                elif solver_name in ("scipy_trust_constr_cpu", "osqp_cpu"):
                    if qp_problem is None:
                        raise qp_convert_error or RuntimeError("QP conversion did not produce problem")
                    qp_cpu = qp_problem.slice_batch(0, 1).to("cpu")
                    if solver_name == "scipy_trust_constr_cpu":
                        baseline = run_scipy_trust_constr_qp(qp_cpu, baseline_cfg)
                    else:
                        baseline = run_osqp_qp(qp_cpu, baseline_cfg)
                    row = _make_baseline_row(
                        run_id=run_id,
                        problem_id=problem_id,
                        solver=solver_name,
                        suite=suite,
                        instance_name=instance_name,
                        rank=dist.rank,
                        world_size=dist.world_size,
                        base_meta=solver_meta,
                        baseline=baseline,
                    )
                else:
                    raise RuntimeError(f"Unknown solver `{solver_name}`")
            except Exception as exc:
                row = _failure_row(
                    run_id=run_id,
                    problem_id=problem_id,
                    solver=solver_name,
                    category="lp",
                    suite=suite,
                    instance_name=instance_name,
                    rank=dist.rank,
                    world_size=dist.world_size,
                    base_meta=solver_meta,
                    stage="solve",
                    status="failed",
                    error=exc,
                )

            _append_jsonl(rank_file, row)
            print(
                f"[rank={dist.rank}] {problem_id} solver={solver_name} status={row['status']} t={row['timing']['median_s']:.4f}s",
                flush=True,
            )

    barrier_if_distributed(dist)

    if dist.rank == 0:
        rows_all: list[dict[str, Any]] = []
        for rr in range(dist.world_size):
            p = out_dir / f"results_rank{rr}.jsonl"
            if not p.exists():
                continue
            with p.open("r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    rows_all.append(json.loads(line))
        write_jsonl(rows_all, out_dir / "results.jsonl")
        print(f"[done] wrote {len(rows_all)} rows to {out_dir / 'results.jsonl'}", flush=True)


if __name__ == "__main__":
    main()
