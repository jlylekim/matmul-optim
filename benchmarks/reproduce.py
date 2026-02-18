from __future__ import annotations

import argparse
import math
import os
import shutil
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from benchmarks.generators.batched_lp import BatchedLPConfig, generate_batched_lp
from benchmarks.generators.dense_parametric_qp import DenseParametricQPConfig, generate_dense_parametric_qp
from benchmarks.generators.portfolio_qp import PortfolioQPConfig, generate_portfolio_qp
from benchmarks.harness import run_lp_solver, run_qp_solver, write_jsonl
from gemm_kkt.baselines.base import BaselineResult, BaselineRunConfig
from gemm_kkt.baselines.cpu_refs import run_highs_lp, run_osqp_qp, run_scipy_trust_constr_qp
from gemm_kkt.solvers.gemm_ipm_qp import GEMMIPMConfig, GemmIPMQPSolver
from gemm_kkt.solvers.gemm_splitting_qp import GEMMSplittingQPConfig, GemmSplittingQPSolver
from gemm_kkt.solvers.lp_first_order_gpu import LPFirstOrderConfig, LPFirstOrderSolver
from gemm_kkt.utils.random import set_deterministic_seed


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


def _tier_name(tol: float) -> str:
    return f"{tol:.0e}".replace("e-0", "e-").replace("e+0", "e+")


def _as_float_or_inf(value: Any) -> float:
    try:
        out = float(value)
        if math.isfinite(out):
            return out
    except Exception:
        pass
    return float("inf")


def _qp_tier_pass(metrics: dict[str, Any], eval_tiers: list[float]) -> tuple[dict[str, bool], str | None]:
    p = _as_float_or_inf(metrics.get("primal_residual", float("inf")))
    d = _as_float_or_inf(metrics.get("dual_residual", float("inf")))
    g = _as_float_or_inf(metrics.get("duality_gap", float("inf")))
    use_gap = math.isfinite(g)

    pass_map: dict[str, bool] = {}
    for tol in eval_tiers:
        key = _tier_name(tol)
        pass_map[key] = bool(p <= tol and d <= tol and ((g <= tol) if use_gap else True))
    best = next((k for k in [_tier_name(t) for t in sorted(eval_tiers)] if pass_map[k]), None)
    return pass_map, best


def _lp_tier_pass(metrics: dict[str, Any], eval_tiers: list[float]) -> tuple[dict[str, bool], str | None]:
    p = _as_float_or_inf(metrics.get("primal_residual", float("inf")))
    d = _as_float_or_inf(metrics.get("dual_residual", float("inf")))
    pass_map: dict[str, bool] = {}
    for tol in eval_tiers:
        key = _tier_name(tol)
        pass_map[key] = bool(p <= tol and d <= tol)
    best = next((k for k in [_tier_name(t) for t in sorted(eval_tiers)] if pass_map[k]), None)
    return pass_map, best


def _baseline_to_record(
    *,
    run_id: str,
    category: str,
    problem_id: str,
    solver_name: str,
    baseline: BaselineResult,
    eval_tiers: list[float],
    tol_p: float,
    tol_d: float,
    tol_g: float | None = None,
) -> dict[str, Any] | None:
    if baseline.status == "unavailable":
        return None

    raw_status = str(baseline.status)
    status_lower = raw_status.lower()
    if "solved" in status_lower or "optimal" in status_lower:
        normalized_status = "solved"
    elif "max" in status_lower and "iter" in status_lower:
        normalized_status = "max_iters"
    elif "infeasible" in status_lower:
        normalized_status = "infeasible"
    elif "unbounded" in status_lower:
        normalized_status = "unbounded"
    else:
        normalized_status = "failed"

    metrics = dict(baseline.metrics)
    if category == "qp_conic":
        tier_pass, best_tier = _qp_tier_pass(metrics, eval_tiers=eval_tiers)
        stopping_pass = (
            _as_float_or_inf(metrics.get("primal_residual", float("inf"))) <= tol_p
            and _as_float_or_inf(metrics.get("dual_residual", float("inf"))) <= tol_d
            and (_as_float_or_inf(metrics.get("duality_gap", float("inf"))) <= tol_g if tol_g is not None else True)
        )
    else:
        tier_pass, best_tier = _lp_tier_pass(metrics, eval_tiers=eval_tiers)
        stopping_pass = (
            _as_float_or_inf(metrics.get("primal_residual", float("inf"))) <= tol_p
            and _as_float_or_inf(metrics.get("dual_residual", float("inf"))) <= tol_d
        )

    return {
        "run_id": run_id,
        "category": category,
        "problem_id": problem_id,
        "solver": solver_name,
        "status": normalized_status,
        "timing": {
            "median_s": baseline.solve_time_s,
            "p90_s": baseline.solve_time_s,
            "repeats": 1,
        },
        "metrics": metrics,
        "metadata": {
            "batch_total": 1,
            "batch_per_rank": 1,
            "distributed": False,
            "world_size": 1,
            "throughput_prob_per_s": 1.0 / max(baseline.solve_time_s, 1e-12),
            "tier_pass": tier_pass,
            "best_tier": best_tier,
            "stopping_pass": bool(stopping_pass),
            "solver_metadata": baseline.metadata,
            "raw_solver_status": raw_status,
            "baseline_version": baseline.version,
            "requested_tolerances": {"tol_p": tol_p, "tol_d": tol_d, "tol_g": tol_g},
        },
        "system": {"cpu_baseline": True},
    }


def _make_main_configs(device: str, seed: int, preset: str) -> list[DenseParametricQPConfig]:
    if preset == "large":
        return [
            DenseParametricQPConfig(batch_size=128, n=256, m=512, rhs_count=4, shared_matrices=True, seed=seed, device=device),
            DenseParametricQPConfig(batch_size=256, n=256, m=512, rhs_count=8, shared_matrices=True, seed=seed + 1, device=device),
            DenseParametricQPConfig(batch_size=64, n=384, m=768, rhs_count=4, shared_matrices=True, seed=seed + 2, device=device),
        ]
    return [
        DenseParametricQPConfig(batch_size=32, n=128, m=256, rhs_count=1, shared_matrices=True, seed=seed, device=device),
        DenseParametricQPConfig(batch_size=64, n=128, m=256, rhs_count=4, shared_matrices=True, seed=seed + 1, device=device),
        DenseParametricQPConfig(batch_size=16, n=256, m=512, rhs_count=2, shared_matrices=True, seed=seed + 2, device=device),
    ]


def _make_portfolio_configs(device: str, seed: int, preset: str) -> list[PortfolioQPConfig]:
    if preset == "large":
        return [
            PortfolioQPConfig(batch_size=128, n_assets=256, n_factors=12, seed=seed + 10, device=device),
            PortfolioQPConfig(batch_size=64, n_assets=512, n_factors=16, seed=seed + 11, device=device),
        ]
    return [
        PortfolioQPConfig(batch_size=64, n_assets=128, n_factors=8, seed=seed + 10, device=device),
        PortfolioQPConfig(batch_size=32, n_assets=256, n_factors=12, seed=seed + 11, device=device),
    ]


def _make_lp_configs(device: str, seed: int, preset: str) -> list[BatchedLPConfig]:
    if preset == "large":
        return [
            BatchedLPConfig(batch_size=256, n=512, m=256, seed=seed + 20, device=device),
            BatchedLPConfig(batch_size=256, n=768, m=384, seed=seed + 21, device=device),
        ]
    return [
        BatchedLPConfig(batch_size=64, n=256, m=128, seed=seed + 20, device=device),
        BatchedLPConfig(batch_size=128, n=384, m=192, seed=seed + 21, device=device),
    ]


def _make_scaling_config(device: str, seed: int, world_size: int, study: str, per_gpu_batch: int, total_batch_strong: int) -> DenseParametricQPConfig:
    if study == "strong":
        batch = max(1, total_batch_strong)
    else:
        batch = max(1, per_gpu_batch * max(1, world_size))

    return DenseParametricQPConfig(
        batch_size=batch,
        n=192,
        m=384,
        rhs_count=2,
        shared_matrices=True,
        seed=seed,
        device=device,
    )


def _planned_experiment_count(
    *,
    studies: list[str],
    device: str,
    seed: int,
    preset: str,
    world_size: int,
    per_gpu_batch: int,
    strong_total_batch: int,
    include_cpu_baselines: bool,
) -> int:
    total = 0
    for study in studies:
        if study == "main":
            dense = _make_main_configs(device, seed, preset)
            total += 3 * len(dense)  # ipm_ns + ipm_robust + splitting
            total += len(_make_portfolio_configs(device, seed, preset))  # robust on portfolio
            total += len(_make_lp_configs(device, seed, preset))  # gpu lp
            if include_cpu_baselines:
                total += 2 * len(dense)  # scipy trust-constr + osqp
                total += len(_make_lp_configs(device, seed, preset))  # highs
        else:
            _ = _make_scaling_config(
                device,
                seed + (1 if study == "weak" else 0),
                world_size,
                study,
                per_gpu_batch,
                strong_total_batch,
            )
            total += 3  # one dense config, three gpu solvers
    return total


def _maybe_autolaunch_torchrun(args: argparse.Namespace) -> None:
    """Auto-launch distributed run across all visible GPUs when not already under torchrun."""
    if args.no_auto_distributed:
        return

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        return

    if not torch.cuda.is_available():
        return

    visible_gpus = torch.cuda.device_count()
    if visible_gpus <= 1:
        return

    if shutil.which("torchrun") is None:
        print("[warn] Multiple GPUs detected but `torchrun` was not found. Running single process.")
        return

    nproc = visible_gpus if args.max_gpus <= 0 else min(max(1, args.max_gpus), visible_gpus)
    if nproc <= 1:
        return
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    script = str(Path(__file__).resolve())
    cmd = [
        "torchrun",
        "--standalone",
        f"--nproc_per_node={nproc}",
        script,
        "--output",
        str(Path(args.output).resolve()),
        "--seed",
        str(args.seed),
        "--study",
        str(args.study),
        "--warmups",
        str(args.warmups),
        "--repeats",
        str(args.repeats),
        "--per-gpu-batch",
        str(args.per_gpu_batch),
        "--strong-total-batch",
        str(args.strong_total_batch),
        "--preset",
        str(args.preset),
        "--no-auto-distributed",
        "--max-gpus",
        str(args.max_gpus),
    ]
    if args.eval_tiers:
        cmd.append("--eval-tiers")
        cmd.extend(str(t) for t in args.eval_tiers)
    print(f"[auto-distributed] launching on {nproc} GPUs: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    raise SystemExit(0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce GEMM-KKT experiments")
    parser.add_argument("--output", type=str, required=True, help="Results output directory")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--study", type=str, choices=["main", "strong", "weak", "all"], default="all")
    parser.add_argument(
        "--preset",
        type=str,
        choices=["quick", "large"],
        default="quick",
        help="Experiment preset size; `large` increases dimensions/batches for GPU-throughput studies",
    )
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--per-gpu-batch", type=int, default=64)
    parser.add_argument("--strong-total-batch", type=int, default=512)
    parser.add_argument(
        "--max-gpus",
        type=int,
        default=0,
        help="Maximum number of GPUs to use in auto-distributed mode (0 means all visible GPUs)",
    )
    parser.add_argument(
        "--no-auto-distributed",
        action="store_true",
        help="Disable automatic torchrun relaunch when multiple GPUs are visible",
    )
    parser.add_argument(
        "--eval-tiers",
        type=float,
        nargs="*",
        default=[1e-2, 1e-3, 1e-4],
        help="Tolerance tiers used for standardized pass/fail reporting",
    )
    args = parser.parse_args()

    _maybe_autolaunch_torchrun(args)
    set_deterministic_seed(args.seed)

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ipm_ns_cfg = GEMMIPMConfig(
        kkt_mode="ns_only",
        precision="bf16" if device == "cuda" else "fp32",
        tol_p=1e-4,
        tol_d=1e-4,
        tol_g=1e-4,
        max_iters=120 if args.preset == "large" else 80,
    )
    ipm_rb_cfg = GEMMIPMConfig(
        kkt_mode="robust",
        precision="bf16" if device == "cuda" else "fp32",
        tol_p=1e-6,
        tol_d=1e-6,
        tol_g=1e-6,
        max_iters=200 if args.preset == "large" else 120,
    )
    split_cfg = GEMMSplittingQPConfig(
        linear_solver="cg_ns",
        tol_p=1e-4,
        tol_d=1e-4,
        max_iters=50_000 if args.preset == "large" else 20_000,
    )

    ipm_ns = GemmIPMQPSolver(ipm_ns_cfg)
    ipm_robust = GemmIPMQPSolver(ipm_rb_cfg)
    split = GemmSplittingQPSolver(split_cfg)
    lp_cfg = LPFirstOrderConfig(max_iters=120_000 if args.preset == "large" else 80_000, tol_p=1e-4, tol_d=1e-4)
    lp_solver = LPFirstOrderSolver(lp_cfg)

    records: list[Any] = []

    studies: list[str]
    if args.study == "all":
        studies = ["main", "strong", "weak"]
    else:
        studies = [args.study]

    total_experiments = _planned_experiment_count(
        studies=studies,
        device=device,
        seed=args.seed,
        preset=args.preset,
        world_size=world_size,
        per_gpu_batch=args.per_gpu_batch,
        strong_total_batch=args.strong_total_batch,
        include_cpu_baselines=(rank == 0),
    )
    if rank == 0:
        print(
            f"[plan] studies={studies}, preset={args.preset}, world_size={world_size}, "
            f"planned_experiments={total_experiments}",
            flush=True,
        )
    progress = 0

    def _log_start(solver_name: str, problem_id: str) -> None:
        nonlocal progress
        if rank != 0:
            return
        progress += 1
        print(f"[{progress:03d}/{total_experiments:03d}] running {solver_name} on {problem_id}", flush=True)

    def _log_end(solver_name: str, status: str, median_s: float | None) -> None:
        if rank != 0:
            return
        if median_s is None:
            print(f"      -> {solver_name}: status={status}", flush=True)
            return
        print(f"      -> {solver_name}: status={status}, median_s={median_s:.4f}", flush=True)

    for study in studies:
        if study == "main":
            configs = _make_main_configs(device, args.seed, args.preset)
        else:
            configs = [
                _make_scaling_config(
                    device,
                    args.seed + (1 if study == "weak" else 0),
                    world_size,
                    study,
                    args.per_gpu_batch,
                    args.strong_total_batch,
                )
            ]

        for idx, cfg in enumerate(configs):
            problem = generate_dense_parametric_qp(cfg)
            problem_id = f"{study}_dense_parametric_{idx}"
            cfg_payload = _jsonable(asdict(cfg))

            _log_start("gemm_ipm_ns", problem_id)
            rec = run_qp_solver(
                run_id=f"{study}_run_{idx}_gemm_ipm_ns",
                problem_id=problem_id,
                solver_name="gemm_ipm_ns",
                solver=ipm_ns.solve,
                problem=problem,
                warmups=args.warmups,
                repeats=args.repeats,
                tol_p=ipm_ns_cfg.tol_p,
                tol_d=ipm_ns_cfg.tol_d,
                tol_g=ipm_ns_cfg.tol_g,
                tiers=args.eval_tiers,
                extra_metadata={
                    "experiment": "parametric_qp",
                    "study": study,
                    "preset": args.preset,
                    "config": cfg_payload,
                    "eval_tiers": args.eval_tiers,
                },
            )
            if rec is not None:
                records.append(rec)
                _log_end("gemm_ipm_ns", rec.status, rec.timing.median_s)

            _log_start("gemm_ipm_robust", problem_id)
            rec = run_qp_solver(
                run_id=f"{study}_run_{idx}_gemm_ipm_robust",
                problem_id=problem_id,
                solver_name="gemm_ipm_robust",
                solver=ipm_robust.solve,
                problem=problem,
                warmups=args.warmups,
                repeats=args.repeats,
                tol_p=ipm_rb_cfg.tol_p,
                tol_d=ipm_rb_cfg.tol_d,
                tol_g=ipm_rb_cfg.tol_g,
                tiers=args.eval_tiers,
                extra_metadata={
                    "experiment": "parametric_qp",
                    "study": study,
                    "preset": args.preset,
                    "config": cfg_payload,
                    "eval_tiers": args.eval_tiers,
                },
            )
            if rec is not None:
                records.append(rec)
                _log_end("gemm_ipm_robust", rec.status, rec.timing.median_s)

            _log_start("gemm_splitting_qp", problem_id)
            rec = run_qp_solver(
                run_id=f"{study}_run_{idx}_gemm_split",
                problem_id=problem_id,
                solver_name="gemm_splitting_qp",
                solver=split.solve,
                problem=problem,
                warmups=args.warmups,
                repeats=args.repeats,
                tol_p=split_cfg.tol_p,
                tol_d=split_cfg.tol_d,
                tol_g=1e9,
                tiers=args.eval_tiers,
                extra_metadata={
                    "experiment": "parametric_qp",
                    "study": study,
                    "preset": args.preset,
                    "config": cfg_payload,
                    "eval_tiers": args.eval_tiers,
                },
            )
            if rec is not None:
                records.append(rec)
                _log_end("gemm_splitting_qp", rec.status, rec.timing.median_s)

            if rank == 0 and study == "main":
                baseline_problem = problem.slice_batch(0, 1).to("cpu")
                bcfg = BaselineRunConfig(
                    tol_p=1e-4,
                    tol_d=1e-4,
                    max_iters=10000,
                    warm_start=False,
                    presolve=True,
                )
                _log_start("scipy_trust_constr_cpu", problem_id)
                scipy_res = run_scipy_trust_constr_qp(baseline_problem, bcfg)
                scipy_row = _baseline_to_record(
                    run_id=f"{study}_run_{idx}_scipy_trust_constr_cpu",
                    category="qp_conic",
                    problem_id=problem_id,
                    solver_name="scipy_trust_constr_cpu",
                    baseline=scipy_res,
                    eval_tiers=args.eval_tiers,
                    tol_p=bcfg.tol_p,
                    tol_d=bcfg.tol_d,
                    tol_g=bcfg.tol_g,
                )
                if scipy_row is not None:
                    records.append(scipy_row)
                    _log_end("scipy_trust_constr_cpu", scipy_row["status"], float(scipy_row["timing"]["median_s"]))
                else:
                    _log_end("scipy_trust_constr_cpu", "unavailable", None)

                _log_start("osqp_cpu", problem_id)
                osqp_res = run_osqp_qp(baseline_problem, bcfg)
                osqp_row = _baseline_to_record(
                    run_id=f"{study}_run_{idx}_osqp_cpu",
                    category="qp_conic",
                    problem_id=problem_id,
                    solver_name="osqp_cpu",
                    baseline=osqp_res,
                    eval_tiers=args.eval_tiers,
                    tol_p=bcfg.tol_p,
                    tol_d=bcfg.tol_d,
                    tol_g=bcfg.tol_g,
                )
                if osqp_row is not None:
                    records.append(osqp_row)
                    _log_end("osqp_cpu", osqp_row["status"], float(osqp_row["timing"]["median_s"]))
                else:
                    _log_end("osqp_cpu", "unavailable", None)

        if study == "main":
            for idx, cfg in enumerate(_make_portfolio_configs(device, args.seed, args.preset)):
                problem = generate_portfolio_qp(cfg)
                problem_id = f"{study}_portfolio_{idx}"
                cfg_payload = _jsonable(asdict(cfg))

                _log_start("gemm_ipm_robust", problem_id)
                rec = run_qp_solver(
                    run_id=f"{study}_portfolio_{idx}_gemm_ipm_robust",
                    problem_id=problem_id,
                    solver_name="gemm_ipm_robust",
                    solver=ipm_robust.solve,
                    problem=problem,
                    warmups=args.warmups,
                    repeats=args.repeats,
                    tol_p=ipm_rb_cfg.tol_p,
                    tol_d=ipm_rb_cfg.tol_d,
                    tol_g=ipm_rb_cfg.tol_g,
                    tiers=args.eval_tiers,
                    extra_metadata={
                        "experiment": "portfolio_qp",
                        "study": study,
                        "preset": args.preset,
                        "config": cfg_payload,
                        "eval_tiers": args.eval_tiers,
                    },
                )
                if rec is not None:
                    records.append(rec)
                    _log_end("gemm_ipm_robust", rec.status, rec.timing.median_s)

            for idx, cfg in enumerate(_make_lp_configs(device, args.seed, args.preset)):
                lp = generate_batched_lp(cfg)
                problem_id = f"{study}_lp_{idx}"
                cfg_payload = _jsonable(asdict(cfg))

                _log_start("lp_first_order_gpu", problem_id)
                rec = run_lp_solver(
                    run_id=f"{study}_lp_{idx}_pdhg",
                    problem_id=problem_id,
                    solver_name="lp_first_order_gpu",
                    solver=lp_solver.solve,
                    problem=lp,
                    warmups=args.warmups,
                    repeats=args.repeats,
                    tol_p=lp_cfg.tol_p,
                    tol_d=lp_cfg.tol_d,
                    tiers=args.eval_tiers,
                    extra_metadata={
                        "experiment": "batched_lp",
                        "study": study,
                        "preset": args.preset,
                        "config": cfg_payload,
                        "eval_tiers": args.eval_tiers,
                    },
                )
                if rec is not None:
                    records.append(rec)
                    _log_end("lp_first_order_gpu", rec.status, rec.timing.median_s)

                if rank == 0:
                    lp_cpu = lp.slice_batch(0, 1).to("cpu")
                    _log_start("highs_cpu", problem_id)
                    highs_res = run_highs_lp(lp_cpu, BaselineRunConfig(tol_p=1e-4, tol_d=1e-4, max_iters=100000))
                    highs_row = _baseline_to_record(
                        run_id=f"{study}_lp_{idx}_highs_cpu",
                        category="lp",
                        problem_id=problem_id,
                        solver_name="highs_cpu",
                        baseline=highs_res,
                        eval_tiers=args.eval_tiers,
                        tol_p=1e-4,
                        tol_d=1e-4,
                        tol_g=None,
                    )
                    if highs_row is not None:
                        records.append(highs_row)
                        _log_end("highs_cpu", highs_row["status"], float(highs_row["timing"]["median_s"]))
                    else:
                        _log_end("highs_cpu", "unavailable", None)

    if rank == 0:
        write_jsonl(records, out_dir / "results.jsonl")
        print(f"Completed {progress}/{total_experiments} planned experiments")
        print(f"Wrote {len(records)} records to {out_dir / 'results.jsonl'}")


if __name__ == "__main__":
    main()
