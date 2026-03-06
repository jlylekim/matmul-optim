from __future__ import annotations

import argparse
import math
import os
import shutil
import subprocess
import time
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from benchmarks.generators.batched_lp import BatchedLPConfig, generate_batched_lp
from benchmarks.generators.dense_parametric_qp import DenseParametricQPConfig, generate_dense_parametric_qp
from benchmarks.generators.portfolio_qp import PortfolioQPConfig, generate_portfolio_qp
from benchmarks.harness import run_lp_solver, run_qp_solver, write_jsonl
from gemm_kkt.baselines.base import BaselineResult, BaselineRunConfig
from gemm_kkt.baselines.cpu_refs import run_highs_lp, run_osqp_qp, run_scipy_trust_constr_qp
from gemm_kkt.solvers.gemm_ipm_qp import GEMMIPMConfig, GemmIPMQPSolver
from gemm_kkt.solvers.gemm_splitting_qp import GEMMSplittingQPConfig, GemmSplittingQPSolver
from gemm_kkt.solvers.lp_first_order_gpu import LPFirstOrderConfig, LPFirstOrderSolver
from gemm_kkt.solvers.types import DenseBatchQP
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
    if preset == "a100_heavy":
        return [
            DenseParametricQPConfig(
                batch_size=192,
                n=512,
                m=1024,
                condition_number=1e4,
                tightness=0.35,
                rhs_count=4,
                shared_matrices=True,
                seed=seed,
                device=device,
            ),
            DenseParametricQPConfig(
                batch_size=128,
                n=768,
                m=1536,
                condition_number=5e4,
                tightness=0.25,
                rhs_count=3,
                shared_matrices=True,
                seed=seed + 1,
                device=device,
            ),
            DenseParametricQPConfig(
                batch_size=96,
                n=1024,
                m=2048,
                condition_number=1e5,
                tightness=0.2,
                rhs_count=2,
                shared_matrices=True,
                seed=seed + 2,
                device=device,
            ),
            DenseParametricQPConfig(
                batch_size=64,
                n=1280,
                m=2560,
                condition_number=3e5,
                tightness=0.15,
                rhs_count=2,
                shared_matrices=True,
                seed=seed + 3,
                device=device,
            ),
        ]
    if preset == "ns_favor":
        return [
            DenseParametricQPConfig(
                batch_size=256,
                n=128,
                m=256,
                condition_number=5e2,
                tightness=0.7,
                rhs_count=16,
                shared_matrices=True,
                seed=seed,
                device=device,
            ),
            DenseParametricQPConfig(
                batch_size=128,
                n=256,
                m=512,
                condition_number=1e3,
                tightness=0.6,
                rhs_count=12,
                shared_matrices=True,
                seed=seed + 1,
                device=device,
            ),
            DenseParametricQPConfig(
                batch_size=96,
                n=320,
                m=640,
                condition_number=2e3,
                tightness=0.5,
                rhs_count=10,
                shared_matrices=True,
                seed=seed + 2,
                device=device,
            ),
            DenseParametricQPConfig(
                batch_size=64,
                n=384,
                m=768,
                condition_number=3e3,
                tightness=0.45,
                rhs_count=8,
                shared_matrices=True,
                seed=seed + 3,
                device=device,
            ),
        ]
    if preset == "mixed":
        return [
            DenseParametricQPConfig(
                batch_size=192,
                n=128,
                m=256,
                condition_number=1e3,
                tightness=0.6,
                rhs_count=8,
                shared_matrices=True,
                seed=seed,
                device=device,
            ),
            DenseParametricQPConfig(
                batch_size=128,
                n=256,
                m=512,
                condition_number=1e4,
                tightness=0.35,
                rhs_count=6,
                shared_matrices=True,
                seed=seed + 1,
                device=device,
            ),
            DenseParametricQPConfig(
                batch_size=80,
                n=384,
                m=768,
                condition_number=5e4,
                tightness=0.25,
                rhs_count=4,
                shared_matrices=True,
                seed=seed + 2,
                device=device,
            ),
            DenseParametricQPConfig(
                batch_size=64,
                n=512,
                m=1024,
                condition_number=1e5,
                tightness=0.2,
                rhs_count=3,
                shared_matrices=True,
                seed=seed + 3,
                device=device,
            ),
        ]
    if preset == "stress":
        return [
            DenseParametricQPConfig(
                batch_size=128,
                n=256,
                m=512,
                condition_number=1e5,
                tightness=0.2,
                rhs_count=4,
                shared_matrices=True,
                seed=seed,
                device=device,
            ),
            DenseParametricQPConfig(
                batch_size=96,
                n=384,
                m=768,
                condition_number=3e5,
                tightness=0.15,
                rhs_count=3,
                shared_matrices=True,
                seed=seed + 1,
                device=device,
            ),
            DenseParametricQPConfig(
                batch_size=64,
                n=512,
                m=1024,
                condition_number=7e5,
                tightness=0.12,
                rhs_count=2,
                shared_matrices=True,
                seed=seed + 2,
                device=device,
            ),
            DenseParametricQPConfig(
                batch_size=48,
                n=640,
                m=1280,
                condition_number=1e6,
                tightness=0.1,
                rhs_count=2,
                shared_matrices=True,
                seed=seed + 3,
                device=device,
            ),
        ]
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
    if preset == "a100_heavy":
        return [
            PortfolioQPConfig(batch_size=96, n_assets=512, n_factors=16, seed=seed + 10, device=device),
            PortfolioQPConfig(batch_size=64, n_assets=768, n_factors=24, seed=seed + 11, device=device),
        ]
    if preset in ("ns_favor", "mixed", "stress"):
        return [
            PortfolioQPConfig(batch_size=128, n_assets=256, n_factors=12, seed=seed + 10, device=device),
            PortfolioQPConfig(batch_size=96, n_assets=384, n_factors=16, seed=seed + 11, device=device),
        ]
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
    if preset == "a100_heavy":
        return [
            BatchedLPConfig(batch_size=256, n=1024, m=512, seed=seed + 20, device=device),
            BatchedLPConfig(batch_size=192, n=1536, m=768, seed=seed + 21, device=device),
        ]
    if preset in ("ns_favor", "mixed", "stress"):
        return [
            BatchedLPConfig(batch_size=256, n=512, m=256, seed=seed + 20, device=device),
            BatchedLPConfig(batch_size=192, n=768, m=384, seed=seed + 21, device=device),
        ]
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
    include_splitting: bool,
    include_portfolio: bool,
    include_lp: bool,
) -> int:
    total = 0
    for study in studies:
        if study == "main":
            dense = _make_main_configs(device, seed, preset)
            dense_solver_count = 2 + (1 if include_splitting else 0)
            total += dense_solver_count * len(dense)  # ipm_ns + ipm_robust (+optional splitting)
            if include_portfolio:
                total += len(_make_portfolio_configs(device, seed, preset))  # robust on portfolio
            if include_lp:
                total += len(_make_lp_configs(device, seed, preset))  # gpu lp
            if include_cpu_baselines:
                total += 2 * len(dense)  # scipy trust-constr + osqp
                if include_lp:
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
            total += 2 + (1 if include_splitting else 0)  # one dense config, two IPM solvers (+optional splitting)
    return total


def _clone_cfg(cfg: GEMMIPMConfig, **kwargs: Any) -> GEMMIPMConfig:
    out = deepcopy(cfg)
    for k, v in kwargs.items():
        if k.startswith("ns_"):
            setattr(out.ns_config, k[3:], v)
        else:
            setattr(out, k, v)
    return out


def _ipm_objective(metrics: dict[str, Any], status: str, elapsed_s: float) -> float:
    p = _as_float_or_inf(metrics.get("primal_residual", float("inf")))
    d = _as_float_or_inf(metrics.get("dual_residual", float("inf")))
    g = _as_float_or_inf(metrics.get("duality_gap", float("inf")))

    score = p + d + (g if math.isfinite(g) else max(p, d))
    if status != "solved":
        score += 10.0
    score += 1e-3 * float(elapsed_s)
    return score


def _candidate_ipm_configs(base: GEMMIPMConfig, mode: str) -> list[GEMMIPMConfig]:
    # Small, explicit grid focused on stabilization knobs that were most sensitive in practice.
    if mode == "ns_only":
        return [
            _clone_cfg(base, kkt_mode="ns_only"),
            _clone_cfg(
                base,
                kkt_mode="ns_only",
                regularization=1e-4,
                mu_sigma=0.2,
                line_search_backoff=0.95,
                refinement_iters=max(base.refinement_iters, 8),
                ns_max_iters=40,
                ns_damping=0.9,
            ),
            _clone_cfg(
                base,
                kkt_mode="ns_only",
                regularization=1e-3,
                mu_sigma=0.3,
                line_search_backoff=0.9,
                refinement_iters=max(base.refinement_iters, 10),
                ns_max_iters=50,
                ns_damping=0.8,
            ),
            _clone_cfg(
                base,
                kkt_mode="ns_only",
                regularization=1e-5,
                mu_sigma=0.05,
                line_search_backoff=0.97,
                refinement_iters=max(base.refinement_iters, 8),
                ns_max_iters=40,
                ns_damping=1.0,
            ),
        ]

    return [
        _clone_cfg(base, kkt_mode="robust"),
        _clone_cfg(
            base,
            kkt_mode="robust",
            regularization=1e-4,
            mu_sigma=0.2,
            line_search_backoff=0.95,
            krylov_tol=1e-7,
            krylov_max_iters=max(base.krylov_max_iters, 200),
            robust_switch_residual=2e-1,
            ns_max_iters=40,
            ns_damping=0.9,
        ),
        _clone_cfg(
            base,
            kkt_mode="robust",
            regularization=1e-3,
            mu_sigma=0.3,
            line_search_backoff=0.9,
            krylov_tol=1e-6,
            krylov_max_iters=max(base.krylov_max_iters, 280),
            robust_switch_residual=1e-2,
            ns_max_iters=50,
            ns_damping=0.8,
        ),
        _clone_cfg(
            base,
            kkt_mode="robust",
            regularization=1e-4,
            mu_sigma=0.05,
            line_search_backoff=0.97,
            krylov_tol=1e-7,
            krylov_solver="gmres",
            krylov_max_iters=max(base.krylov_max_iters, 220),
            robust_switch_residual=5e-2,
            ns_max_iters=40,
            ns_damping=1.0,
        ),
    ]


def _tune_ipm_config(
    *,
    base_cfg: GEMMIPMConfig,
    mode: str,
    problem: DenseBatchQP,
    tune_batch: int,
    tune_iters: int,
    rank: int,
    study: str,
) -> GEMMIPMConfig:
    candidates = _candidate_ipm_configs(base_cfg, mode=mode)
    tune_problem = problem
    if problem.batch_size > tune_batch:
        tune_problem = problem.slice_batch(0, tune_batch)

    best_cfg = candidates[0]
    best_score = float("inf")
    best_status = "failed"
    best_metrics: dict[str, Any] = {}

    for idx, cand in enumerate(candidates):
        eval_cfg = deepcopy(cand)
        eval_cfg.max_iters = min(eval_cfg.max_iters, tune_iters)
        solver = GemmIPMQPSolver(eval_cfg)

        t0 = time.perf_counter()
        result = solver.solve(tune_problem)
        elapsed = time.perf_counter() - t0
        score = _ipm_objective(result.metrics, result.status, elapsed_s=elapsed)
        if score < best_score:
            best_score = score
            best_cfg = cand
            best_status = result.status
            best_metrics = result.metrics

        if rank == 0:
            print(
                "[ns-grid]"
                f" study={study} mode={mode} cand={idx}"
                f" score={score:.6g} status={result.status}"
                f" p={_as_float_or_inf(result.metrics.get('primal_residual')):.3e}"
                f" d={_as_float_or_inf(result.metrics.get('dual_residual')):.3e}"
                f" g={_as_float_or_inf(result.metrics.get('duality_gap')):.3e}"
                f" t={elapsed:.3f}s",
                flush=True,
            )

    if rank == 0:
        print(
            "[ns-grid] selected"
            f" study={study} mode={mode}"
            f" score={best_score:.6g} status={best_status}"
            f" p={_as_float_or_inf(best_metrics.get('primal_residual')):.3e}"
            f" d={_as_float_or_inf(best_metrics.get('dual_residual')):.3e}"
            f" g={_as_float_or_inf(best_metrics.get('duality_gap')):.3e}",
            flush=True,
        )
    return best_cfg


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
        "--max-experiments",
        str(args.max_experiments),
        "--ns-grid-tune-batch",
        str(args.ns_grid_tune_batch),
        "--ns-grid-tune-iters",
        str(args.ns_grid_tune_iters),
        "--target-tol",
        str(args.target_tol),
        "--max-iter-scale",
        str(args.max_iter_scale),
    ]
    if args.disable_ns_grid_search:
        cmd.append("--disable-ns-grid-search")
    if args.skip_splitting:
        cmd.append("--skip-splitting")
    if args.skip_cpu_baselines:
        cmd.append("--skip-cpu-baselines")
    if args.focus_ns_baselines:
        cmd.append("--focus-ns-baselines")
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
        choices=["quick", "large", "a100_heavy", "ns_favor", "mixed", "stress"],
        default="quick",
        help="Experiment preset (`ns_favor`/`mixed`/`stress` recommended for discovery sweeps)",
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
        default=[1e-1, 5e-2, 1e-2],
        help="Tolerance tiers used for standardized pass/fail reporting",
    )
    parser.add_argument(
        "--max-experiments",
        type=int,
        default=10,
        help="Maximum number of experiments to execute in one reproduce run (<=0 means no cap)",
    )
    parser.add_argument(
        "--disable-ns-grid-search",
        action="store_true",
        help="Disable NS-IPM hyperparameter search and use static defaults",
    )
    parser.add_argument(
        "--skip-splitting",
        action="store_true",
        help="Skip gemm_splitting_qp runs",
    )
    parser.add_argument(
        "--skip-cpu-baselines",
        action="store_true",
        help="Skip CPU baseline wrappers (SciPy/OSQP/HiGHS)",
    )
    parser.add_argument(
        "--focus-ns-baselines",
        action="store_true",
        help="Run only NS-IPM variants + reliable CPU QP baselines (skip splitting/portfolio/LP)",
    )
    parser.add_argument(
        "--ns-grid-tune-batch",
        type=int,
        default=8,
        help="Batch size used during NS-IPM tuning",
    )
    parser.add_argument(
        "--ns-grid-tune-iters",
        type=int,
        default=40,
        help="Max IPM iterations used during NS-IPM tuning",
    )
    parser.add_argument(
        "--target-tol",
        type=float,
        default=1e-2,
        help="Target solver tolerance (used for primal/dual/gap stopping)",
    )
    parser.add_argument(
        "--max-iter-scale",
        type=float,
        default=1.5,
        help="Multiplier applied to default solver iteration budgets",
    )
    args = parser.parse_args()

    _maybe_autolaunch_torchrun(args)
    set_deterministic_seed(args.seed)

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    if torch.cuda.is_available():
        if world_size > 1:
            torch.cuda.set_device(local_rank)
            device = f"cuda:{local_rank}"
        else:
            device = "cuda"
    else:
        device = "cpu"
    print(f"[rank-bind] rank={rank} local_rank={local_rank} world_size={world_size} device={device}", flush=True)
    use_cuda = str(device).startswith("cuda")
    tol = float(args.target_tol)
    iter_scale = max(float(args.max_iter_scale), 0.1)
    include_splitting = not args.skip_splitting
    include_portfolio = True
    include_lp = True
    include_cpu_baselines = not args.skip_cpu_baselines
    if args.focus_ns_baselines:
        include_splitting = False
        include_portfolio = False
        include_lp = False

    base_ipm_ns_cfg = GEMMIPMConfig(
        kkt_mode="ns_only",
        precision="bf16" if use_cuda else "fp32",
        tol_p=tol,
        tol_d=tol,
        tol_g=tol,
        max_iters=max(20, int(round((120 if args.preset == "large" else 80) * iter_scale))),
    )
    base_ipm_rb_cfg = GEMMIPMConfig(
        kkt_mode="robust",
        precision="bf16" if use_cuda else "fp32",
        tol_p=tol,
        tol_d=tol,
        tol_g=tol,
        max_iters=max(20, int(round((200 if args.preset == "large" else 120) * iter_scale))),
    )
    split_cfg = GEMMSplittingQPConfig(
        linear_solver="cg_ns",
        tol_p=tol,
        tol_d=tol,
        max_iters=max(2000, int(round((50_000 if args.preset == "large" else 20_000) * iter_scale))),
    )
    split = GemmSplittingQPSolver(split_cfg) if include_splitting else None
    lp_cfg = LPFirstOrderConfig(
        max_iters=max(8000, int(round((120_000 if args.preset == "large" else 80_000) * iter_scale))),
        tol_p=tol,
        tol_d=tol,
    )
    lp_solver = LPFirstOrderSolver(lp_cfg) if include_lp else None

    records: list[Any] = []
    results_path = out_dir / "results.jsonl"

    def _persist_records() -> None:
        if rank == 0:
            write_jsonl(records, results_path)

    def _append_record(rec: Any | None) -> None:
        if rec is None:
            return
        records.append(rec)
        _persist_records()

    def _rank_barrier() -> None:
        if world_size <= 1:
            return
        if not dist.is_available() or not dist.is_initialized():
            return
        if torch.cuda.is_available():
            dist.barrier(device_ids=[local_rank])
        else:
            dist.barrier()

    studies: list[str]
    if args.study == "all":
        studies = ["main", "strong", "weak"]
    else:
        studies = [args.study]

    raw_total_experiments = _planned_experiment_count(
        studies=studies,
        device=device,
        seed=args.seed,
        preset=args.preset,
        world_size=world_size,
        per_gpu_batch=args.per_gpu_batch,
        strong_total_batch=args.strong_total_batch,
        include_cpu_baselines=include_cpu_baselines,
        include_splitting=include_splitting,
        include_portfolio=include_portfolio,
        include_lp=include_lp,
    )
    requested_cap = int(args.max_experiments)
    if requested_cap <= 0:
        total_experiments = raw_total_experiments
    else:
        total_experiments = min(raw_total_experiments, requested_cap)
    total_experiments = max(0, total_experiments)
    if rank == 0:
        cap_desc = "all" if requested_cap <= 0 else str(requested_cap)
        print(
            f"[plan] studies={studies}, preset={args.preset}, world_size={world_size}, "
            f"planned_experiments={total_experiments} (raw={raw_total_experiments}), rank0_device={device}, "
            f"target_tol={tol:.2e}, max_iter_scale={iter_scale:.2f}, "
            f"splitting={include_splitting}, portfolio={include_portfolio}, lp={include_lp}, "
            f"max_experiments={cap_desc}",
            flush=True,
        )
    progress = 0
    remaining_slots = total_experiments
    stop_all = total_experiments <= 0

    def _reserve_slot() -> bool:
        nonlocal remaining_slots
        if remaining_slots <= 0:
            return False
        remaining_slots -= 1
        return True

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
        if stop_all:
            break

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

        study_ipm_ns_cfg = deepcopy(base_ipm_ns_cfg)
        study_ipm_rb_cfg = deepcopy(base_ipm_rb_cfg)
        if configs and not args.disable_ns_grid_search:
            tune_problem = generate_dense_parametric_qp(configs[0])
            study_ipm_ns_cfg = _tune_ipm_config(
                base_cfg=study_ipm_ns_cfg,
                mode="ns_only",
                problem=tune_problem,
                tune_batch=max(1, args.ns_grid_tune_batch),
                tune_iters=max(5, args.ns_grid_tune_iters),
                rank=rank,
                study=study,
            )
            study_ipm_rb_cfg = _tune_ipm_config(
                base_cfg=study_ipm_rb_cfg,
                mode="robust",
                problem=tune_problem,
                tune_batch=max(1, args.ns_grid_tune_batch),
                tune_iters=max(5, args.ns_grid_tune_iters),
                rank=rank,
                study=study,
            )

        ipm_ns = GemmIPMQPSolver(study_ipm_ns_cfg)
        ipm_robust = GemmIPMQPSolver(study_ipm_rb_cfg)
        ns_solver_cfg_payload = _jsonable(asdict(study_ipm_ns_cfg))
        rb_solver_cfg_payload = _jsonable(asdict(study_ipm_rb_cfg))

        for idx, cfg in enumerate(configs):
            if stop_all:
                break
            problem = generate_dense_parametric_qp(cfg)
            problem_id = f"{study}_dense_parametric_{idx}"
            cfg_payload = _jsonable(asdict(cfg))

            if not _reserve_slot():
                stop_all = True
                break
            _log_start("gemm_ipm_ns", problem_id)
            rec = run_qp_solver(
                run_id=f"{study}_run_{idx}_gemm_ipm_ns",
                problem_id=problem_id,
                solver_name="gemm_ipm_ns",
                solver=ipm_ns.solve,
                problem=problem,
                warmups=args.warmups,
                repeats=args.repeats,
                tol_p=study_ipm_ns_cfg.tol_p,
                tol_d=study_ipm_ns_cfg.tol_d,
                tol_g=study_ipm_ns_cfg.tol_g,
                tiers=args.eval_tiers,
                extra_metadata={
                    "experiment": "parametric_qp",
                    "study": study,
                    "preset": args.preset,
                    "config": cfg_payload,
                    "solver_config": ns_solver_cfg_payload,
                    "eval_tiers": args.eval_tiers,
                },
            )
            if rec is not None:
                _append_record(rec)
                _log_end("gemm_ipm_ns", rec.status, rec.timing.median_s)

            if not _reserve_slot():
                stop_all = True
                break
            _log_start("gemm_ipm_robust", problem_id)
            rec = run_qp_solver(
                run_id=f"{study}_run_{idx}_gemm_ipm_robust",
                problem_id=problem_id,
                solver_name="gemm_ipm_robust",
                solver=ipm_robust.solve,
                problem=problem,
                warmups=args.warmups,
                repeats=args.repeats,
                tol_p=study_ipm_rb_cfg.tol_p,
                tol_d=study_ipm_rb_cfg.tol_d,
                tol_g=study_ipm_rb_cfg.tol_g,
                tiers=args.eval_tiers,
                extra_metadata={
                    "experiment": "parametric_qp",
                    "study": study,
                    "preset": args.preset,
                    "config": cfg_payload,
                    "solver_config": rb_solver_cfg_payload,
                    "eval_tiers": args.eval_tiers,
                },
            )
            if rec is not None:
                _append_record(rec)
                _log_end("gemm_ipm_robust", rec.status, rec.timing.median_s)

            if include_splitting and split is not None:
                if not _reserve_slot():
                    stop_all = True
                    break
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
                    _append_record(rec)
                    _log_end("gemm_splitting_qp", rec.status, rec.timing.median_s)

            if study == "main" and include_cpu_baselines:
                _rank_barrier()
                if not _reserve_slot():
                    stop_all = True
                    break
                if rank == 0:
                    baseline_problem = problem.slice_batch(0, 1).to("cpu")
                    bcfg = BaselineRunConfig(
                        tol_p=tol,
                        tol_d=tol,
                        tol_g=tol,
                        max_iters=max(5000, int(round(10000 * iter_scale))),
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
                        _append_record(scipy_row)
                        _log_end("scipy_trust_constr_cpu", scipy_row["status"], float(scipy_row["timing"]["median_s"]))
                    else:
                        _log_end("scipy_trust_constr_cpu", "unavailable", None)

                if not _reserve_slot():
                    stop_all = True
                    break
                if rank == 0:
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
                        _append_record(osqp_row)
                        _log_end("osqp_cpu", osqp_row["status"], float(osqp_row["timing"]["median_s"]))
                    else:
                        _log_end("osqp_cpu", "unavailable", None)
                _rank_barrier()

        if stop_all:
            break

        if study == "main" and include_portfolio:
            for idx, cfg in enumerate(_make_portfolio_configs(device, args.seed, args.preset)):
                if stop_all:
                    break
                problem = generate_portfolio_qp(cfg)
                problem_id = f"{study}_portfolio_{idx}"
                cfg_payload = _jsonable(asdict(cfg))

                if not _reserve_slot():
                    stop_all = True
                    break
                _log_start("gemm_ipm_robust", problem_id)
                rec = run_qp_solver(
                    run_id=f"{study}_portfolio_{idx}_gemm_ipm_robust",
                    problem_id=problem_id,
                    solver_name="gemm_ipm_robust",
                    solver=ipm_robust.solve,
                    problem=problem,
                    warmups=args.warmups,
                    repeats=args.repeats,
                    tol_p=study_ipm_rb_cfg.tol_p,
                    tol_d=study_ipm_rb_cfg.tol_d,
                    tol_g=study_ipm_rb_cfg.tol_g,
                    tiers=args.eval_tiers,
                    extra_metadata={
                        "experiment": "portfolio_qp",
                        "study": study,
                        "preset": args.preset,
                        "config": cfg_payload,
                        "solver_config": rb_solver_cfg_payload,
                        "eval_tiers": args.eval_tiers,
                    },
                )
                if rec is not None:
                    _append_record(rec)
                    _log_end("gemm_ipm_robust", rec.status, rec.timing.median_s)

        if study == "main" and include_lp and lp_solver is not None:
            for idx, cfg in enumerate(_make_lp_configs(device, args.seed, args.preset)):
                if stop_all:
                    break
                lp = generate_batched_lp(cfg)
                problem_id = f"{study}_lp_{idx}"
                cfg_payload = _jsonable(asdict(cfg))

                if not _reserve_slot():
                    stop_all = True
                    break
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
                    _append_record(rec)
                    _log_end("lp_first_order_gpu", rec.status, rec.timing.median_s)

                if not include_cpu_baselines:
                    continue
                if not _reserve_slot():
                    stop_all = True
                    break
                _rank_barrier()
                if rank == 0:
                    lp_cpu = lp.slice_batch(0, 1).to("cpu")
                    _log_start("highs_cpu", problem_id)
                    highs_res = run_highs_lp(
                        lp_cpu,
                        BaselineRunConfig(tol_p=tol, tol_d=tol, max_iters=max(20000, int(round(100000 * iter_scale)))),
                    )
                    highs_row = _baseline_to_record(
                        run_id=f"{study}_lp_{idx}_highs_cpu",
                        category="lp",
                        problem_id=problem_id,
                        solver_name="highs_cpu",
                        baseline=highs_res,
                        eval_tiers=args.eval_tiers,
                        tol_p=tol,
                        tol_d=tol,
                        tol_g=None,
                    )
                    if highs_row is not None:
                        _append_record(highs_row)
                        _log_end("highs_cpu", highs_row["status"], float(highs_row["timing"]["median_s"]))
                    else:
                        _log_end("highs_cpu", "unavailable", None)
                _rank_barrier()

    if rank == 0:
        _persist_records()
        print(f"Completed {progress}/{total_experiments} planned experiments")
        print(f"Wrote {len(records)} records to {results_path}")


if __name__ == "__main__":
    main()
