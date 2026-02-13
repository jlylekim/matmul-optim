from __future__ import annotations

import argparse
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from benchmarks.generators.batched_lp import BatchedLPConfig, generate_batched_lp
from benchmarks.generators.dense_parametric_qp import DenseParametricQPConfig, generate_dense_parametric_qp
from benchmarks.generators.portfolio_qp import PortfolioQPConfig, generate_portfolio_qp
from benchmarks.harness import run_lp_solver, run_qp_solver, write_jsonl
from gemm_kkt.baselines.base import BaselineRunConfig
from gemm_kkt.baselines.cpu_refs import run_highs_lp, run_osqp_qp
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


def _make_main_configs(device: str, seed: int) -> list[DenseParametricQPConfig]:
    return [
        DenseParametricQPConfig(batch_size=32, n=128, m=256, rhs_count=1, shared_matrices=True, seed=seed, device=device),
        DenseParametricQPConfig(batch_size=64, n=128, m=256, rhs_count=4, shared_matrices=True, seed=seed + 1, device=device),
        DenseParametricQPConfig(batch_size=16, n=256, m=512, rhs_count=2, shared_matrices=True, seed=seed + 2, device=device),
    ]


def _make_portfolio_configs(device: str, seed: int) -> list[PortfolioQPConfig]:
    return [
        PortfolioQPConfig(batch_size=64, n_assets=128, n_factors=8, seed=seed + 10, device=device),
        PortfolioQPConfig(batch_size=32, n_assets=256, n_factors=12, seed=seed + 11, device=device),
    ]


def _make_lp_configs(device: str, seed: int) -> list[BatchedLPConfig]:
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce GEMM-KKT experiments")
    parser.add_argument("--output", type=str, required=True, help="Artifact output directory")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--study", type=str, choices=["main", "strong", "weak", "all"], default="all")
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--per-gpu-batch", type=int, default=64)
    parser.add_argument("--strong-total-batch", type=int, default=512)
    args = parser.parse_args()

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
        max_iters=40,
    )
    ipm_rb_cfg = GEMMIPMConfig(
        kkt_mode="robust",
        precision="bf16" if device == "cuda" else "fp32",
        tol_p=1e-6,
        tol_d=1e-6,
        tol_g=1e-6,
        max_iters=60,
    )
    split_cfg = GEMMSplittingQPConfig(
        linear_solver="cg_ns",
        tol_p=1e-4,
        tol_d=1e-4,
        max_iters=2000,
    )

    ipm_ns = GemmIPMQPSolver(ipm_ns_cfg)
    ipm_robust = GemmIPMQPSolver(ipm_rb_cfg)
    split = GemmSplittingQPSolver(split_cfg)
    lp_cfg = LPFirstOrderConfig(max_iters=8_000, tol_p=1e-4, tol_d=1e-4)
    lp_solver = LPFirstOrderSolver(lp_cfg)

    records: list[Any] = []

    studies: list[str]
    if args.study == "all":
        studies = ["main", "strong", "weak"]
    else:
        studies = [args.study]

    for study in studies:
        if study == "main":
            configs = _make_main_configs(device, args.seed)
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
                extra_metadata={
                    "experiment": "parametric_qp",
                    "study": study,
                    "config": cfg_payload,
                },
            )
            if rec is not None:
                records.append(rec)

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
                extra_metadata={
                    "experiment": "parametric_qp",
                    "study": study,
                    "config": cfg_payload,
                },
            )
            if rec is not None:
                records.append(rec)

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
                extra_metadata={
                    "experiment": "parametric_qp",
                    "study": study,
                    "config": cfg_payload,
                },
            )
            if rec is not None:
                records.append(rec)

            if rank == 0 and study == "main":
                baseline_problem = problem.slice_batch(0, 1).to("cpu")
                bcfg = BaselineRunConfig(
                    tol_p=1e-4,
                    tol_d=1e-4,
                    max_iters=10000,
                    warm_start=False,
                    presolve=True,
                )
                osqp_res = run_osqp_qp(baseline_problem, bcfg)
                if osqp_res.status != "unavailable":
                    records.append(
                        {
                            "run_id": f"{study}_run_{idx}_osqp_cpu",
                            "category": "qp_conic",
                            "problem_id": problem_id,
                            "solver": "osqp_cpu",
                            "status": osqp_res.status,
                            "timing": {
                                "median_s": osqp_res.solve_time_s,
                                "p90_s": osqp_res.solve_time_s,
                                "repeats": 1,
                            },
                            "metrics": osqp_res.metrics,
                            "metadata": {
                                "batch_total": 1,
                                "batch_per_rank": 1,
                                "distributed": False,
                                "world_size": 1,
                                "throughput_prob_per_s": 1.0 / max(osqp_res.solve_time_s, 1e-12),
                                "solver_metadata": osqp_res.metadata,
                                "baseline_version": osqp_res.version,
                            },
                            "system": {"cpu_baseline": True},
                        }
                    )

        if study == "main":
            for idx, cfg in enumerate(_make_portfolio_configs(device, args.seed)):
                problem = generate_portfolio_qp(cfg)
                problem_id = f"{study}_portfolio_{idx}"
                cfg_payload = _jsonable(asdict(cfg))

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
                    extra_metadata={
                        "experiment": "portfolio_qp",
                        "study": study,
                        "config": cfg_payload,
                    },
                )
                if rec is not None:
                    records.append(rec)

            for idx, cfg in enumerate(_make_lp_configs(device, args.seed)):
                lp = generate_batched_lp(cfg)
                problem_id = f"{study}_lp_{idx}"
                cfg_payload = _jsonable(asdict(cfg))

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
                    extra_metadata={
                        "experiment": "batched_lp",
                        "study": study,
                        "config": cfg_payload,
                    },
                )
                if rec is not None:
                    records.append(rec)

                if rank == 0:
                    lp_cpu = lp.slice_batch(0, 1).to("cpu")
                    highs_res = run_highs_lp(lp_cpu, BaselineRunConfig(tol_p=1e-4, tol_d=1e-4, max_iters=100000))
                    if highs_res.status != "unavailable":
                        records.append(
                            {
                                "run_id": f"{study}_lp_{idx}_highs_cpu",
                                "category": "lp",
                                "problem_id": problem_id,
                                "solver": "highs_cpu",
                                "status": highs_res.status,
                                "timing": {
                                    "median_s": highs_res.solve_time_s,
                                    "p90_s": highs_res.solve_time_s,
                                    "repeats": 1,
                                },
                                "metrics": highs_res.metrics,
                                "metadata": {
                                    "batch_total": 1,
                                    "batch_per_rank": 1,
                                    "distributed": False,
                                    "world_size": 1,
                                    "throughput_prob_per_s": 1.0 / max(highs_res.solve_time_s, 1e-12),
                                    "solver_metadata": highs_res.metadata,
                                    "baseline_version": highs_res.version,
                                },
                                "system": {"cpu_baseline": True},
                            }
                        )

    if rank == 0:
        write_jsonl(records, out_dir / "results.jsonl")
        print(f"Wrote {len(records)} records to {out_dir / 'results.jsonl'}")


if __name__ == "__main__":
    main()
