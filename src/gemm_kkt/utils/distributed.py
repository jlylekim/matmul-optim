from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist

from gemm_kkt.solvers.types import DenseBatchLP, DenseBatchQP


@dataclass
class DistInfo:
    rank: int = 0
    world_size: int = 1
    local_rank: int = 0
    is_distributed: bool = False


def init_distributed(backend: str = "nccl") -> DistInfo:
    """Initialize distributed mode if launcher env vars are present."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if world_size <= 1:
        return DistInfo()

    if not dist.is_initialized():
        dist.init_process_group(backend=backend)

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    return DistInfo(rank=rank, world_size=world_size, local_rank=local_rank, is_distributed=True)


def barrier_if_distributed(info: DistInfo) -> None:
    if info.is_distributed and dist.is_initialized():
        dist.barrier()


def gather_objects(info: DistInfo, obj: Any) -> list[Any]:
    if not info.is_distributed:
        return [obj]
    gathered: list[Any] = [None for _ in range(info.world_size)]
    dist.all_gather_object(gathered, obj)
    return gathered


def _compute_shard(batch_size: int, rank: int, world_size: int) -> tuple[int, int]:
    base = batch_size // world_size
    rem = batch_size % world_size
    start = rank * base + min(rank, rem)
    end = start + base + (1 if rank < rem else 0)
    return start, end


def shard_qp(problem: DenseBatchQP, info: DistInfo) -> DenseBatchQP:
    if not info.is_distributed or problem.batch_size == 1:
        return problem
    start, end = _compute_shard(problem.batch_size, info.rank, info.world_size)
    return problem.slice_batch(start, end)


def shard_lp(problem: DenseBatchLP, info: DistInfo) -> DenseBatchLP:
    if not info.is_distributed or problem.batch_size == 1:
        return problem
    start, end = _compute_shard(problem.batch_size, info.rank, info.world_size)
    return problem.slice_batch(start, end)
