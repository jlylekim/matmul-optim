from __future__ import annotations

import heapq
import math
import time
from dataclasses import dataclass, field
from typing import Callable

import torch

from gemm_kkt.solvers.common import SolverResult
from gemm_kkt.solvers.types import DenseBatchQP


@dataclass(order=True)
class _QueuedNode:
    priority: float
    node: "BnBNode" = field(compare=False)


@dataclass
class BnBNode:
    node_id: int
    assignment: dict[int, int]
    depth: int


@dataclass
class BnBResult:
    status: str
    incumbent_value: float
    lower_bound: float
    gap: float
    nodes_processed: int
    incumbent_x: torch.Tensor | None
    time_s: float


class BatchedBranchAndBound:
    """Optional MIQP wrapper where node relaxations are convex QPs.

    The user must provide:
    - node_to_relaxation(node): returns a convex QP relaxation for that node
    - choose_branch_var(x_relaxed, assignment): variable index to branch on, or None if integral
    - objective_fn(x): objective value for incumbent update
    """

    def __init__(
        self,
        relaxation_solver: Callable[[DenseBatchQP], SolverResult],
        *,
        batch_size: int = 32,
        max_nodes: int = 50_000,
        eps_gap: float = 1e-4,
    ):
        self.relaxation_solver = relaxation_solver
        self.batch_size = batch_size
        self.max_nodes = max_nodes
        self.eps_gap = eps_gap

    def solve(
        self,
        *,
        root: BnBNode,
        node_to_relaxation: Callable[[list[BnBNode]], DenseBatchQP],
        choose_branch_var: Callable[[torch.Tensor, dict[int, int]], int | None],
        objective_fn: Callable[[torch.Tensor], float],
    ) -> BnBResult:
        t0 = time.perf_counter()

        queue: list[_QueuedNode] = []
        heapq.heappush(queue, _QueuedNode(priority=-math.inf, node=root))

        incumbent = math.inf
        incumbent_x: torch.Tensor | None = None
        global_lb = -math.inf
        processed = 0

        while queue and processed < self.max_nodes:
            batch_nodes: list[BnBNode] = []
            while queue and len(batch_nodes) < self.batch_size:
                batch_nodes.append(heapq.heappop(queue).node)

            qp_batch = node_to_relaxation(batch_nodes)
            res = self.relaxation_solver(qp_batch)

            x = res.x
            if x.dim() == 1:
                x = x.unsqueeze(0)

            # Lower bound from relaxation objective proxy.
            relax_obj = torch.einsum("bi,bi->b", qp_batch.q.float(), x.float())

            for i, node in enumerate(batch_nodes):
                processed += 1
                lb = float(relax_obj[i].item())
                global_lb = max(global_lb, lb)
                if lb >= incumbent:
                    continue

                xi = x[i]
                branch_idx = choose_branch_var(xi, node.assignment)
                if branch_idx is None:
                    cand = objective_fn(xi)
                    if cand < incumbent:
                        incumbent = cand
                        incumbent_x = xi.detach().clone()
                    continue

                left = BnBNode(
                    node_id=processed * 2,
                    assignment={**node.assignment, branch_idx: 0},
                    depth=node.depth + 1,
                )
                right = BnBNode(
                    node_id=processed * 2 + 1,
                    assignment={**node.assignment, branch_idx: 1},
                    depth=node.depth + 1,
                )
                heapq.heappush(queue, _QueuedNode(priority=lb, node=left))
                heapq.heappush(queue, _QueuedNode(priority=lb, node=right))

            if math.isfinite(incumbent):
                gap = max(0.0, incumbent - global_lb) / max(1.0, abs(incumbent))
                if gap <= self.eps_gap:
                    total = time.perf_counter() - t0
                    return BnBResult(
                        status="optimality_gap_met",
                        incumbent_value=incumbent,
                        lower_bound=global_lb,
                        gap=gap,
                        nodes_processed=processed,
                        incumbent_x=incumbent_x,
                        time_s=total,
                    )

        total = time.perf_counter() - t0
        if math.isfinite(incumbent):
            gap = max(0.0, incumbent - global_lb) / max(1.0, abs(incumbent))
            status = "max_nodes"
        else:
            gap = math.inf
            status = "infeasible_or_no_incumbent"

        return BnBResult(
            status=status,
            incumbent_value=incumbent,
            lower_bound=global_lb,
            gap=gap,
            nodes_processed=processed,
            incumbent_x=incumbent_x,
            time_s=total,
        )
