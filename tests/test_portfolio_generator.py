from __future__ import annotations

import torch

from benchmarks.generators.portfolio_qp import PortfolioQPConfig, generate_portfolio_qp


def test_portfolio_qp_generator_shapes() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = PortfolioQPConfig(batch_size=4, n_assets=32, n_factors=4, seed=7, device=device)
    qp = generate_portfolio_qp(cfg)

    assert qp.q.shape == (4, 32)
    assert qp.A.shape == (1 + 32, 32)
    assert qp.l.shape == (4, 1 + 32)
    assert qp.u.shape == (4, 1 + 32)
