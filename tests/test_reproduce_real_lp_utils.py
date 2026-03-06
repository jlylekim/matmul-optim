from __future__ import annotations

from benchmarks.reproduce_real_lp import _failure_row, _status_from_baseline


def test_status_from_baseline_mapping() -> None:
    assert _status_from_baseline("optimal") == "solved"
    assert _status_from_baseline("solved") == "solved"
    assert _status_from_baseline("maximum iterations reached") == "max_iters"
    assert _status_from_baseline("infeasible") == "infeasible"
    assert _status_from_baseline("unbounded") == "unbounded"
    assert _status_from_baseline("unavailable") == "unavailable"
    assert _status_from_baseline("weird") == "failed"


def test_failure_row_contains_error_metadata() -> None:
    err = RuntimeError("boom")
    row = _failure_row(
        run_id="r0",
        problem_id="netlib:afiro",
        solver="gemm_ipm_ns",
        category="lp",
        suite="netlib",
        instance_name="afiro",
        rank=0,
        world_size=8,
        base_meta={"n": 10, "m": 20},
        stage="solve",
        status="failed",
        error=err,
        elapsed_s=1.25,
    )
    assert row["status"] == "failed"
    assert row["timing"]["median_s"] == 1.25
    assert row["metadata"]["error_type"] == "RuntimeError"
    assert "boom" in row["metadata"]["error_message"]
    assert row["metadata"]["rank"] == 0
    assert row["metadata"]["world_size"] == 8

