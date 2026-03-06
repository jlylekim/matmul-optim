from __future__ import annotations

"""Dataset utilities for real LP benchmark suites."""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from benchmarks.datasets.load_lp_instances import CanonicalLP

__all__ = ["CanonicalLP"]


def __getattr__(name: str) -> Any:
    if name == "CanonicalLP":
        from benchmarks.datasets.load_lp_instances import CanonicalLP

        return CanonicalLP
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
