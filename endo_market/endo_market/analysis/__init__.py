"""Analysis: convergence diagnostics, sweep aggregation, plotting, metrics."""

from .convergence import (
    classify_run,
    empirical_lipschitz,
    fixed_point_residual,
    is_oscillating,
)

__all__ = [
    "classify_run",
    "empirical_lipschitz",
    "fixed_point_residual",
    "is_oscillating",
]
