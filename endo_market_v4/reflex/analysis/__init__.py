"""Analysis: convergence diagnostics, sweeps, market metrics, and plotting.

The closed-form theory modules live in :mod:`reflex.theory`; the model-free
``epsilon`` estimators live in :mod:`reflex.estimators`.  This subpackage holds
the run-level diagnostics and aggregation built on top of them.
"""

from .convergence import (
    classify_run,
    empirical_lipschitz,
    fixed_point_residual,
    is_oscillating,
)
from .fragility import FragilityResult, compute_fragility, save_fragility
from .metrics import MarketMetrics, compute_metrics
from .sweep import SweepPoint, SweepResult, load_sweep_spec, run_sweep

__all__ = [
    "classify_run",
    "empirical_lipschitz",
    "fixed_point_residual",
    "is_oscillating",
    "FragilityResult",
    "compute_fragility",
    "save_fragility",
    "MarketMetrics",
    "compute_metrics",
    "SweepPoint",
    "SweepResult",
    "load_sweep_spec",
    "run_sweep",
]
