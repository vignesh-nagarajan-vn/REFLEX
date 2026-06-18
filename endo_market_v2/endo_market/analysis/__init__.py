"""Analysis: convergence diagnostics, the BR-slope modulus estimator, sweeps,
market metrics, and plotting."""

from .convergence import (
    classify_run,
    empirical_lipschitz,
    fixed_point_residual,
    is_oscillating,
)
from .response_modulus import ResponseModulusResult, measure_response_modulus
from .metrics import MarketMetrics, compute_metrics
from .sweep import SweepPoint, SweepResult, load_sweep_spec, run_sweep

__all__ = [
    "classify_run",
    "empirical_lipschitz",
    "fixed_point_residual",
    "is_oscillating",
    "ResponseModulusResult",
    "measure_response_modulus",
    "MarketMetrics",
    "compute_metrics",
    "SweepPoint",
    "SweepResult",
    "load_sweep_spec",
    "run_sweep",
]
