"""Analysis: convergence diagnostics, the BR-slope modulus estimator, sweeps,
market metrics, and plotting."""

from .convergence import (
    classify_run,
    empirical_lipschitz,
    fixed_point_residual,
    is_oscillating,
)
from .response_modulus import ResponseModulusResult, measure_response_modulus
from .analytic_boundary import (
    AnalyticBoundary,
    ReferenceState,
    analytic_boundary,
    best_response,
    reference_state,
    solve_fixed_point,
)
from .multi_dealer_modulus import (
    CommonModeProbe,
    MeanFieldLimit,
    MultiDealerBoundary,
    common_mode_probe,
    identify_kappa,
    joint_jacobian,
    mean_field_boundary,
    measure_common_mode_modulus,
    measure_differential_modulus,
    multi_dealer_boundary,
    n_eff,
    run_joint_rrm,
    strong_coupling_limit,
    sweep_dealer_count,
)
from .metrics import MarketMetrics, compute_metrics
from .sweep import SweepPoint, SweepResult, load_sweep_spec, run_sweep

__all__ = [
    "classify_run",
    "empirical_lipschitz",
    "fixed_point_residual",
    "is_oscillating",
    "ResponseModulusResult",
    "measure_response_modulus",
    # analytic stability boundary (math-theory 1.1 foundation)
    "AnalyticBoundary",
    "ReferenceState",
    "analytic_boundary",
    "best_response",
    "reference_state",
    "solve_fixed_point",
    # multi-dealer / systemic risk (math-theory 1.3)
    "CommonModeProbe",
    "MeanFieldLimit",
    "MultiDealerBoundary",
    "common_mode_probe",
    "identify_kappa",
    "joint_jacobian",
    "mean_field_boundary",
    "measure_common_mode_modulus",
    "measure_differential_modulus",
    "multi_dealer_boundary",
    "n_eff",
    "run_joint_rrm",
    "strong_coupling_limit",
    "sweep_dealer_count",
    "MarketMetrics",
    "compute_metrics",
    "SweepPoint",
    "SweepResult",
    "load_sweep_spec",
    "run_sweep",
]
