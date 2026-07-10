"""Equilibrium machinery: data collection, operator fitting, policy
optimization and the outer retraining loops (blind RRM and PerfGD-corrected).

The 1-D analytic PerfGD machinery (cobweb vs. corrected-gradient dynamics on
the central half-spread) lives in :mod:`reflex.theory.perfgd`; this subpackage
holds the full ML loops that drive the simulator, the learned operator and the
policy optimizer.
"""

from .data_collection import collect, collect_initial_states
from .fit_operator import (
    DeploymentRecord,
    FitResult,
    build_dataset,
    fit_operator,
    fit_operator_windowed,
    transition_rows,
)
from .optimize_policy import OptimizeResult, optimize_policy, rgd_step

__all__ = [
    "collect",
    "collect_initial_states",
    "DeploymentRecord",
    "FitResult",
    "fit_operator",
    "fit_operator_windowed",
    "build_dataset",
    "transition_rows",
    "OptimizeResult",
    "optimize_policy",
    "rgd_step",
]

from .rrm_loop import RRMIterate, RRMTrajectory, run_rrm  # noqa: E402

__all__ += ["RRMIterate", "RRMTrajectory", "run_rrm"]

from .loops import LOOP_MODES, LoopDiagnostics, LoopResult, run_loop  # noqa: E402

__all__ += ["LOOP_MODES", "LoopDiagnostics", "LoopResult", "run_loop"]
