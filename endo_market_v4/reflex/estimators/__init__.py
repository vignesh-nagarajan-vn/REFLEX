"""Model-free estimators of the distribution sensitivity ``epsilon``.

Three independent measurement instruments (the 1.1 triangulation protocol):

* :mod:`.br_slope` -- the best-response-slope contraction modulus via symmetric
  finite differences with common random numbers (decision-space leg).
* :mod:`.sinkhorn` -- Wasserstein rate of the induced toxic-flow distribution,
  via exact 1-D quantile coupling or debiased log-domain Sinkhorn
  (distribution-space leg).
* :mod:`.cks` -- the Cont-Kukanov-Stoikov informed-flow slope from a fitted
  structural flow curve (structural-fit leg).

:mod:`.triangulate` runs all three against the closed-form ``epsilon`` of
:mod:`reflex.theory.analytic_boundary` -- agreement across the legs is the
evidentiary bar for the analytic boundary theorem.
"""

from .br_slope import (
    ResponseModulusResult,
    RGDResponseResult,
    measure_response_modulus,
    measure_rgd_response,
)
from .cks import CKSEpsilonResult, estimate_epsilon_cks
from .sinkhorn import (
    TUNED_REL_REG,
    SinkhornEpsilonResult,
    SinkhornTuning,
    estimate_epsilon_sinkhorn,
    quantile_w1,
    sinkhorn_divergence,
    tune_sinkhorn_reg,
    tune_sinkhorn_reg_for_config,
)
from .triangulate import TriangulationResult, triangulate_epsilon

__all__ = [
    "ResponseModulusResult",
    "RGDResponseResult",
    "measure_response_modulus",
    "measure_rgd_response",
    "CKSEpsilonResult",
    "estimate_epsilon_cks",
    "TUNED_REL_REG",
    "SinkhornEpsilonResult",
    "SinkhornTuning",
    "estimate_epsilon_sinkhorn",
    "quantile_w1",
    "sinkhorn_divergence",
    "tune_sinkhorn_reg",
    "tune_sinkhorn_reg_for_config",
    "TriangulationResult",
    "triangulate_epsilon",
]
