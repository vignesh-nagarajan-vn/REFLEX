"""Model-free estimators of the distribution sensitivity ``epsilon``.

Three independent measurement instruments (the 1.1 triangulation protocol):

* :mod:`.br_slope` -- the best-response-slope contraction modulus via symmetric
  finite differences with common random numbers (the v2 headline estimator).
* :mod:`.sinkhorn` -- entropic-OT (Sinkhorn) Wasserstein sensitivity of the
  induced flow distribution ``W(D(h+d), D(h-d)) / 2d``.  (Added in P3.)
* :mod:`.cks` -- the Cont-Kukanov-Stoikov informed-flow slope
  ``|d lambda_informed / d h|`` regressed from deployment logs.  (Added in P3.)

Agreement across the three (and with the closed-form ``epsilon`` from
:mod:`reflex.theory.analytic_boundary`) is the evidentiary bar for the analytic
boundary theorem.
"""

from .br_slope import ResponseModulusResult, measure_response_modulus

__all__ = [
    "ResponseModulusResult",
    "measure_response_modulus",
]
