"""Verification layer (v4): machine-checkable justification of the theory.

Two complementary halves:

* :mod:`.certificates` -- **numerical proof certificates**: every load-bearing
  identity, inequality and dynamical claim of theory 1.1-1.6 is re-derived
  numerically (finite differences, spectral checks, Monte Carlo rates,
  constructed counter-regimes) against the closed-form implementations, on
  any config.  These run in seconds and are part of the test suite.
* ``endo_market_v4/lean/`` -- **Lean 4 formalisations** of the *logical
  skeletons* of the same results (the contraction argument, the lazy-deploy
  algebra, the N_eff eigen-identity, the robust-certificate soundness, the
  echo-chamber separation).  See ``lean/README.md`` for scope and the honest
  compile status.

The split is deliberate: the parts of each proof that are pure logic /
algebra are stated formally; the parts that are *model-specific calculus*
(the GLFT curvature integrals, the gate means, the fixed-point locations)
are certified numerically where a formal real-analysis development would
add cost without adding assurance about the *code*.
"""

from .certificates import (
    Certificate,
    certify_boundary,
    certify_factor_scaling,
    certify_lazy_deploy,
    certify_multi_dealer,
    certify_perfgd,
    certify_perfgd_dynamics,
    certify_robust,
    run_all_certificates,
)

__all__ = [
    "Certificate",
    "certify_boundary",
    "certify_factor_scaling",
    "certify_lazy_deploy",
    "certify_multi_dealer",
    "certify_perfgd",
    "certify_perfgd_dynamics",
    "certify_robust",
    "run_all_certificates",
]
