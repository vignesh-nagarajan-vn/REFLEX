"""reflex: performative prediction in an endogenous OTC bond market (v3).

Third-generation REFLEX codebase (``endo_market_v3``).  It unifies, in one
package:

* the structural market simulator and the learned Market Response Operator
  ``T_theta`` (lineage: ``endo_market_v2``), with the operator **un-blinded** --
  fit across a window of deployments so it can learn the distribution response
  ``dD/dphi``;
* the five closed-form math-theory results (1.1 analytic stability boundary,
  1.2 PerfGD correction, 1.3 multi-dealer systemic risk, 1.4 distributionally
  robust boundary, 1.5 factor-model scaling) as :mod:`reflex.theory`;
* the epsilon-estimator triangulation (BR-slope / Sinkhorn / CKS) as
  :mod:`reflex.estimators`; and
* real-data calibration (VIX-regime microstructure fits, 1990-2026) as
  :mod:`reflex.calibration`.

Everything is CPU-only and reproducible from ``(config, seed)``.
"""

from __future__ import annotations

from .config import Config, load_config
from .types import (
    Fills,
    MarketState,
    Quotes,
    Transition,
    policy_summary,
)
from .utils.seeding import seed_everything

__all__ = [
    "Config",
    "load_config",
    "MarketState",
    "Quotes",
    "Fills",
    "Transition",
    "policy_summary",
    "seed_everything",
]

__version__ = "3.0.0"
