"""reflex: performative prediction in an endogenous OTC bond market (v4).

Fourth and final generation of the REFLEX codebase (``endo_market_v4``).  It
unifies, in one package:

* the structural market simulator and the learned Market Response Operator
  ``T_theta`` (lineage: ``endo_market_v2``), with the operator **un-blinded** --
  fit across a window of deployments so it can learn the distribution response
  ``dD/dphi``;
* the six closed-form math-theory results (1.1 analytic stability boundary,
  1.2 PerfGD correction, 1.3 multi-dealer systemic risk, 1.4 distributionally
  robust boundary, 1.5 factor-model scaling, 1.6 lazy deployment) as
  :mod:`reflex.theory`;
* the epsilon-estimator triangulation (BR-slope / Sinkhorn / CKS, with the
  tuned Sinkhorn blur) as :mod:`reflex.estimators`;
* real-data calibration (VIX-regime microstructure fits, 1990-2026) as
  :mod:`reflex.calibration`;
* the **structurally-anchored learned loop** (``perfgd_structural``) that
  closes the v3 loop-level gap, in :mod:`reflex.equilibrium`; and
* the verification layer (numerical proof certificates + the Lean skeletons
  in ``lean/``) as :mod:`reflex.verification`.

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

__version__ = "4.0.0"
