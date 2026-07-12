"""endo_market: performative prediction in an endogenous bond market-making env.

The package studies Repeated Risk Minimization (RRM) for a dealer policy whose
deployment changes the very distribution of market data it is trained on.  The
headline experiment characterizes when the policy<->distribution loop converges
vs. diverges as a function of how adversarial (toxic) the market is.
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

__version__ = "0.1.0"
