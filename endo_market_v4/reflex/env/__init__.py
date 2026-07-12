"""Structural ground-truth market environment (``T_true``).

Sub-modules:

* :mod:`bonds` -- the static bond universe and cross-bond correlation.
* :mod:`liquidity_field` -- the latent, coupled liquidity dynamics.
* :mod:`clients` -- uninformed + ``alpha``-controlled informed order flow.
* :mod:`simulator` -- the single-dealer ``StructuralSimulator``.
* :mod:`multi_dealer` -- the genuine ``N``-dealer market sharing one informed
  pool (math-theory 1.3); reduces bit-for-bit to the single-dealer market at
  ``N = 1``.
"""

from .bonds import BondUniverse
from .clients import ClientFlowModel, FlowSample
from .liquidity_field import LiquidityField
from .multi_dealer import (
    DealerStepResult,
    MultiDealerSimulator,
    MultiMarketState,
    MultiTransition,
)
from .simulator import StructuralSimulator

__all__ = [
    "BondUniverse",
    "ClientFlowModel",
    "FlowSample",
    "LiquidityField",
    "StructuralSimulator",
    "DealerStepResult",
    "MultiDealerSimulator",
    "MultiMarketState",
    "MultiTransition",
]
