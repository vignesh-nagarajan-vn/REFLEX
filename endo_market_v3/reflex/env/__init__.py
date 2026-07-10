"""Structural ground-truth market environment (``T_true``).

Sub-modules:

* :mod:`bonds` -- the static bond universe and cross-bond correlation.
* :mod:`liquidity_field` -- the latent, coupled liquidity dynamics.
* :mod:`clients` -- uninformed + ``alpha``-controlled informed order flow.
* :mod:`simulator` -- the ``StructuralSimulator`` that ties them together.
"""

from .bonds import BondUniverse
from .clients import ClientFlowModel, FlowSample
from .liquidity_field import LiquidityField
from .simulator import StructuralSimulator

__all__ = [
    "BondUniverse",
    "ClientFlowModel",
    "FlowSample",
    "LiquidityField",
    "StructuralSimulator",
]
