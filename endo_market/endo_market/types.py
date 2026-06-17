"""Core data structures shared across the project.

Tensors are used wherever differentiability matters (states, quotes, P&L
components) so that the same objects flow through both the (non-differentiable)
structural simulator ``T_true`` and the (differentiable) learned operator
``T_theta``.

Observability convention
------------------------
The dealer policy and the learned operator may only depend on *observable*
state: ``inventory``, ``flow_recent`` (signed recent net flow) and
``vol_recent`` (recent gross traded volume, used as a liquidity proxy).  The
``fundamental`` (latent true value) and ``liquidity`` (latent field) are hidden
-- this partial observability is exactly why the operator cannot anticipate how
toxicity shifts when the policy is redeployed, which is the source of
performative instability.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch
from torch import Tensor

# Number of observable per-bond features exposed to the policy / operator.
N_OBS_FEATURES = 3
# Length of the global policy summary vector fed into T_theta.
POLICY_SUMMARY_DIM = 3
# Canonical ordering of the next-state delta channels modelled by T_theta.
DELTA_KEYS: List[str] = ["d_inventory", "d_mid", "d_flow", "d_vol"]
# P&L component channels also predicted by T_theta (conditional expectations).
PNL_KEYS: List[str] = ["spread_capture", "inventory_pnl", "adverse_selection_loss"]


@dataclass
class MarketState:
    """Snapshot of the market across all bonds.

    Attributes
    ----------
    inventory, mid, flow_recent, vol_recent:
        Observable per-bond tensors of shape ``[N]``.
    fundamental, liquidity:
        Latent per-bond tensors of shape ``[N]`` (hidden from policy/operator).
    t:
        Integer time step within an episode.
    """

    inventory: Tensor
    mid: Tensor
    fundamental: Tensor  # latent
    liquidity: Tensor  # latent
    flow_recent: Tensor
    vol_recent: Tensor
    t: int = 0

    @property
    def n_bonds(self) -> int:
        return int(self.inventory.shape[-1])

    def observable_features(self) -> Tensor:
        """Return the ``[N, N_OBS_FEATURES]`` observable feature matrix."""
        return torch.stack([self.inventory, self.flow_recent, self.vol_recent], dim=-1)

    def detach(self) -> "MarketState":
        return MarketState(
            inventory=self.inventory.detach(),
            mid=self.mid.detach(),
            fundamental=self.fundamental.detach(),
            liquidity=self.liquidity.detach(),
            flow_recent=self.flow_recent.detach(),
            vol_recent=self.vol_recent.detach(),
            t=self.t,
        )

    def clone(self) -> "MarketState":
        return MarketState(
            inventory=self.inventory.clone(),
            mid=self.mid.clone(),
            fundamental=self.fundamental.clone(),
            liquidity=self.liquidity.clone(),
            flow_recent=self.flow_recent.clone(),
            vol_recent=self.vol_recent.clone(),
            t=self.t,
        )


@dataclass
class Quotes:
    """Dealer quotes per bond.

    ``half_spread`` is the (strictly positive) half-width around the mid.
    ``skew`` is a signed midpoint shift used to lean against inventory.
    The effective bid/ask are ``mid + skew -/+ half_spread``.
    """

    half_spread: Tensor
    skew: Tensor

    def bid(self, mid: Tensor) -> Tensor:
        return mid + self.skew - self.half_spread

    def ask(self, mid: Tensor) -> Tensor:
        return mid + self.skew + self.half_spread


@dataclass
class Fills:
    """Realized fills aggregated per bond for a single step.

    ``qty`` is the *signed* dealer inventory change (``+`` = dealer bought).
    ``gross_volume`` is total notional traded; ``informed_volume`` is the part
    coming from informed (toxic) counterparties.
    """

    qty: Tensor  # signed (+ = dealer bought)
    price: Tensor  # volume-weighted average trade price
    gross_volume: Tensor
    informed_volume: Tensor
    was_informed: Tensor  # bool: informed_volume > 0


@dataclass
class Transition:
    """A single ``(s, a, fills, s')`` transition with its P&L breakdown."""

    state: MarketState
    quotes: Quotes
    fills: Fills
    next_state: MarketState
    pnl_components: Dict[str, Tensor]  # keys in PNL_KEYS (+ optional extras), shape [N]


def policy_summary(state: MarketState, policy) -> Tensor:
    """Fixed-length descriptor of current quoting behaviour fed into ``T_theta``.

    Returns a length-``POLICY_SUMMARY_DIM`` vector
    ``[mean_half_spread, mean_abs_skew, spread_elasticity]`` where the elasticity
    proxy measures how strongly the policy widens its half-spread in response to
    a unit increase in inventory (estimated by finite differences and detached).

    The mean-half-spread and mean-skew terms are kept differentiable w.r.t. the
    policy parameters so this summary participates in pathwise gradients; the
    elasticity proxy is treated as a slowly-varying context.
    """
    quotes = policy.quote(state)
    mean_hs = quotes.half_spread.mean()
    mean_skew = quotes.skew.abs().mean()

    # Elasticity proxy: response of half-spread to a +1 inventory shock.
    with torch.no_grad():
        bumped = state.clone()
        bumped.inventory = bumped.inventory + 1.0
        q2 = policy.quote(bumped)
        elasticity = (q2.half_spread - quotes.half_spread).mean()

    return torch.stack([mean_hs, mean_skew, elasticity.detach()])
