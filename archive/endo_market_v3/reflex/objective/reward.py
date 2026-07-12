"""Dealer reward / objective ``R(phi; s)``.

The dealer's economic objective per step is

    objective = spread_capture + inventory_pnl - adverse_selection_loss
                - inv_risk_weight * inventory_after**2

(optionally scaled by ``pnl_scale``).  The *performative risk* we minimise is
``PR(phi) = - E_{s ~ D(phi)}[objective]``; throughout the codebase we return the
objective to be **maximised** and negate it where a loss is needed.

Two marking conventions are supported, because the dealer optimises against a
model that cannot see the latent fundamental:

* ``"fundamental"`` -- use the exact components from the structural simulator
  (these mark inventory carry on the true fundamental move).  Used to evaluate
  the true performative risk on fresh ``T_true`` rollouts.
* ``"observable"`` -- mark inventory carry on the observable *mid* move instead.
  The adverse-selection term is not directly observable; when optimising against
  the operator it is supplied by the operator's predicted P&L channel.  This
  convention is what makes the operator-side objective computable from observable
  quantities plus the operator's predictions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal, Mapping

import torch
from torch import Tensor

from ..config import RewardConfig
from ..types import MarketState, Transition

Marking = Literal["fundamental", "observable"]


@dataclass
class RewardBreakdown:
    """Per-component reward decomposition (each a scalar tensor, summed over bonds)."""

    spread_capture: Tensor
    inventory_pnl: Tensor
    adverse_selection_loss: Tensor
    inventory_risk: Tensor
    objective: Tensor  # spread_capture + inventory_pnl - adverse - inv_risk - quoting_cost, scaled
    quoting_cost: Tensor | None = None  # quadratic quoting-cost convexity (0 if disabled)

    def as_dict(self) -> Dict[str, float]:
        """Return a plain ``{name: float}`` view for logging."""
        return {
            "spread_capture": float(self.spread_capture),
            "inventory_pnl": float(self.inventory_pnl),
            "adverse_selection_loss": float(self.adverse_selection_loss),
            "inventory_risk": float(self.inventory_risk),
            "quoting_cost": float(self.quoting_cost) if self.quoting_cost is not None else 0.0,
            "objective": float(self.objective),
        }


def reward_from_components(
    pnl_components: Mapping[str, Tensor],
    inventory_after: Tensor,
    cfg: RewardConfig,
    reduce: bool = True,
    half_spread: Tensor | None = None,
) -> RewardBreakdown:
    """Assemble the dealer objective from P&L component tensors.

    This is the primitive used both for ``T_true`` transitions and for ``T_theta``
    rollouts (whose predicted P&L channels are passed in here).

    Parameters
    ----------
    pnl_components:
        Mapping with keys ``spread_capture``, ``inventory_pnl``,
        ``adverse_selection_loss`` (each ``[N]`` or ``[..., N]``).
    inventory_after:
        Post-trade inventory ``[N]`` (or broadcastable) for the quadratic risk
        penalty.
    cfg:
        Reward configuration (``inv_risk_weight``, ``pnl_scale``,
        ``quote_anchor_weight``, ``quote_anchor_ref``).
    reduce:
        If ``True`` (default) sum over all elements to a scalar; otherwise keep
        per-bond components.
    half_spread:
        The deployed/candidate half-spread ``[N]``.  When provided together with a
        positive ``quote_anchor_weight``, a quadratic quoting cost
        ``weight * (h - ref)**2`` is subtracted from the objective.  This convexity
        pins the dealer's optimum and makes the best-response sensitivity (and
        hence the cobweb modulus) finite and tunable.  Quoting costs / risk limits
        that keep market makers near a target spread are a standard friction.

    Returns
    -------
    RewardBreakdown
        The components and the (scaled) objective.
    """
    sc = pnl_components["spread_capture"]
    ip = pnl_components["inventory_pnl"]
    adv = pnl_components["adverse_selection_loss"]
    inv_risk = cfg.inv_risk_weight * inventory_after.pow(2)

    if half_spread is not None and cfg.quote_anchor_weight > 0.0:
        quoting_cost = cfg.quote_anchor_weight * (half_spread - cfg.quote_anchor_ref).pow(2)
    else:
        quoting_cost = torch.zeros_like(adv)

    if reduce:
        sc = sc.sum()
        ip = ip.sum()
        adv = adv.sum()
        inv_risk = inv_risk.sum()
        quoting_cost = quoting_cost.sum()

    objective = cfg.pnl_scale * (sc + ip - adv - inv_risk - quoting_cost)
    return RewardBreakdown(
        spread_capture=sc,
        inventory_pnl=ip,
        adverse_selection_loss=adv,
        inventory_risk=inv_risk,
        objective=objective,
        quoting_cost=quoting_cost,
    )


def _observable_components(transition: Transition) -> Dict[str, Tensor]:
    """Recompute P&L components marking inventory carry on the *mid* move.

    ``spread_capture`` and ``adverse_selection_loss`` are unchanged (the latter
    is still defined against the realised fundamental for evaluation purposes);
    only ``inventory_pnl`` is re-marked to the observable mid change
    ``q_after * (mid_next - mid)``.
    """
    state: MarketState = transition.state
    nxt: MarketState = transition.next_state
    q_after = nxt.inventory
    comp = dict(transition.pnl_components)
    comp["inventory_pnl"] = q_after * (nxt.mid - state.mid)
    return comp


def reward(
    transition: Transition,
    cfg: RewardConfig,
    marking: Marking = "fundamental",
    reduce: bool = True,
) -> RewardBreakdown:
    """Compute the dealer objective for a single ``T_true`` transition.

    Parameters
    ----------
    transition:
        A transition produced by the structural simulator.
    cfg:
        Reward configuration.
    marking:
        ``"fundamental"`` (exact) or ``"observable"`` (mark carry on the mid).
    reduce:
        Sum over bonds if ``True``.

    Returns
    -------
    RewardBreakdown
    """
    if marking == "fundamental":
        comps = transition.pnl_components
    elif marking == "observable":
        comps = _observable_components(transition)
    else:  # pragma: no cover - guarded by typing
        raise ValueError(f"unknown marking {marking!r}")
    return reward_from_components(
        comps, transition.next_state.inventory, cfg, reduce=reduce,
        half_spread=transition.quotes.half_spread,
    )


def performative_risk(objective_sum: Tensor, n_steps: int) -> Tensor:
    """Convert a summed objective into performative risk (negated, per-step mean)."""
    return -(objective_sum / max(n_steps, 1))
