"""Latent liquidity field with cross-bond coupling.

Liquidity is a hidden state: the dealer never observes it directly, only its
effects on how much flow shows up.  The dynamics encode the central source of
*fragility* in the environment:

* **Transient boost.** Tight quotes attract extra flow, which temporarily makes
  a bond look more liquid (``liq_flow_boost``).
* **Persistent degradation.** Quoting *too* tight for too long (relative to the
  reference half-spread) erodes the underlying liquidity (``liq_overtighten_decay``)
  -- market makers pull back, the bond gets picked off, depth thins.  This is the
  slow variable that punishes a policy which collapses its spreads.
* **Mean reversion + coupling.** Each bond's liquidity reverts toward
  ``liq_mean`` and is nudged by its neighbours through the universe correlation
  matrix (``liq_coupling``), so liquidity shocks propagate across the portfolio.

The field is part of the structural ground truth ``T_true`` and is *not* trained.
"""

from __future__ import annotations

import torch
from torch import Tensor

from ..config import SimulatorConfig
from .bonds import BondUniverse


class LiquidityField:
    """Mean-reverting latent liquidity process coupled across bonds."""

    def __init__(self, bonds: BondUniverse, cfg: SimulatorConfig) -> None:
        self.bonds = bonds
        self.cfg = cfg
        # Row-normalised coupling matrix (off-diagonal correlations only) so the
        # neighbour term is a weighted average of *other* bonds' liquidity.
        corr = bonds.corr.clone()
        corr.fill_diagonal_(0.0)
        row_sum = corr.abs().sum(dim=1, keepdim=True).clamp_min(1e-6)
        self._coupling: Tensor = corr / row_sum

    def init_state(self) -> Tensor:
        """Initial liquidity: every bond starts at the long-run mean."""
        return torch.full((self.bonds.n_bonds,), float(self.cfg.liq_mean))

    def step(
        self,
        liquidity: Tensor,
        half_spread: Tensor,
        gross_flow: Tensor,
        spread_ref: float,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """Advance the liquidity field one step.

        Parameters
        ----------
        liquidity:
            Current liquidity ``[N]`` (strictly positive).
        half_spread:
            Deployed half-spread ``[N]`` for this step.
        gross_flow:
            Total (unsigned) traded notional ``[N]`` this step -- the realised
            attention/flow the quotes attracted.
        spread_ref:
            Reference half-spread; quoting below it counts as "tight".
        generator:
            Optional RNG for the idiosyncratic shock.

        Returns
        -------
        Tensor
            Next-step liquidity ``[N]`` (clamped to stay positive).
        """
        cfg = self.cfg
        # Mean reversion toward the long-run level.
        reversion = cfg.liq_reversion * (cfg.liq_mean - liquidity)

        # Cross-bond coupling: pull toward the (correlation-weighted) average
        # liquidity of neighbours -- shocks propagate through the portfolio.
        neighbour = self._coupling @ liquidity
        coupling = cfg.liq_coupling * (neighbour - liquidity)

        # Transient boost: realised flow temporarily lifts apparent liquidity.
        boost = cfg.liq_flow_boost * gross_flow

        # Persistent degradation: how far *below* the reference we are quoting.
        tightness = torch.relu(spread_ref - half_spread)
        degradation = cfg.liq_overtighten_decay * tightness

        noise = cfg.liq_vol * torch.randn(
            self.bonds.n_bonds, generator=generator
        )

        nxt = liquidity + reversion + coupling + boost - degradation + noise
        return nxt.clamp_min(0.05)
