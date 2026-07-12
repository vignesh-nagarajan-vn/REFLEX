"""Structural ground-truth simulator ``T_true``.

This is the *real* market.  It is parameterised (through its sub-models) by the
adversariality knob ``alpha`` but is otherwise fixed -- it is never trained.  A
single :meth:`StructuralSimulator.step` does, in order:

1. draw client orders conditional on the dealer's quotes (:class:`ClientFlowModel`);
2. compute fills and update inventory;
3. evolve the latent fundamental as a (correlated) random walk;
4. push the mid toward the fundamental plus the price impact of net flow;
5. evolve the latent liquidity field;
6. compute the three P&L components.

P&L accounting (exact -- the three components sum to total economic P&L)
----------------------------------------------------------------------
With ``S`` = dealer sell volume (clients lift the ask at ``mid + skew + h``),
``B`` = dealer buy volume (clients hit the bid at ``mid + skew - h``),
``q_after = q_before + B - S`` the post-trade inventory, ``g = v - m`` the
mispricing and ``v' `` next-step fundamental:

* ``spread_capture        = S * (h + skew) + B * (h - skew)``
* ``inventory_pnl         = q_after * (v' - v)``        (carry / inventory risk)
* ``adverse_selection_loss = (B - S) * (m - v)``         (positive in expectation)

and one can verify algebraically that

    spread_capture + inventory_pnl - adverse_selection_loss
        == delta_cash + (q_after * v' - q_before * v),

i.e. the realised cash flow plus the mark-to-fundamental change in inventory
value.  The dealer reward (``objective/reward.py``) is this total minus a
quadratic inventory-risk penalty.
"""

from __future__ import annotations

import torch
from torch import Tensor

from ..config import Config
from ..types import Fills, MarketState, Quotes, Transition
from .bonds import BondUniverse
from .clients import ClientFlowModel
from .liquidity_field import LiquidityField


class StructuralSimulator:
    """Ground-truth market dynamics ``T_true``.

    Parameters
    ----------
    cfg:
        Full project config (uses the ``bonds``, ``clients`` and ``simulator``
        sections).
    bonds:
        Optional pre-built universe; if ``None`` one is constructed from ``cfg``
        using ``cfg.seed`` so the universe is reproducible.
    """

    def __init__(self, cfg: Config, bonds: BondUniverse | None = None) -> None:
        self.cfg = cfg
        self.sim_cfg = cfg.simulator
        self.bonds = bonds if bonds is not None else BondUniverse(cfg.bonds, seed=cfg.seed)
        self.clients = ClientFlowModel(
            self.bonds, cfg.clients, liq_mean=self.sim_cfg.liq_mean
        )
        self.liquidity = LiquidityField(self.bonds, self.sim_cfg)

    # ------------------------------------------------------------------ #
    # Episode lifecycle                                                   #
    # ------------------------------------------------------------------ #
    def reset(self, seed: int | None = None) -> MarketState:
        """Sample a fresh initial market state.

        Parameters
        ----------
        seed:
            Optional seed for the initial draw (fundamental level, initial
            mispricing).  Pass distinct seeds to get distinct episodes.

        Returns
        -------
        MarketState
            Initial state with zero inventory and a small random mid/fundamental
            gap.
        """
        gen = None
        if seed is not None:
            gen = torch.Generator().manual_seed(int(seed))
        n = self.bonds.n_bonds

        fundamental = 100.0 + self.bonds.correlated_normal(gen)  # latent true value
        gap = self.sim_cfg.init_mispricing_vol * torch.randn(n, generator=gen)
        mid = fundamental + gap
        liquidity = self.liquidity.init_state()

        return MarketState(
            inventory=torch.zeros(n),
            mid=mid,
            fundamental=fundamental,
            liquidity=liquidity,
            flow_recent=torch.zeros(n),
            vol_recent=torch.zeros(n),
            t=0,
        )

    def step(
        self,
        state: MarketState,
        quotes: Quotes,
        generator: torch.Generator | None = None,
    ) -> Transition:
        """Advance the market one step under the dealer's quotes.

        Parameters
        ----------
        state:
            Current market state.
        quotes:
            Dealer quotes (half-spread + skew) for this step.
        generator:
            Optional RNG; pass one for a reproducible rollout.

        Returns
        -------
        Transition
            ``(state, quotes, fills, next_state, pnl_components)`` where
            ``pnl_components`` is a dict with keys ``spread_capture``,
            ``inventory_pnl`` and ``adverse_selection_loss``.
        """
        sim = self.sim_cfg
        h = quotes.half_spread.clamp_min(0.0)
        skew = quotes.skew
        v = state.fundamental
        m = state.mid
        q0 = state.inventory
        mispricing = v - m  # g

        # --- 1. client orders given quotes -------------------------------- #
        flow = self.clients.sample(
            half_spread=h,
            skew=skew,
            mispricing=mispricing,
            liquidity=state.liquidity,
            generator=generator,
        )
        S = flow.dealer_sell  # dealer sold (clients bought at ask)
        B = flow.dealer_buy  # dealer bought (clients sold at bid)

        # --- 2. fills and inventory update -------------------------------- #
        d_inv = B - S  # signed dealer inventory change
        q_after = q0 + d_inv
        # Volume-weighted average fill price per bond (for the Fills record).
        bid = m + skew - h
        ask = m + skew + h
        denom = (S + B).clamp_min(1e-8)
        avg_price = (S * ask + B * bid) / denom

        fills = Fills(
            qty=d_inv,
            price=avg_price,
            gross_volume=flow.gross_volume,
            informed_volume=flow.informed_volume,
            was_informed=flow.informed_volume > 0,
        )

        # --- 3. evolve fundamental (correlated random walk) --------------- #
        shock = self.bonds.correlated_normal(generator)
        v_next = v + sim.fundamental_vol * shock

        # --- 4. evolve mid: price discovery + impact of net flow ---------- #
        # Net client buying pressure (clients bought S, sold B) pushes the mid up.
        net_client_buy = S - B
        # Price discovery pulls the mid toward the fundamental.
        reversion = sim.mid_reversion * mispricing
        impact = sim.impact * net_client_buy
        noise = sim.mid_noise * torch.randn(self.bonds.n_bonds, generator=generator)
        # Cap the per-step move so an extreme-flow step cannot make the mid run away.
        delta_m = (reversion + impact + noise).clamp(-sim.mid_move_cap, sim.mid_move_cap)
        m_next = m + delta_m

        # --- 5. evolve liquidity field ------------------------------------ #
        liq_next = self.liquidity.step(
            liquidity=state.liquidity,
            half_spread=h,
            gross_flow=flow.gross_volume,
            spread_ref=self.cfg.clients.spread_ref,
            generator=generator,
        )

        # --- 6. P&L components (exact decomposition) ---------------------- #
        spread_capture = S * (h + skew) + B * (h - skew)
        inventory_pnl = q_after * (v_next - v)
        adverse_selection_loss = d_inv * (m - v)  # (B - S)(m - v)

        pnl_components = {
            "spread_capture": spread_capture,
            "inventory_pnl": inventory_pnl,
            "adverse_selection_loss": adverse_selection_loss,
        }

        next_state = MarketState(
            inventory=q_after,
            mid=m_next,
            fundamental=v_next,
            liquidity=liq_next,
            flow_recent=net_client_buy,  # observable signed flow (S - B)
            vol_recent=flow.gross_volume,  # observable gross volume (liquidity proxy)
            t=state.t + 1,
        )

        return Transition(
            state=state,
            quotes=quotes,
            fills=fills,
            next_state=next_state,
            pnl_components=pnl_components,
        )
