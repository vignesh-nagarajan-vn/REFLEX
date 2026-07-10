"""Genuine ``N``-dealer market environment (math-theory 1.3, simulated).

``N`` symmetric dealers quote the same bond universe and share **one informed
pool**: an informed trader routes to the dealer whose quotes are easiest to
pick off, with cross-dealer spillover ``kappa``.  Dealer ``i``'s toxic
spread-responsiveness mirrors the coupled form of the 1.3 derivation exactly:

    spread_resp_i = (1 - kappa) * exp(-c_t * h_i)
                    + kappa * mean_j exp(-c_t * h_j)                (1.3 §2.1)

so ``kappa = 0`` decouples the dealers and ``kappa = 1`` makes toxicity a pure
common-pool quantity.  Uninformed (relationship) flow stays dealer-specific
with the usual GLFT demand curve in the dealer's own spread.

**Exact single-dealer reduction.**  At ``N = 1`` the coupled factor collapses
to ``exp(-c_t*h)`` and this simulator draws its random variates in *exactly*
the same order as :class:`~reflex.env.simulator.StructuralSimulator` +
:class:`~reflex.env.clients.ClientFlowModel`, so given the same seeds it
reproduces the single-dealer market **bit for bit** -- asserted in
``tests/test_multi_dealer_env.py``.  That equivalence anchors the multi-dealer
extension to everything already validated single-dealer.

Shared market state (mid, fundamental, liquidity) evolves under the *combined*
flow of all dealers: price impact from total net client flow, liquidity boosted
by total gross flow and degraded by the *tightest* dealer's over-quoting.

**Gotcha (documented, real).**  Because the liquidity boost is driven by the
*total* gross flow of all ``N`` dealers, multi-dealer runs inflate the shared
liquidity ratio relative to single-dealer runs at the same config; that can
push per-dealer informed volume into the ``info_cap`` saturation, compressing
cross-dealer toxicity differences.  For flow-allocation studies either scale
``liq_flow_boost`` down by ``1/N`` or raise ``info_cap`` -- and check the cap
is slack before interpreting toxic shares.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
from torch import Tensor

from ..config import Config
from ..types import MarketState, Quotes
from .bonds import BondUniverse
from .clients import _ARRIVAL_NOISE, _UNINF_IMBALANCE_STD
from .liquidity_field import LiquidityField


@dataclass
class MultiMarketState:
    """Shared market snapshot plus per-dealer inventories.

    ``inventories`` is ``[N_dealers, n_bonds]``; the shared fields match
    :class:`~reflex.types.MarketState`.
    """

    inventories: Tensor
    mid: Tensor
    fundamental: Tensor  # latent
    liquidity: Tensor  # latent
    flow_recent: Tensor  # shared observable: total signed client flow
    vol_recent: Tensor  # shared observable: total gross volume
    t: int = 0

    @property
    def n_dealers(self) -> int:
        return int(self.inventories.shape[0])

    @property
    def n_bonds(self) -> int:
        return int(self.inventories.shape[-1])

    def dealer_state(self, i: int) -> MarketState:
        """Dealer ``i``'s view as a single-dealer :class:`MarketState`."""
        return MarketState(
            inventory=self.inventories[i],
            mid=self.mid,
            fundamental=self.fundamental,
            liquidity=self.liquidity,
            flow_recent=self.flow_recent,
            vol_recent=self.vol_recent,
            t=self.t,
        )

    def detach(self) -> "MultiMarketState":
        return MultiMarketState(
            inventories=self.inventories.detach(),
            mid=self.mid.detach(),
            fundamental=self.fundamental.detach(),
            liquidity=self.liquidity.detach(),
            flow_recent=self.flow_recent.detach(),
            vol_recent=self.vol_recent.detach(),
            t=self.t,
        )


@dataclass
class DealerStepResult:
    """One dealer's realised flow and P&L for a step."""

    dealer_sell: Tensor  # volume sold to clients (asks lifted)
    dealer_buy: Tensor  # volume bought from clients (bids hit)
    informed_volume: Tensor
    gross_volume: Tensor
    pnl_components: Dict[str, Tensor]  # spread_capture / inventory_pnl / adverse_selection_loss


@dataclass
class MultiTransition:
    """One step of the ``N``-dealer market."""

    state: MultiMarketState
    quotes: List[Quotes]
    dealers: List[DealerStepResult]
    next_state: MultiMarketState


class MultiDealerSimulator:
    """Ground-truth ``N``-dealer market sharing one informed pool."""

    def __init__(self, cfg: Config, bonds: Optional[BondUniverse] = None) -> None:
        self.cfg = cfg
        self.sim_cfg = cfg.simulator
        self.n_dealers = max(int(cfg.clients.n_dealers), 1)
        self.kappa = float(cfg.clients.toxic_spillover)
        if not 0.0 <= self.kappa <= 1.0:
            raise ValueError(f"toxic_spillover must be in [0, 1], got {self.kappa}")
        self.bonds = bonds if bonds is not None else BondUniverse(cfg.bonds, seed=cfg.seed)
        self.liquidity = LiquidityField(self.bonds, self.sim_cfg)
        self.liq_mean = max(float(self.sim_cfg.liq_mean), 1e-6)

    # ------------------------------------------------------------------ #
    # Episode lifecycle (mirrors StructuralSimulator.reset draw order)     #
    # ------------------------------------------------------------------ #
    def reset(self, seed: Optional[int] = None) -> MultiMarketState:
        gen = None
        if seed is not None:
            gen = torch.Generator().manual_seed(int(seed))
        n = self.bonds.n_bonds

        fundamental = 100.0 + self.bonds.correlated_normal(gen)
        gap = self.sim_cfg.init_mispricing_vol * torch.randn(n, generator=gen)
        mid = fundamental + gap
        liquidity = self.liquidity.init_state()

        return MultiMarketState(
            inventories=torch.zeros(self.n_dealers, n),
            mid=mid,
            fundamental=fundamental,
            liquidity=liquidity,
            flow_recent=torch.zeros(n),
            vol_recent=torch.zeros(n),
            t=0,
        )

    # ------------------------------------------------------------------ #
    # One step                                                            #
    # ------------------------------------------------------------------ #
    def step(
        self,
        state: MultiMarketState,
        quotes: List[Quotes],
        generator: Optional[torch.Generator] = None,
    ) -> MultiTransition:
        """Advance the shared market one step under all dealers' quotes.

        Random draws occur in dealer order (3 per dealer: arrival noise,
        imbalance, informed signal) followed by the market draws (fundamental
        shock, mid noise, liquidity noise) -- at ``N = 1`` this is exactly the
        single-dealer draw order.
        """
        if len(quotes) != self.n_dealers:
            raise ValueError(f"expected {self.n_dealers} quote sets, got {len(quotes)}")
        c = self.cfg.clients
        sim = self.sim_cfg
        n = self.bonds.n_bonds
        v = state.fundamental
        m = state.mid
        mispricing = v - m
        liq_ratio = (state.liquidity / self.liq_mean).clamp_min(0.0)

        hs = [q.half_spread.clamp_min(0.0) for q in quotes]
        # The coupled toxic responsiveness (1.3 §2.1): own + spillover pool.
        own_resp = [torch.exp(-c.info_spread_decay * h) for h in hs]
        pool_resp = torch.stack(own_resp, dim=0).mean(dim=0)

        dealers: List[DealerStepResult] = []
        S_total = torch.zeros(n)
        B_total = torch.zeros(n)
        for i in range(self.n_dealers):
            h = hs[i]
            skew = quotes[i].skew

            # ---- uninformed (dealer-specific relationship flow) ---------- #
            base = c.base_arrival_rate * torch.exp(-c.demand_elasticity * h) * liq_ratio
            arrival_noise = 1.0 + _ARRIVAL_NOISE * torch.randn(n, generator=generator)
            uninf_vol = (base * arrival_noise).clamp_min(0.0)
            imb = 0.5 + _UNINF_IMBALANCE_STD * torch.randn(n, generator=generator)
            imb = imb.clamp(0.0, 1.0)
            u_buy_from_dealer = uninf_vol * imb
            u_sell_to_dealer = uninf_vol * (1.0 - imb)

            # ---- informed (shared pool with spillover) -------------------- #
            signal = mispricing + c.info_signal_noise * torch.randn(n, generator=generator)
            edge_scale = max(c.info_signal_noise, 1e-6)
            gate = torch.tanh(signal.abs() / edge_scale)
            spread_resp = (1.0 - self.kappa) * own_resp[i] + self.kappa * pool_resp
            toxic = gate * (
                c.info_base_intensity
                + c.alpha * c.toxicity_feedback * c.info_intensity * spread_resp
            )
            inf_vol = (toxic * liq_ratio).clamp_min(0.0)
            cap = max(c.info_cap, 1e-6)
            inf_vol = cap * torch.tanh(inf_vol / cap)
            direction = torch.sign(signal)
            informed_buy_from_dealer = inf_vol * (direction > 0).float()
            informed_sell_to_dealer = inf_vol * (direction < 0).float()

            S_i = u_buy_from_dealer + informed_buy_from_dealer  # dealer sells
            B_i = u_sell_to_dealer + informed_sell_to_dealer  # dealer buys
            S_total = S_total + S_i
            B_total = B_total + B_i

            q0 = state.inventories[i]
            q_after = q0 + (B_i - S_i)
            spread_capture = S_i * (h + skew) + B_i * (h - skew)
            adverse = (B_i - S_i) * (m - v)
            dealers.append(
                DealerStepResult(
                    dealer_sell=S_i,
                    dealer_buy=B_i,
                    informed_volume=informed_buy_from_dealer + informed_sell_to_dealer,
                    gross_volume=S_i + B_i,
                    pnl_components={
                        "spread_capture": spread_capture,
                        # inventory_pnl needs v_next; filled in below.
                        "inventory_pnl": torch.zeros(n),
                        "adverse_selection_loss": adverse,
                    },
                )
            )

        # ---- shared market evolution (single-dealer draw order) ---------- #
        shock = self.bonds.correlated_normal(generator)
        v_next = v + sim.fundamental_vol * shock

        net_client_buy = S_total - B_total
        reversion = sim.mid_reversion * mispricing
        impact = sim.impact * net_client_buy
        noise = sim.mid_noise * torch.randn(n, generator=generator)
        delta_m = (reversion + impact + noise).clamp(-sim.mid_move_cap, sim.mid_move_cap)
        m_next = m + delta_m

        gross_total = S_total + B_total
        h_tightest = torch.stack(hs, dim=0).min(dim=0).values
        liq_next = self.liquidity.step(
            liquidity=state.liquidity,
            half_spread=h_tightest,
            gross_flow=gross_total,
            spread_ref=self.cfg.clients.spread_ref,
            generator=generator,
        )

        # Complete the per-dealer inventory P&L now that v' exists.
        inventories_next = state.inventories.clone()
        for i, d in enumerate(dealers):
            q_after = state.inventories[i] + (d.dealer_buy - d.dealer_sell)
            inventories_next[i] = q_after
            d.pnl_components["inventory_pnl"] = q_after * (v_next - v)

        next_state = MultiMarketState(
            inventories=inventories_next,
            mid=m_next,
            fundamental=v_next,
            liquidity=liq_next,
            flow_recent=net_client_buy,
            vol_recent=gross_total,
            t=state.t + 1,
        )
        return MultiTransition(
            state=state, quotes=list(quotes), dealers=dealers, next_state=next_state
        )
