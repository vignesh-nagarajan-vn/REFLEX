"""Client order-flow model -- the policy-dependence channel of the environment.

The dealer's quotes change *which* orders arrive and *from whom*; that is exactly
the endogeneity at the heart of performative prediction.  Two streams arrive each
step:

**Uninformed flow** is benign liquidity demand.  Its notional decays with the
quoted half-spread (demand elasticity) and scales with current liquidity.  It is
roughly balanced between buys and sells, so on its own it earns the dealer the
spread with little adverse selection.

**Informed (toxic) flow** is the adversarial component, gated by ``alpha``.
Informed clients observe a noisy signal of the mispricing ``g = fundamental -
mid`` and trade in the direction that picks the dealer off (buying when the
dealer's mid is too low, selling when it is too high).  Crucially they only
bother to trade when their perceived edge exceeds the half-spread they must pay,

    informed_notional = alpha * info_intensity * toxicity_feedback
                        * relu(|signal| - half_spread) * (liquidity / liq_mean).

The ``relu(|signal| - half_spread)`` term is the feedback that makes the whole
study work: **tighter quotes admit proportionally more toxic flow**, and the
*slope* of that relationship scales with ``alpha`` (and the explicit
``toxicity_feedback`` lever).  When the dealer redeploys a tighter policy it
invites more adverse selection, which pushes its best-response spread back out --
the cobweb dynamics whose contraction modulus we measure.

This model is part of the structural ground truth ``T_true`` and is not trained.
Because it is ground truth it may sample freely (it need not be differentiable);
only the learned operator ``T_theta`` must support pathwise gradients.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from ..config import ClientsConfig
from .bonds import BondUniverse

# Small structural constants (not swept -- kept here to avoid config churn).
_ARRIVAL_NOISE = 0.15  # multiplicative noise on uninformed arrival notional
_UNINF_IMBALANCE_STD = 0.10  # idiosyncratic buy/sell imbalance of uninformed flow


@dataclass
class FlowSample:
    """Realised one-step order flow, decomposed for P&L and diagnostics.

    All tensors are shape ``[N]`` and expressed in notional, from the **dealer's**
    perspective:

    * ``dealer_buy`` -- volume the dealer bought (clients hit the bid).
    * ``dealer_sell`` -- volume the dealer sold (clients lifted the ask).
    * ``informed_buy`` / ``informed_sell`` -- the toxic share of each side.
    * ``informed_volume`` -- total informed notional (``informed_buy + informed_sell``).
    * ``gross_volume`` -- total traded notional (``dealer_buy + dealer_sell``).
    """

    dealer_buy: Tensor
    dealer_sell: Tensor
    informed_buy: Tensor
    informed_sell: Tensor
    informed_volume: Tensor
    gross_volume: Tensor


class ClientFlowModel:
    """Samples uninformed + informed client orders in response to quotes."""

    def __init__(
        self, bonds: BondUniverse, cfg: ClientsConfig, liq_mean: float = 1.0
    ) -> None:
        self.bonds = bonds
        self.cfg = cfg
        self.liq_mean = max(float(liq_mean), 1e-6)  # for normalising liquidity

    def sample(
        self,
        half_spread: Tensor,
        skew: Tensor,
        mispricing: Tensor,
        liquidity: Tensor,
        generator: torch.Generator | None = None,
    ) -> FlowSample:
        """Draw one step of client order flow.

        Parameters
        ----------
        half_spread:
            Deployed half-spread ``[N]`` (already positive).
        skew:
            Quote skew ``[N]`` (shifts both quotes; affects which side informed
            flow finds attractive only through the realised fill price, not the
            arrival model -- kept for interface symmetry).
        mispricing:
            True mispricing ``g = fundamental - mid`` ``[N]`` (latent; informed
            traders see only a noisy version).
        liquidity:
            Current latent liquidity ``[N]``.
        generator:
            Optional RNG for reproducibility.

        Returns
        -------
        FlowSample
            Decomposed order flow for this step.
        """
        cfg = self.cfg
        n = self.bonds.n_bonds
        h = half_spread.clamp_min(0.0)
        # liquidity normalised by the field's long-run mean.
        liq_ratio = (liquidity / self.liq_mean).clamp_min(0.0)

        # ---------------- uninformed (benign) flow ------------------------- #
        base = cfg.base_arrival_rate * torch.exp(-cfg.demand_elasticity * h) * liq_ratio
        arrival_noise = 1.0 + _ARRIVAL_NOISE * torch.randn(n, generator=generator)
        uninf_vol = (base * arrival_noise).clamp_min(0.0)
        # Split into buys/sells around 50-50 with a small idiosyncratic imbalance.
        imb = 0.5 + _UNINF_IMBALANCE_STD * torch.randn(n, generator=generator)
        imb = imb.clamp(0.0, 1.0)
        u_buy_from_dealer = uninf_vol * imb  # clients buy -> dealer sells
        u_sell_to_dealer = uninf_vol * (1.0 - imb)  # clients sell -> dealer buys

        # ---------------- informed (toxic) flow ---------------------------- #
        # Informed traders see a noisy signal of the mispricing and lean on the
        # dealer in the picking-off direction.  The toxic notional is split into
        #
        #   * a *baseline* level ``info_base_intensity * gate`` that does not depend
        #     on the half-spread.  It fixes the adverse-selection regime (and hence
        #     the curvature of the dealer's objective, i.e. the response gain
        #     ``kappa``) at a value that is **independent of alpha**; and
        #   * a *spread-responsive* term ``alpha * toxicity_feedback * info_intensity
        #     * relu(spread_ref - h) * gate`` whose slope in the half-spread is what
        #     creates the cobweb.  Its magnitude -- and therefore ``dtau/dh`` -- is
        #     **linear in alpha**.
        #
        # Because the level is alpha-independent but the slope scales with alpha,
        # the empirical contraction modulus ``m = kappa * (dtau/dh)`` scales with
        # alpha and crosses 1 at a tunable ``alpha*`` (set via ``toxicity_feedback``).
        signal = mispricing + cfg.info_signal_noise * torch.randn(n, generator=generator)
        edge = signal.abs()
        edge_scale = max(cfg.info_signal_noise, 1e-6)
        gate = torch.tanh(edge / edge_scale)  # in [0, 1); "is there edge", alpha-free
        # Spread-responsiveness: tighter quotes summon more toxic flow.  Using a
        # smooth, strictly-decreasing factor ``exp(-elasticity * h)`` keeps the
        # feedback active across the *entire* operating range (not only inside a
        # reference spread), so ``dtau/dh < 0`` everywhere and is linear in alpha.
        spread_resp = torch.exp(-cfg.info_spread_decay * h)
        toxic = gate * (
            cfg.info_base_intensity
            + cfg.alpha * cfg.toxicity_feedback * cfg.info_intensity * spread_resp
        )
        inf_vol = (toxic * liq_ratio).clamp_min(0.0)
        # Gentle global saturation as a numerical safety net (rarely binds).
        cap = max(cfg.info_cap, 1e-6)
        inf_vol = cap * torch.tanh(inf_vol / cap)
        direction = torch.sign(signal)  # +1 -> buy from dealer, -1 -> sell to dealer
        informed_buy_from_dealer = inf_vol * (direction > 0).float()  # dealer sells
        informed_sell_to_dealer = inf_vol * (direction < 0).float()  # dealer buys

        # ---------------- aggregate (dealer perspective) ------------------- #
        dealer_sell = u_buy_from_dealer + informed_buy_from_dealer  # lifted asks
        dealer_buy = u_sell_to_dealer + informed_sell_to_dealer  # hit bids
        informed_volume = informed_buy_from_dealer + informed_sell_to_dealer
        gross_volume = dealer_sell + dealer_buy

        return FlowSample(
            dealer_buy=dealer_buy,
            dealer_sell=dealer_sell,
            informed_buy=informed_sell_to_dealer,  # informed share the dealer bought
            informed_sell=informed_buy_from_dealer,  # informed share the dealer sold
            informed_volume=informed_volume,
            gross_volume=gross_volume,
        )
