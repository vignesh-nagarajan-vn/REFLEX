"""The learned, differentiable Market Response Operator ``T_theta``.

``T_theta`` is a surrogate for the structural simulator that the dealer can
optimise against with pathwise gradients.  Given the **observable** state, the
dealer's **quotes** for the step, and a global **policy summary**, it predicts a
distribution over

    [d_inventory, d_mid, d_flow, d_vol,                 (next-state deltas)
     spread_capture, inventory_pnl, adverse_selection_loss]   (P&L channels)

so ``D = len(DELTA_KEYS) + len(PNL_KEYS) = 7``.  Predicting the P&L channels'
*conditional means* (especially the adverse-selection loss, whose mean given the
observables is the toxicity the dealer is paying) lets the dealer feel the cost
of adverse selection without the operator needing to model the latent
fundamental.

The crucial modelling choice for the convergence study lives in how the
``policy_summary`` is used.  Within a single deployment the policy summary is
(approximately) constant, so a freshly-refit operator cannot learn the
*derivative* of the induced distribution with respect to the policy -- it has
only one policy's worth of data.  When the dealer later optimises a new policy
against the operator the summary is **frozen** at the deployed policy's value
(see :meth:`rollout`'s ``policy_summary_override``), so the dealer's quotes still
move the mechanical fill/spread channels but the toxicity *level* stays anchored
to the old regime.  That gap -- the dealer not anticipating that re-tightening
will summon more toxic flow -- is the performative shift whose contraction
modulus the experiment measures.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.distributions as dist
from torch import Tensor, nn

from ..config import Config
from ..objective.reward import RewardBreakdown, reward_from_components
from ..types import (
    DELTA_KEYS,
    PNL_KEYS,
    POLICY_SUMMARY_DIM,
    MarketState,
    Quotes,
    policy_summary,
)
from .heads import build_head

# Per-bond operator input = observable feats (3) + quotes (2) + policy summary (3).
N_INPUT_FEATURES = 3 + 2 + POLICY_SUMMARY_DIM
TARGET_KEYS: List[str] = DELTA_KEYS + PNL_KEYS
OUT_DIM = len(TARGET_KEYS)


@dataclass
class OperatorRollout:
    """Result of a differentiable rollout under ``T_theta``.

    Attributes
    ----------
    objective:
        Scalar total dealer objective summed over the horizon (to be *maximised*;
        the policy optimiser minimises its negative).
    step_objectives:
        Per-step objective tensors (length ``horizon``).
    states:
        Predicted observable states along the rollout (length ``horizon + 1``).
    pnl_components:
        Per-step dict of predicted P&L channels.
    """

    objective: Tensor
    step_objectives: List[Tensor]
    states: List[MarketState]
    pnl_components: List[Dict[str, Tensor]]
    logprob_sum: Optional[Tensor] = None  # populated only in score-function mode


class MarketResponseOperator(nn.Module):
    """Differentiable surrogate dynamics ``T_theta``."""

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        op = cfg.operator
        hidden = int(op.hidden)
        self.encoder = nn.Sequential(
            nn.Linear(N_INPUT_FEATURES, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.head = build_head(
            op.head_type,
            in_dim=hidden,
            out_dim=OUT_DIM,
            n_mixture=op.n_mixture,
            min_logstd=op.min_logstd,
            max_logstd=op.max_logstd,
        )
        # Target normalizer (identity until fit_operator sets it). The head models
        # standardized targets so the very different channel scales (e.g. mid
        # deltas vs spread capture) are balanced; rsample/log_prob convert back.
        self.register_buffer("target_mean", torch.zeros(OUT_DIM))
        self.register_buffer("target_std", torch.ones(OUT_DIM))

    @torch.no_grad()
    def set_normalizer(self, mean: Tensor, std: Tensor) -> None:
        """Store per-channel target mean/std used to standardize the head's target."""
        self.target_mean.copy_(mean.reshape(-1))
        self.target_std.copy_(std.reshape(-1).clamp_min(1e-6))

    # ------------------------------------------------------------------ #
    # Feature construction                                                #
    # ------------------------------------------------------------------ #
    @staticmethod
    def build_features(
        state: MarketState, quotes: Quotes, summary: Tensor
    ) -> Tensor:
        """Assemble the per-bond input matrix ``[N, N_INPUT_FEATURES]``.

        ``summary`` is the length-3 policy summary, broadcast across bonds.
        """
        obs = state.observable_features()  # [N, 3]
        quote_feats = torch.stack([quotes.half_spread, quotes.skew], dim=-1)  # [N, 2]
        n = obs.shape[0]
        summ = summary.reshape(1, -1).expand(n, -1)  # [N, 3]
        return torch.cat([obs, quote_feats, summ], dim=-1)

    def forward(self, features: Tensor) -> dist.Distribution:
        """Return the predictive distribution over the target given features."""
        return self.head(self.encoder(features))

    def predict(
        self, state: MarketState, quotes: Quotes, summary: Tensor
    ) -> dist.Distribution:
        """Predictive distribution for a state/quotes/summary triple."""
        return self.forward(self.build_features(state, quotes, summary))

    def rsample(
        self,
        features: Tensor,
        generator: Optional[torch.Generator] = None,
    ) -> Tensor:
        """Reparameterised sample ``[N, OUT_DIM]`` in real units (generator-controlled)."""
        hidden = self.encoder(features)
        sample_norm = self.head.rsample(hidden, generator=generator)
        return sample_norm * self.target_std + self.target_mean

    def log_prob(
        self,
        features: Tensor,
        target: Tensor,
    ) -> Tensor:
        """Per-row log-likelihood ``[N]`` of ``target`` under the prediction.

        The target is standardized with the stored normalizer before scoring; the
        constant Jacobian term ``-sum(log std)`` is dropped (it is identical
        across operators sharing a normalizer, so trained-vs-baseline NLL
        comparisons are unaffected).
        """
        target_norm = (target - self.target_mean) / self.target_std
        return self.forward(features).log_prob(target_norm)

    # ------------------------------------------------------------------ #
    # Learned distribution response (the v3 un-blinding readout)           #
    # ------------------------------------------------------------------ #
    def distribution_response(
        self, state: MarketState, quotes: Quotes, summary: Tensor
    ) -> Tensor:
        """Learned sensitivity of every predicted channel to the deployed spread.

        Returns a detached length-``OUT_DIM`` tensor whose ``j``-th entry is

            d E_hat[channel_j] / d summary_h ,

        i.e. the autograd derivative of the operator's predicted per-bond mean
        (averaged across bonds, in real units) with respect to the *policy
        summary's central-half-spread component*.  This is the operator's
        learned counterpart of the theory's distribution response: for the
        adverse-selection channel the closed form is ``d(psi*tau)/dh =
        -psi*epsilon(h)`` (theory 1.1/1.2), so a *windowed-fit* operator should
        report a negative slope of comparable magnitude -- and an operator fit
        on a single deployment (constant summary) has no signal to learn it
        from.  This readout is the ML<->math seam of the package.
        """
        summ = summary.detach().clone().requires_grad_(True)
        feats = self.build_features(
            state.detach(),
            Quotes(half_spread=quotes.half_spread.detach(), skew=quotes.skew.detach()),
            summ,
        )
        mean_norm = self.forward(feats).mean  # [N, OUT_DIM] (standardized units)
        mean_real = mean_norm * self.target_std + self.target_mean
        per_channel = mean_real.mean(dim=0)  # [OUT_DIM]
        grads = []
        for j in range(per_channel.shape[0]):
            (g,) = torch.autograd.grad(
                per_channel[j], summ, retain_graph=(j < per_channel.shape[0] - 1)
            )
            grads.append(g[0])  # component 0 of the summary = mean half-spread
        return torch.stack(grads).detach()

    def toxic_slope(
        self, state: MarketState, quotes: Quotes, summary: Tensor
    ) -> float:
        """Learned ``d E_hat[adverse_selection_loss] / d h`` (expected negative).

        Convenience view of :meth:`distribution_response` restricted to the
        adverse-selection channel.  Divide by the adverse severity ``psi``
        (theory 1.1 §2.5) to convert into a learned ``epsilon_hat(h)``.
        """
        adverse_idx = len(DELTA_KEYS) + PNL_KEYS.index("adverse_selection_loss")
        return float(self.distribution_response(state, quotes, summary)[adverse_idx])

    # ------------------------------------------------------------------ #
    # Differentiable rollout (the object policy optimisation backprops through)
    # ------------------------------------------------------------------ #
    def rollout(
        self,
        init_state: MarketState,
        policy,
        horizon: int,
        reward_cfg,
        policy_summary_override: Optional[Tensor] = None,
        generator: Optional[torch.Generator] = None,
        score_function: bool = False,
    ) -> OperatorRollout:
        """Roll the policy forward under ``T_theta`` for ``horizon`` steps.

        Parameters
        ----------
        init_state:
            Starting observable state (latent fields may be placeholders -- they
            are never read).
        policy:
            The dealer policy whose quotes drive the rollout (differentiable).
        horizon:
            Number of steps.
        reward_cfg:
            Reward configuration for assembling the per-step objective.
        policy_summary_override:
            If given, this frozen length-3 vector is used as the policy summary at
            **every** step (the Repeated-Risk-Minimization convention: optimise
            against the deployed regime).  If ``None`` the summary is recomputed
            from the current policy each step (used for diagnostics).
        generator:
            Optional RNG for reproducible reparameterised draws.
        score_function:
            If ``True`` use the score-function (REINFORCE) estimator: the operator
            outputs are sampled *without* reparameterisation and the summed
            log-probability (differentiable w.r.t. the policy through the
            predicted distribution) is returned in ``logprob_sum``.  The default
            (``False``) uses pathwise/reparameterised gradients.

        Returns
        -------
        OperatorRollout
        """
        state = init_state
        step_objs: List[Tensor] = []
        pnl_list: List[Dict[str, Tensor]] = []
        states: List[MarketState] = [state]
        total = state.inventory.new_zeros(())
        logprob_sum = state.inventory.new_zeros(()) if score_function else None

        for _ in range(int(horizon)):
            quotes = policy.quote(state)
            if policy_summary_override is not None:
                summ = policy_summary_override
            else:
                summ = policy_summary(state, policy)

            feats = self.build_features(state, quotes, summ)

            if score_function:
                d = self.forward(feats)  # distribution over standardized target
                sample_norm = d.sample()  # detached (no reparameterisation)
                logprob_sum = logprob_sum + d.log_prob(sample_norm).sum()
                sample = sample_norm * self.target_std + self.target_mean
            else:
                sample = self.rsample(feats, generator=generator)  # [N, OUT_DIM]

            deltas = sample[:, : len(DELTA_KEYS)]
            pnl = sample[:, len(DELTA_KEYS) :]

            inv_next = state.inventory + deltas[:, 0]
            mid_next = state.mid + deltas[:, 1]
            flow_next = state.flow_recent + deltas[:, 2]
            vol_next = (state.vol_recent + deltas[:, 3]).clamp_min(0.0)

            pnl_components = {
                "spread_capture": pnl[:, 0],
                "inventory_pnl": pnl[:, 1],
                "adverse_selection_loss": pnl[:, 2],
            }
            rb: RewardBreakdown = reward_from_components(
                pnl_components, inv_next, reward_cfg, reduce=True,
                half_spread=quotes.half_spread,
            )
            step_objs.append(rb.objective)
            pnl_list.append(pnl_components)
            total = total + rb.objective

            state = MarketState(
                inventory=inv_next,
                mid=mid_next,
                fundamental=state.fundamental,  # placeholder, unused
                liquidity=state.liquidity,  # placeholder, unused
                flow_recent=flow_next,
                vol_recent=vol_next,
                t=state.t + 1,
            )
            states.append(state)

        return OperatorRollout(
            objective=total,
            step_objectives=step_objs,
            states=states,
            pnl_components=pnl_list,
            logprob_sum=logprob_sum,
        )
