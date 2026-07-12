"""Triangulate ``epsilon`` three independent ways against the closed form.

The evidentiary bar of methodology 1.1: the analytic boundary is only a
contribution if independent measurement instruments agree on the quantity it
predicts.  This module runs, at a common reference spread and with common
random numbers,

1. the **BR-slope** probe (:mod:`.br_slope`) -- measures the cobweb modulus
   ``m = epsilon*beta/gamma`` and converts to ``epsilon`` via the closed-form
   ``gamma/beta`` (decision-space leg);
2. the **Sinkhorn/Wasserstein** probe (:mod:`.sinkhorn`) -- the Wasserstein
   rate of the induced toxic-flow distribution (distribution-space leg); and
3. the **CKS flow-curve** probe (:mod:`.cks`) -- the derivative of the fitted
   informed-arrival curve (structural-fit leg);

and reports them next to the analytic ``epsilon(h_ref)`` of theory 1.1.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from ..config import Config
from ..env.simulator import StructuralSimulator
from ..theory.analytic_boundary import (
    ReferenceState,
    beta as beta_of,
    epsilon as epsilon_of,
    gamma as gamma_of,
    reference_state,
)
from ..types import Quotes
from .br_slope import _deployed_policy_at, measure_response_modulus
from .cks import CKSEpsilonResult, estimate_epsilon_cks
from .sinkhorn import SinkhornEpsilonResult, estimate_epsilon_sinkhorn


def realized_reference(
    cfg: Config,
    simulator: StructuralSimulator,
    h_ref: float,
    seed: int = 0,
    n_episodes: int = 8,
) -> ReferenceState:
    """The reference state realised *under the deployment*, not assumed.

    Theory 1.1 evaluates its constants at a frozen reference state (A2:
    liquidity at its long-run mean, one-sigma mispricing).  The probes run on
    the realised state distribution, whose liquidity and mispricing drift with
    the deployment (tight quoting boosts the liquidity field and lets
    mispricings build), so the a-priori closed form can understate the
    realised flow sensitivity several-fold.  Following the 1.1 §9 protocol,
    this measures the mean |mispricing| and mean liquidity ratio over the same
    episode seeds the probes use and rebuilds the reference state there.  The
    *remaining* gap between the realised-state closed form and the measured
    legs is the state-feedback channel (d state/d h) that any frozen-state
    closed form necessarily omits.
    """
    policy = _deployed_policy_at(cfg, float(h_ref))
    gen = torch.Generator().manual_seed(seed + 424242)
    horizon = int(cfg.simulator.horizon)
    liq_mean = max(float(cfg.simulator.liq_mean), 1e-6)
    g_sum = 0.0
    rho_sum = 0.0
    n = 0
    with torch.no_grad():
        for ep in range(int(n_episodes)):
            state = simulator.reset(seed=seed + 71 * ep)
            for _ in range(horizon):
                g_sum += float((state.fundamental - state.mid).abs().mean())
                rho_sum += float(state.liquidity.mean()) / liq_mean
                n += 1
                q = policy.quote(state)
                tr = simulator.step(state, Quotes(q.half_spread, q.skew), generator=gen)
                state = tr.next_state
    n = max(n, 1)
    return reference_state(cfg, mispricing=g_sum / n, liquidity_ratio=rho_sum / n)


@dataclass
class TriangulationResult:
    """The three measured ``epsilon`` estimates next to the closed forms.

    Two analytic values are reported: ``epsilon_analytic`` at the a-priori
    reference state (theory A2) and ``epsilon_analytic_realized`` at the
    state actually realised under the ``h_ref`` deployment (theory 1.1 §9) --
    the latter is the apples-to-apples comparison for the measured legs.
    """

    h_ref: float
    epsilon_analytic: float  # a-priori reference state (A2)
    epsilon_analytic_realized: float  # realised-state closed form (§9)
    epsilon_br: float  # m_hat * gamma_realized / beta (BR-slope leg)
    epsilon_sinkhorn: float
    epsilon_cks: float
    modulus_br: float  # the raw BR-slope modulus m_hat
    gamma: float  # a-priori
    gamma_realized: float
    beta: float
    realized_mispricing: float
    realized_rho: float
    sinkhorn: SinkhornEpsilonResult
    cks: CKSEpsilonResult

    def as_dict(self) -> dict:
        return {
            "h_ref": self.h_ref,
            "epsilon_analytic": self.epsilon_analytic,
            "epsilon_analytic_realized": self.epsilon_analytic_realized,
            "epsilon_br": self.epsilon_br,
            "epsilon_sinkhorn": self.epsilon_sinkhorn,
            "epsilon_cks": self.epsilon_cks,
            "modulus_br": self.modulus_br,
            "gamma": self.gamma,
            "gamma_realized": self.gamma_realized,
            "beta": self.beta,
            "realized_mispricing": self.realized_mispricing,
            "realized_rho": self.realized_rho,
        }

    @property
    def max_relative_spread(self) -> float:
        """Max pairwise relative disagreement across the three measured legs."""
        vals = np.array([self.epsilon_br, self.epsilon_sinkhorn, self.epsilon_cks])
        vals = vals[np.isfinite(vals) & (vals > 0)]
        if vals.size < 2:
            return float("inf")
        return float((vals.max() - vals.min()) / vals.mean())


def triangulate_epsilon(
    cfg: Config,
    h_ref: Optional[float] = None,
    seed: int = 0,
    n_episodes: int = 8,
    br_delta_frac: float = 0.25,
    simulator: Optional[StructuralSimulator] = None,
) -> TriangulationResult:
    """Run the three-leg ``epsilon`` triangulation at ``h_ref``.

    ``h_ref`` defaults to the quoting anchor ``reward.quote_anchor_ref`` -- the
    spread level where the retraining loop actually operates (and, in
    calibrated configs, the *observed* market spread).  It deliberately does
    NOT default to the analytic fixed point ``h*``: the learned pipeline's
    operating point sits well inside ``h*`` (the blind operator underprices the
    toxic channel), and at ``h*`` the learned best-response map is nearly flat,
    so the BR leg of the triangulation has no signal there.  All three legs and
    the closed forms ``epsilon``/``gamma`` are evaluated at the *same*
    ``h_ref``, so the comparison stays apples-to-apples at any probe point a
    caller passes explicitly.  All probe widths are relative to ``h_ref`` so
    calibrated (real-unit) configs work unchanged.
    """
    ref = reference_state(cfg)
    if h_ref is None:
        h_ref = float(cfg.reward.quote_anchor_ref)
    h_ref = float(h_ref)

    if simulator is None:
        torch.manual_seed(seed)
        simulator = StructuralSimulator(cfg)

    g = gamma_of(cfg, h_ref, ref)
    b = beta_of(cfg)
    eps_true = epsilon_of(cfg, h_ref, ref)

    # Realised-state closed form (1.1 §9): the probes run on the state
    # distribution the deployment actually induces, not the a-priori A2 state.
    ref_real = realized_reference(cfg, simulator, h_ref, seed=seed, n_episodes=n_episodes)
    g_real = gamma_of(cfg, h_ref, ref_real)
    eps_real = epsilon_of(cfg, h_ref, ref_real)

    br = measure_response_modulus(
        cfg, seed=seed, h_ref=h_ref, delta=br_delta_frac * h_ref, simulator=simulator
    )
    sink = estimate_epsilon_sinkhorn(
        cfg, h_ref=h_ref, seed=seed, n_episodes=n_episodes, simulator=simulator
    )
    cks = estimate_epsilon_cks(
        cfg, h_ref=h_ref, seed=seed, n_episodes=n_episodes, simulator=simulator
    )

    return TriangulationResult(
        h_ref=h_ref,
        epsilon_analytic=eps_true,
        epsilon_analytic_realized=eps_real,
        epsilon_br=br.modulus * g_real / b if b > 0 else float("inf"),
        epsilon_sinkhorn=sink.epsilon_hat,
        epsilon_cks=cks.epsilon_hat,
        modulus_br=br.modulus,
        gamma=g,
        gamma_realized=g_real,
        beta=b,
        realized_mispricing=ref_real.mispricing,
        realized_rho=ref_real.rho,
        sinkhorn=sink,
        cks=cks,
    )
