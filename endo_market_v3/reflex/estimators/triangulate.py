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
    analytic_boundary,
    beta as beta_of,
    epsilon as epsilon_of,
    gamma as gamma_of,
    reference_state,
)
from .br_slope import measure_response_modulus
from .cks import CKSEpsilonResult, estimate_epsilon_cks
from .sinkhorn import SinkhornEpsilonResult, estimate_epsilon_sinkhorn


@dataclass
class TriangulationResult:
    """The three measured ``epsilon`` estimates next to the closed form."""

    h_ref: float
    epsilon_analytic: float
    epsilon_br: float  # m_hat * gamma / beta (BR-slope leg)
    epsilon_sinkhorn: float
    epsilon_cks: float
    modulus_br: float  # the raw BR-slope modulus m_hat
    gamma: float
    beta: float
    sinkhorn: SinkhornEpsilonResult
    cks: CKSEpsilonResult

    def as_dict(self) -> dict:
        return {
            "h_ref": self.h_ref,
            "epsilon_analytic": self.epsilon_analytic,
            "epsilon_br": self.epsilon_br,
            "epsilon_sinkhorn": self.epsilon_sinkhorn,
            "epsilon_cks": self.epsilon_cks,
            "modulus_br": self.modulus_br,
            "gamma": self.gamma,
            "beta": self.beta,
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

    ``h_ref`` defaults to the analytic fixed point ``h*`` (theory 1.1 §4), the
    point at which the boundary statement is made.  All probe widths are
    relative to ``h_ref`` so calibrated (real-unit) configs work unchanged.
    """
    ref = reference_state(cfg)
    if h_ref is None:
        h_ref = analytic_boundary(cfg).h_star
    h_ref = float(h_ref)

    if simulator is None:
        torch.manual_seed(seed)
        simulator = StructuralSimulator(cfg)

    g = gamma_of(cfg, h_ref, ref)
    b = beta_of(cfg)
    eps_true = epsilon_of(cfg, h_ref, ref)

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
        epsilon_br=br.modulus * g / b if b > 0 else float("inf"),
        epsilon_sinkhorn=sink.epsilon_hat,
        epsilon_cks=cks.epsilon_hat,
        modulus_br=br.modulus,
        gamma=g,
        beta=b,
        sinkhorn=sink,
        cks=cks,
    )
