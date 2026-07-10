"""Phase-diagram assembly: analytic predictions next to measured sweeps.

Thin, dependency-light helpers the experiment scripts share:

* :func:`predicted_epsilon_sweep` -- the closed-form modulus curve
  ``m_pred(f)`` over a ``toxicity_feedback`` grid (theory 1.1's a-priori
  prediction, evaluated *before* any simulation), the overlay for the measured
  phase diagram (predict-then-verify protocol, 1.1 §8).
* :func:`dealer_phase_grid` -- the analytic ``(N, f)`` stability surface
  ``m_N = N_eff * m_1`` (theory 1.3), the multi-dealer phase diagram.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

from ..config import Config
from ..theory.analytic_boundary import analytic_boundary
from ..theory.multi_dealer import multi_dealer_boundary


@dataclass
class PredictedPoint:
    """Closed-form stability quantities at one feedback-gain value."""

    value: float  # toxicity_feedback f
    m_pred: float  # analytic modulus at the fixed point
    h_star: float
    gamma: float
    epsilon: float
    eps_star: float  # gamma/beta


def predicted_epsilon_sweep(
    base_cfg: Config, grid: Sequence[float]
) -> List[PredictedPoint]:
    """Evaluate the analytic boundary across a ``toxicity_feedback`` grid."""
    out: List[PredictedPoint] = []
    for f in grid:
        cfg = copy.deepcopy(base_cfg)
        cfg.clients.toxicity_feedback = float(f)
        ab = analytic_boundary(cfg)
        out.append(
            PredictedPoint(
                value=float(f),
                m_pred=ab.modulus,
                h_star=ab.h_star,
                gamma=ab.gamma,
                epsilon=ab.epsilon,
                eps_star=ab.boundary_epsilon,
            )
        )
    return out


def predicted_crossing(points: List[PredictedPoint]) -> Optional[float]:
    """Interpolate where the predicted modulus crosses 1 (the a-priori f*)."""
    vals = np.array([p.value for p in points])
    ms = np.array([p.m_pred for p in points])
    for i in range(len(vals) - 1):
        if ms[i] < 1.0 <= ms[i + 1]:
            t = (1.0 - ms[i]) / (ms[i + 1] - ms[i])
            return float(vals[i] + t * (vals[i + 1] - vals[i]))
    return None


def dealer_phase_grid(
    base_cfg: Config,
    n_values: Sequence[int],
    f_grid: Sequence[float],
    kappa: Optional[float] = None,
) -> np.ndarray:
    """Analytic joint modulus ``m_N`` on an ``(N, f)`` grid (rows = N)."""
    out = np.zeros((len(n_values), len(f_grid)))
    for i, n in enumerate(n_values):
        for j, f in enumerate(f_grid):
            cfg = copy.deepcopy(base_cfg)
            cfg.clients.toxicity_feedback = float(f)
            mb = multi_dealer_boundary(cfg, n_dealers=int(n), kappa=kappa)
            out[i, j] = mb.m_N
    return out
