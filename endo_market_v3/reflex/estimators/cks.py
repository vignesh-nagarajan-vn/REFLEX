"""CKS informed-flow-slope estimator of the distribution sensitivity ``epsilon``.

Cont-Kukanov-Stoikov ground the performative feedback microstructurally: the
informed-arrival intensity is a function of the quoted spread, and its
derivative at the operating point *is* ``epsilon``:

    eps_hat_CKS = | d lambda_informed / d h |  at  h = h_ref .

This is the third leg of the 1.1 triangulation and the only one estimated from
a *fitted structural model* of the flow curve rather than a local difference:
we deploy fixed-``h`` policies across a small grid, record the mean per-step
informed notional ``lambda(h)`` (common random numbers across grid points),
fit the theory's own functional form

    lambda(h) = C0 + C1 * exp(-c * h)          (theory 1.1: tau(h))

by nonlinear least squares, and differentiate the fit at ``h_ref``:
``eps_hat = c * C1 * exp(-c * h_ref)``.  If the exponential fit fails to
converge (flat curves, tiny grids) the estimator falls back to the central
finite-difference slope of the measured curve and flags it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
import torch

from ..config import Config
from ..env.simulator import StructuralSimulator
from .br_slope import _deployed_policy_at


@dataclass
class CKSEpsilonResult:
    """Result of the CKS informed-flow-slope probe."""

    epsilon_hat: float  # |d lambda_inf / d h| at h_ref
    h_grid: List[float]
    lambda_grid: List[float]  # measured mean informed notional per step
    C0: float  # fitted baseline informed level
    C1: float  # fitted spread-responsive scale
    c: float  # fitted decay rate
    h_ref: float
    fit_ok: bool  # False -> finite-difference fallback was used
    residual_rms: float


def _mean_informed(
    cfg: Config,
    simulator: StructuralSimulator,
    h_dep: float,
    seed: int,
    n_episodes: int,
) -> float:
    """Mean per-step, **per-bond** informed notional under a fixed-``h`` deploy.

    Per-bond so the fitted curve is directly comparable to the theory's
    ``tau(h)`` (single representative bond, 1.1 A1).  Common random numbers
    across grid points.
    """
    policy = _deployed_policy_at(cfg, h_dep)
    gen = torch.Generator().manual_seed(seed)
    horizon = int(cfg.simulator.horizon)
    n_bonds = simulator.bonds.n_bonds
    total = 0.0
    n_steps = 0
    with torch.no_grad():
        for ep in range(int(n_episodes)):
            state = simulator.reset(seed=seed + 71 * ep)
            for _ in range(horizon):
                q = policy.quote(state)
                tr = simulator.step(state, q, generator=gen)
                total += float(tr.fills.informed_volume.sum()) / n_bonds
                n_steps += 1
                state = tr.next_state
    return total / max(n_steps, 1)


def estimate_epsilon_cks(
    cfg: Config,
    h_ref: float,
    h_grid: Optional[Sequence[float]] = None,
    seed: int = 0,
    n_episodes: int = 8,
    simulator: Optional[StructuralSimulator] = None,
) -> CKSEpsilonResult:
    """Estimate ``epsilon`` from the fitted informed-flow curve.

    ``h_grid`` defaults to five points spanning ``[0.5, 1.5] * h_ref`` --
    scale-relative, so calibrated real-unit configs probe proportionally.
    """
    if h_grid is None:
        h_grid = [f * h_ref for f in (0.5, 0.75, 1.0, 1.25, 1.5)]
    h_arr = np.asarray(sorted(float(h) for h in h_grid), dtype=float)
    if h_arr.size < 3:
        raise ValueError("CKS needs at least 3 grid points to fit the flow curve")
    if simulator is None:
        torch.manual_seed(seed)
        simulator = StructuralSimulator(cfg)

    lam = np.array(
        [_mean_informed(cfg, simulator, float(h), seed, n_episodes) for h in h_arr]
    )

    # Fit lambda(h) = C0 + C1 * exp(-c*h) (the structural form of tau(h)).
    from scipy.optimize import curve_fit

    def model(h, C0, C1, c):
        return C0 + C1 * np.exp(-c * h)

    lam_span = float(lam.max() - lam.min())
    p0 = (float(lam.min()), max(lam_span, 1e-6), 1.0 / max(float(h_ref), 1e-9))
    fit_ok = True
    try:
        popt, _ = curve_fit(
            model, h_arr, lam, p0=p0,
            bounds=([0.0, 0.0, 0.0], [np.inf, np.inf, np.inf]),
            maxfev=20000,
        )
        C0, C1, c = (float(v) for v in popt)
        eps_hat = c * C1 * float(np.exp(-c * h_ref))
        resid = lam - model(h_arr, *popt)
    except Exception:
        fit_ok = False
        C0, C1, c = float("nan"), float("nan"), float("nan")
        # Central finite-difference fallback on the measured curve at h_ref.
        i = int(np.argmin(np.abs(h_arr - h_ref)))
        i = min(max(i, 1), h_arr.size - 2)
        eps_hat = abs(float((lam[i + 1] - lam[i - 1]) / (h_arr[i + 1] - h_arr[i - 1])))
        resid = np.zeros_like(lam)

    return CKSEpsilonResult(
        epsilon_hat=float(eps_hat),
        h_grid=[float(h) for h in h_arr],
        lambda_grid=[float(v) for v in lam],
        C0=C0,
        C1=C1,
        c=c,
        h_ref=float(h_ref),
        fit_ok=fit_ok,
        residual_rms=float(np.sqrt(np.mean(resid**2))),
    )
