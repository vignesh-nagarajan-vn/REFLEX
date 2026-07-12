"""Wasserstein / Sinkhorn estimator of the distribution sensitivity ``epsilon``.

Perdomo et al.'s ``epsilon`` is a Wasserstein-1 Lipschitz constant of the
decision->distribution map: ``W1(D(h), D(h')) <= epsilon * |h - h'|``.  This
module estimates it *directly in distribution space* -- the second leg of the
1.1 triangulation, methodologically independent of the BR-slope probe:

    eps_hat_W = W1( D(h_ref + d), D(h_ref - d) ) / (2 d)

where the samples are per-step realised *toxic notionals* (summed over bonds)
collected from paired fixed-``h`` deployments driven by **common random
numbers** (same episode seeds and RNG streams, so the difference isolates the
spread response).

Two distance backends:

* :func:`quantile_w1` -- the exact 1-D Wasserstein-1 distance via quantile
  coupling (sorted samples).  Default: the toxic-notional marginal is 1-D, and
  exactness beats approximation where both are available.
* :func:`sinkhorn_divergence` -- debiased log-domain entropic OT for point
  clouds of any dimension (Cuturi-Peyre), used for the joint
  ``(toxic, gross)`` flow features and as a cross-check of the 1-D path.
  Implemented from scratch on numpy (no extra dependency).

Relation to the closed form: theory 1.1's ``epsilon = |d E[tau]/dh|`` is the
*mean-shift* sensitivity, and ``W1 >= |E[tau1] - E[tau2]|`` in general, with
equality for pure location shifts.  The toxic response is
location-shift-dominated at the operating point (Miller et al.'s location-scale
reading), so agreement of ``eps_hat_W`` with the analytic ``epsilon`` -- rather
than a large gap -- is itself evidence for that structural reading.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch

from ..config import Config
from ..env.simulator import StructuralSimulator
from ..types import Quotes
from .br_slope import _deployed_policy_at


# --------------------------------------------------------------------------- #
# Distances                                                                    #
# --------------------------------------------------------------------------- #
def quantile_w1(x: np.ndarray, y: np.ndarray) -> float:
    """Exact 1-D Wasserstein-1 distance via the quantile coupling.

    For equal sample sizes this is ``mean(|sort(x) - sort(y)|)``; for unequal
    sizes both empirical quantile functions are evaluated on a common grid.
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if x.size == 0 or y.size == 0:
        raise ValueError("quantile_w1 needs non-empty samples")
    if x.size == y.size:
        return float(np.mean(np.abs(np.sort(x) - np.sort(y))))
    grid = np.linspace(0.0, 1.0, 512, endpoint=False) + 0.5 / 512
    qx = np.quantile(x, grid)
    qy = np.quantile(y, grid)
    return float(np.mean(np.abs(qx - qy)))


def _sinkhorn_potentials(
    cost: np.ndarray, reg: float, n_iters: int
) -> "tuple[np.ndarray, np.ndarray]":
    """Log-domain Sinkhorn iterations for uniform marginals; returns (f, g)."""
    n, m = cost.shape
    log_mu = -math_log(n) * np.ones(n)
    log_nu = -math_log(m) * np.ones(m)
    f = np.zeros(n)
    g = np.zeros(m)
    for _ in range(int(n_iters)):
        # f_i = -reg * logsumexp( (g_j - C_ij)/reg + log_nu_j )
        M = (g[None, :] - cost) / reg + log_nu[None, :]
        f = -reg * _logsumexp(M, axis=1)
        M = (f[:, None] - cost) / reg + log_mu[:, None]
        g = -reg * _logsumexp(M, axis=0)
    return f, g


def _logsumexp(a: np.ndarray, axis: int) -> np.ndarray:
    amax = np.max(a, axis=axis, keepdims=True)
    out = np.log(np.sum(np.exp(a - amax), axis=axis)) + np.squeeze(amax, axis=axis)
    return out


def math_log(x: float) -> float:
    return float(np.log(x))


def _entropic_ot_cost(x: np.ndarray, y: np.ndarray, reg: float, n_iters: int) -> float:
    """Entropic OT transport cost <pi, C> with Euclidean ground cost."""
    x = np.atleast_2d(np.asarray(x, dtype=float))
    y = np.atleast_2d(np.asarray(y, dtype=float))
    if x.shape[0] == 1 and x.shape[1] > 1 and x.ndim == 2 and y.shape[0] == 1:
        # row-vector 1-D inputs -> column vectors
        x = x.T
        y = y.T
    cost = np.sqrt(((x[:, None, :] - y[None, :, :]) ** 2).sum(-1) + 1e-16)
    f, g = _sinkhorn_potentials(cost, reg, n_iters)
    n, m = cost.shape
    log_mu = -math_log(n) * np.ones(n)
    log_nu = -math_log(m) * np.ones(m)
    log_pi = (f[:, None] + g[None, :] - cost) / reg + log_mu[:, None] + log_nu[None, :]
    pi = np.exp(log_pi)
    return float((pi * cost).sum())


def sinkhorn_divergence(
    x: np.ndarray, y: np.ndarray, reg: float = 0.05, n_iters: int = 200
) -> float:
    """Debiased Sinkhorn divergence ``S(x,y) - (S(x,x) + S(y,y))/2``.

    With Euclidean ground cost this approximates ``W1`` as ``reg -> 0`` and the
    debiasing removes the entropic blur at finite ``reg``.  Samples are point
    clouds ``[n, d]`` (1-D inputs are treated as ``d = 1``).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    if y.ndim == 1:
        y = y[:, None]
    sxy = _entropic_ot_cost(x, y, reg, n_iters)
    sxx = _entropic_ot_cost(x, x, reg, n_iters)
    syy = _entropic_ot_cost(y, y, reg, n_iters)
    return float(max(sxy - 0.5 * (sxx + syy), 0.0))


# --------------------------------------------------------------------------- #
# Entropic-regularisation tuning (v4)                                          #
# --------------------------------------------------------------------------- #
# Tuned default: entropic blur as a fraction of the pooled sample scale (std).
# An *absolute* blur breaks on calibrated real-unit configs (the same gotcha as
# absolute probe widths), so the auto path is scale-relative.  The value is the
# bias-minimising point of the U-shaped curve measured by
# experiments/run_tuning.py at the default iteration budget (300): rel bias
# 93.6% at 0.002 and 35.4% at 0.01 (log-domain under-convergence), 5.4% at
# 0.02 (the minimum), 13.1% at 0.05 and 24.6% at 0.10 (entropic blur beyond
# what debiasing removes) on synthetic location-shift pairs with known W1;
# the config's CRN toxic samples give the same minimiser (bias 1.2% at 0.02).
# tests/test_tuning.py re-verifies the tuner on the synthetic case.
TUNED_REL_REG = 0.02


@dataclass
class SinkhornTuning:
    """Result of tuning the entropic blur against the exact 1-D W1."""

    rel_grid: List[float]  # blur / pooled-std grid
    abs_grid: List[float]  # absolute blur values used
    divergences: List[float]  # debiased Sinkhorn divergence per blur
    rel_bias: List[float]  # |S_reg - W1_exact| / max(W1_exact, floor)
    w1_exact: float  # exact quantile-coupling W1 (the 1-D ground truth)
    scale: float  # pooled sample std used for the relative grid
    best_rel_reg: float
    best_abs_reg: float
    best_rel_bias: float


def tune_sinkhorn_reg(
    x: np.ndarray,
    y: np.ndarray,
    rel_grid: Optional[List[float]] = None,
    n_iters: int = 300,
) -> SinkhornTuning:
    """Tune the entropic blur on a sample pair with 1-D ground truth.

    The debiased divergence approaches ``W1`` as ``reg -> 0`` *only with enough
    iterations* -- at a fixed budget a too-small blur under-converges (bias up)
    and a too-large blur over-smooths beyond what debiasing removes (bias up),
    so the bias curve is U-shaped with a data-dependent optimum.  In 1-D the
    exact quantile ``W1`` is available, so the bias is directly measurable:
    this picks the blur minimising ``|S_reg - W1_exact|`` on a scale-relative
    grid.  The point of tuning on the 1-D marginal is to carry the tuned
    *relative* blur to the joint (multi-dimensional) features where no exact
    ``W1`` exists.
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if rel_grid is None:
        rel_grid = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
    scale = float(np.concatenate([x, y]).std())
    scale = max(scale, 1e-9)
    w1 = quantile_w1(x, y)
    floor = max(abs(w1), 1e-9)
    abs_grid = [float(r) * scale for r in rel_grid]
    divs = [sinkhorn_divergence(x, y, reg=a, n_iters=n_iters) for a in abs_grid]
    bias = [abs(d - w1) / floor for d in divs]
    i_best = int(np.argmin(bias))
    return SinkhornTuning(
        rel_grid=[float(r) for r in rel_grid],
        abs_grid=abs_grid,
        divergences=divs,
        rel_bias=bias,
        w1_exact=w1,
        scale=scale,
        best_rel_reg=float(rel_grid[i_best]),
        best_abs_reg=abs_grid[i_best],
        best_rel_bias=float(bias[i_best]),
    )


def tune_sinkhorn_reg_for_config(
    cfg: Config,
    h_ref: float,
    delta: Optional[float] = None,
    seed: int = 0,
    n_episodes: int = 8,
    rel_grid: Optional[List[float]] = None,
    n_iters: int = 300,
    simulator: Optional[StructuralSimulator] = None,
) -> SinkhornTuning:
    """Tune the blur on the config's own CRN toxic-flow samples at ``h_ref``.

    Collects the same paired fixed-``h`` deployments as
    :func:`estimate_epsilon_sinkhorn` and runs :func:`tune_sinkhorn_reg` on
    them -- the operational tuning used by ``experiments/run_tuning.py``.
    """
    delta = 0.25 * h_ref if delta is None else float(delta)
    if simulator is None:
        torch.manual_seed(seed)
        simulator = StructuralSimulator(cfg)
    tau_plus = _collect_toxic_samples(cfg, simulator, h_ref + delta, seed, n_episodes)
    tau_minus = _collect_toxic_samples(cfg, simulator, h_ref - delta, seed, n_episodes)
    return tune_sinkhorn_reg(tau_plus, tau_minus, rel_grid=rel_grid, n_iters=n_iters)


# --------------------------------------------------------------------------- #
# The epsilon estimator                                                        #
# --------------------------------------------------------------------------- #
@dataclass
class SinkhornEpsilonResult:
    """Result of the Wasserstein-based ``epsilon`` probe."""

    epsilon_hat: float  # W1 / (2*delta)
    w1: float  # distance between the two induced toxic-flow distributions
    mean_shift: float  # |mean(tau-) - mean(tau+)| (location-shift component)
    h_ref: float
    delta: float
    method: str  # "quantile" or "sinkhorn"
    n_samples: int


def _collect_toxic_samples(
    cfg: Config,
    simulator: StructuralSimulator,
    h_dep: float,
    seed: int,
    n_episodes: int,
) -> np.ndarray:
    """Per-step, **per-bond mean** toxic notionals under a fixed-``h`` policy.

    Per-bond so the estimate is directly comparable to the theory's ``tau(h)``
    (derived for a single representative bond, 1.1 A1).  Seeding is derived
    deterministically from ``seed`` alone, so paired calls at ``h_ref +/-
    delta`` share every RNG stream (common random numbers).
    """
    policy = _deployed_policy_at(cfg, h_dep)
    gen = torch.Generator().manual_seed(seed)
    horizon = int(cfg.simulator.horizon)
    n_bonds = simulator.bonds.n_bonds
    samples: List[float] = []
    with torch.no_grad():
        for ep in range(int(n_episodes)):
            state = simulator.reset(seed=seed + 71 * ep)
            for _ in range(horizon):
                q = policy.quote(state)
                tr = simulator.step(state, Quotes(q.half_spread, q.skew), generator=gen)
                samples.append(float(tr.fills.informed_volume.sum()) / n_bonds)
                state = tr.next_state
    return np.asarray(samples, dtype=float)


def estimate_epsilon_sinkhorn(
    cfg: Config,
    h_ref: float,
    delta: Optional[float] = None,
    seed: int = 0,
    n_episodes: int = 8,
    method: str = "quantile",
    reg: "float | str" = "auto",
    simulator: Optional[StructuralSimulator] = None,
) -> SinkhornEpsilonResult:
    """Estimate ``epsilon`` as the Wasserstein rate of the induced toxic flow.

    ``delta`` defaults to ``0.25 * h_ref`` (scale-relative, so calibrated
    real-unit configs probe proportionally).  ``reg`` (used only when
    ``method="sinkhorn"``) defaults to ``"auto"``: the tuned scale-relative
    blur ``TUNED_REL_REG * pooled_std`` -- an absolute blur would break on
    real-unit configs the same way absolute probe widths do.
    """
    if method not in ("quantile", "sinkhorn"):
        raise ValueError(f"unknown method {method!r}")
    delta = 0.25 * h_ref if delta is None else float(delta)
    if simulator is None:
        torch.manual_seed(seed)
        simulator = StructuralSimulator(cfg)
    tau_plus = _collect_toxic_samples(cfg, simulator, h_ref + delta, seed, n_episodes)
    tau_minus = _collect_toxic_samples(cfg, simulator, h_ref - delta, seed, n_episodes)
    if method == "quantile":
        w1 = quantile_w1(tau_plus, tau_minus)
    else:
        if reg == "auto":
            pooled = np.concatenate([tau_plus, tau_minus])
            reg_abs = TUNED_REL_REG * max(float(pooled.std()), 1e-9)
        else:
            reg_abs = float(reg)
        w1 = sinkhorn_divergence(tau_plus, tau_minus, reg=reg_abs)
    return SinkhornEpsilonResult(
        epsilon_hat=w1 / (2.0 * delta),
        w1=w1,
        mean_shift=abs(float(tau_minus.mean() - tau_plus.mean())),
        h_ref=float(h_ref),
        delta=delta,
        method=method,
        n_samples=int(tau_plus.size),
    )
