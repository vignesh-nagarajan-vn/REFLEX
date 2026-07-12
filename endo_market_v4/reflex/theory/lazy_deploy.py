"""Lazy deployment: the K-step RGD outer map and its effective curvature (theory 1.6).

The outer retraining loop never takes an exact best response in practice: with
``rrm.update_rule = "rgd"`` each deployment is followed by ``K = rrm.rgd_steps``
plain gradient steps on the risk under the freshly-refit operator.  This module
closes the gap between the two idealisations the earlier theory covers -- the
exact-BR cobweb (``K -> infinity``, modulus ``m = epsilon*beta/gamma``, theory
1.1) and a single greedy step (``K = 1``, the RGD scheme of Mendler-Duenner et
al. 2020) -- with one closed form derived in
``research/math-theory/06-lazy-deployment.md``:

Linearising the inner gradient flow at the fixed point ``h*``, each inner step
contracts the iterate toward the frozen best response ``BR(h_k)`` by the factor
``c = 1 - eta_h*gamma`` (``eta_h`` the step size *in spread units*), and the
frozen best response itself moves with the deployed spread at the signed slope
``BR'(h*) = -m``.  Composing the two gives the **K-step deployment map slope**

    mu(K) = -m + c^K * (1 + m)                                        (6.1)

with the lazy-deploy reading ``mu(K) = (1 - lam_K)*1 + lam_K*(-m)``,
``lam_K = 1 - c^K``: the outer map is a convex combination of "do nothing"
(slope 1) and the exact cobweb (slope ``-m``).  Consequences, each implemented
below and verified by ``tests/test_lazy_deploy.py``:

* ``mu(0) = 1`` and ``mu(K) -> -m`` monotonically as ``K -> infinity``;
* the effective modulus is ``m_eff(K) = |mu(K)|``, with a **deadbeat** step
  count ``K_db = ln(m/(1+m)) / ln(c)`` where the map slope crosses zero;
* for an RRM-unstable market (``m > 1``) the loop remains stable for
  ``K <= K_max = ln((m-1)/(m+1)) / ln(c)`` -- *laziness stabilises* a cobweb
  that exact retraining would blow up, the quantitative version of the RGD >
  RRM stability-range result; and
* the **effective curvature** ``gamma_eff(K) = gamma * m / m_eff(K)`` -- the
  curvature a plain-RRM reading of the measured modulus would infer.  It has
  two branches, split at the equal-modulus count ``K_eq`` where
  ``c^K = 2m/(1+m)``: below ``K_eq`` the under-trained loop crawls
  (``m_eff > m``), so it reads as a *softer*-than-true objective
  (``gamma_eff < gamma`` -- inertia); above ``K_eq`` the overshoot-cancelling
  regime reads as *extra stiffness* (``gamma_eff > gamma``), diverging at the
  deadbeat count and decaying to the true ``gamma`` from above as
  ``K -> infinity``.

Protocol note (audit-consistent).  ``eta_h`` is a *spread-space* step size; the
ML loop's ``rrm.rgd_lr`` lives in parameter space, so ``c`` is not known a
priori.  The verification protocol of 06 therefore measures the signed slope
``mu_hat(K)`` with the CRN K-step probe
(:func:`reflex.estimators.br_slope.measure_rgd_response`), fits the single
parameter ``c`` by least squares (:func:`fit_inner_contraction`), and checks
the *functional form* plus the parameter-free predictions (monotonicity, the
``K -> infinity`` limit ``-m``, and the deadbeat crossing).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

from ..config import Config
from .analytic_boundary import ReferenceState, analytic_boundary, epsilon as epsilon_of, gamma as gamma_of, reference_state


# --------------------------------------------------------------------------- #
# The closed forms (06 sections 2--4)                                          #
# --------------------------------------------------------------------------- #
def k_step_slope(m: float, c: float, k: float) -> float:
    """Signed K-step deployment-map slope ``mu(K) = -m + c^K (1 + m)`` (06 eq. 6.1).

    ``m >= 0`` is the exact-BR cobweb modulus ``epsilon*beta/gamma`` and
    ``c in [0, 1)`` the per-inner-step contraction ``1 - eta_h*gamma``.
    ``K`` may be fractional (the continuous relaxation used by the fit).
    """
    if not 0.0 <= c < 1.0:
        raise ValueError(f"inner contraction c must be in [0, 1), got {c}")
    if m < 0.0:
        raise ValueError(f"modulus m must be >= 0, got {m}")
    return -m + (c ** k) * (1.0 + m)


def effective_modulus(m: float, c: float, k: float) -> float:
    """Effective outer modulus ``m_eff(K) = |mu(K)|`` (06 section 3.1)."""
    return abs(k_step_slope(m, c, k))


def lazy_weight(c: float, k: float) -> float:
    """The lazy-deploy mixing weight ``lam_K = 1 - c^K`` (06 section 2.2).

    ``mu(K) = (1 - lam_K) * 1 + lam_K * (-m)``: ``lam_K`` is how much of the
    exact cobweb one deployment realises.
    """
    if not 0.0 <= c < 1.0:
        raise ValueError(f"inner contraction c must be in [0, 1), got {c}")
    return 1.0 - c ** k


def deadbeat_k(m: float, c: float) -> float:
    """Continuous step count where the outer map slope crosses zero (06 section 3.2).

    ``mu(K_db) = 0  <=>  c^K = m/(1+m)  <=>  K_db = ln(m/(1+m)) / ln(c)``.
    At ``K_db`` the linearised outer loop converges in one deployment
    (deadbeat control).  Requires ``m > 0`` and ``c in (0, 1)``.
    """
    if m <= 0.0 or not 0.0 < c < 1.0:
        return float("nan")
    return math.log(m / (1.0 + m)) / math.log(c)


def max_stable_k(m: float, c: float) -> float:
    """Largest ``K`` keeping an RRM-unstable market stable (06 section 3.3).

    For ``m > 1`` the exact cobweb diverges, but ``|mu(K)| < 1`` holds while
    ``c^K > (m-1)/(m+1)``, i.e. for

        K < K_max = ln((m-1)/(m+1)) / ln(c) .

    Returns ``inf`` when ``m <= 1`` (every ``K`` is stable in the linearised
    map) and the *continuous* threshold otherwise (callers floor it).
    """
    if m <= 1.0:
        return float("inf")
    if not 0.0 < c < 1.0:
        return float("nan")
    return math.log((m - 1.0) / (m + 1.0)) / math.log(c)


def gamma_eff(gamma: float, m: float, c: float, k: float) -> float:
    """Effective curvature ``gamma_eff(K) = gamma * m / m_eff(K)`` (06 section 4).

    Reading the measured K-step modulus through the plain-RRM identity
    ``m_eff = epsilon*beta/gamma_eff``.  Two branches, split at
    :func:`equal_modulus_k`: below ``K_eq`` the under-trained loop's slope
    exceeds ``m`` (inertia), so ``gamma_eff < gamma``; above ``K_eq`` it reads
    as extra stiffness (``gamma_eff > gamma``), diverging at the deadbeat
    count (the map slope crosses zero, so no finite curvature reproduces it)
    and decaying to ``gamma`` from above as ``K -> infinity`` -- callers should
    report it alongside ``m_eff``, not instead of it.
    """
    me = effective_modulus(m, c, k)
    if me <= 0.0:
        return float("inf")
    return gamma * m / me


def equal_modulus_k(m: float, c: float) -> float:
    """Step count where the lazy modulus equals the exact one (06 section 4).

    ``m_eff(K) = m`` on the positive branch at ``c^K = 2m/(1+m)``:

        K_eq = ln( 2m/(1+m) ) / ln(c) .

    Below ``K_eq``: ``m_eff > m`` (gamma_eff < gamma, the inertia branch);
    above: ``m_eff < m`` (gamma_eff > gamma).  Returns ``0.0`` when
    ``2m/(1+m) >= 1`` (i.e. ``m >= 1``: every ``K >= 1`` is already in the
    stiff branch) and ``nan`` for degenerate inputs.
    """
    if not 0.0 < c < 1.0 or m <= 0.0:
        return float("nan")
    ratio = 2.0 * m / (1.0 + m)
    if ratio >= 1.0:
        return 0.0
    return math.log(ratio) / math.log(c)


def fit_inner_contraction(
    k_values: Sequence[float],
    measured_slopes: Sequence[float],
    m: float,
    n_grid: int = 4001,
) -> float:
    """Least-squares fit of the single parameter ``c`` from measured slopes.

    Minimises ``sum_K (mu_hat(K) - (-m + c^K (1+m)))^2`` over ``c in [0, 1)``
    on a fine grid followed by a golden-section refinement (the objective is
    smooth and unimodal in the operating range; the grid guards against the
    shallow tail near ``c -> 1``).
    """
    ks = np.asarray(list(k_values), dtype=float)
    mus = np.asarray(list(measured_slopes), dtype=float)
    if ks.size == 0 or ks.size != mus.size:
        raise ValueError("k_values and measured_slopes must be equal-length and non-empty")

    def sse(c: float) -> float:
        pred = -m + (c ** ks) * (1.0 + m)
        return float(((mus - pred) ** 2).sum())

    grid = np.linspace(0.0, 1.0 - 1e-6, int(n_grid))
    losses = np.array([sse(float(c)) for c in grid])
    c0 = float(grid[int(np.argmin(losses))])
    # Golden-section refinement in a +/- one-grid-step bracket.
    lo = max(0.0, c0 - 1.5 / n_grid)
    hi = min(1.0 - 1e-9, c0 + 1.5 / n_grid)
    phi = (math.sqrt(5.0) - 1.0) / 2.0
    a, b = lo, hi
    for _ in range(60):
        x1 = b - phi * (b - a)
        x2 = a + phi * (b - a)
        if sse(x1) < sse(x2):
            b = x2
        else:
            a = x1
    return float(0.5 * (a + b))


# --------------------------------------------------------------------------- #
# Assembled analysis                                                           #
# --------------------------------------------------------------------------- #
@dataclass
class LazyDeployCurve:
    """Closed-form K-curve for one config (06 sections 2--4)."""

    m: float  # exact-BR cobweb modulus at the evaluation spread
    c: float  # inner per-step contraction (given or fitted)
    k_values: List[float]
    slopes: List[float]  # mu(K), signed
    moduli: List[float]  # |mu(K)|
    gamma: float
    gamma_eff: List[float]  # gamma * m / |mu(K)|
    deadbeat: float  # continuous deadbeat K (nan if m == 0)
    k_max_stable: float  # inf if m <= 1


def lazy_deploy_curve(
    cfg: Config,
    k_values: Sequence[float],
    c: float,
    h_eval: Optional[float] = None,
    ref: Optional[ReferenceState] = None,
) -> LazyDeployCurve:
    """Evaluate the closed-form K-curve for ``cfg`` at inner contraction ``c``.

    ``h_eval`` defaults to the analytic fixed point ``h*``; pass the probe
    spread to compare against measured slopes (the audit's probe-at-the-
    operating-spread convention).  ``m`` is assembled from the same
    ``epsilon``/``gamma``/``beta`` closed forms as theory 1.1.
    """
    if ref is None:
        ref = reference_state(cfg)
    ab = analytic_boundary(cfg)
    if h_eval is None:
        h_eval = ab.h_star
        m = ab.modulus
        g = ab.gamma
    else:
        g = gamma_of(cfg, float(h_eval), ref)
        eps = epsilon_of(cfg, float(h_eval), ref)
        m = eps * ab.beta / g if g > 0.0 else float("inf")
    ks = [float(k) for k in k_values]
    slopes = [k_step_slope(m, c, k) for k in ks]
    return LazyDeployCurve(
        m=m,
        c=float(c),
        k_values=ks,
        slopes=slopes,
        moduli=[abs(s) for s in slopes],
        gamma=g,
        gamma_eff=[gamma_eff(g, m, c, k) for k in ks],
        deadbeat=deadbeat_k(m, c),
        k_max_stable=max_stable_k(m, c),
    )
