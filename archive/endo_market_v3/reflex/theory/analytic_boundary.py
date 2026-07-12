"""Closed-form performative-stability constants for the single-dealer market.

This module is the *analytic* companion to the model-free probe in
:mod:`reflex.estimators.br_slope`.  It implements, as pure functions of
:class:`~reflex.config.Config`, the closed forms derived in
``research/math-theory/01-analytic-stability-boundary.md`` (§2--§4): the
strong-convexity constant ``gamma``, the joint-smoothness constant ``beta``, the
distribution-sensitivity ``epsilon(h)``, the toxic-flow level ``tau(h)``, the
adverse severity ``psi``, the best-response map, its fixed point ``h*``, and the
contraction modulus

    m = epsilon * beta / gamma       (stable  <=>  m < 1  <=>  epsilon < gamma/beta).

Every quantity is a closed form in the config and the reference state -- none is
swept or tuned -- so the boundary is an *a-priori* predictor that can be checked
against :func:`measure_response_modulus` (the predict-then-verify protocol of 1.1
§8).  The module is deliberately dependency-light (numpy + scipy quadrature/root
finding, no torch) so it is cheap to call inside sweeps.

It is also the shared foundation the two downstream priorities build on:

* ``equilibrium/perfgd_loop.py`` (math-theory 1.2) imports ``epsilon``, ``psi``
  and the best-response map to add the analytic PerfGD correction; and
* ``analysis/multi_dealer_modulus.py`` (math-theory 1.3) reuses ``gamma``,
  ``beta``, ``epsilon`` and ``solve_fixed_point`` for the ``N``-dealer boundary.

Reference-state convention
--------------------------
The curvature and gate constants are state dependent (1.1 A2): they are evaluated
at the probe reference state ``s0 = simulator.reset(...)`` with inventory
``q0 ~ 0``.  To keep this a *pure* function of ``Config`` we take representative
reference values -- the liquidity ratio ``rho = liquidity / liq_mean`` (defaults
to ``1.0``, the field's long-run mean) and the mispricing ``g`` (defaults to a
one-sigma draw ``init_mispricing_vol``) -- as optional arguments.  Callers that
want to compare against a specific realised reference state may pass its measured
``mispricing`` and ``liquidity_ratio`` explicitly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.polynomial.hermite_e import hermegauss

from ..config import Config

# Gauss--Hermite quadrature (probabilists' weight exp(-x^2/2)) for the two
# state-constant Gaussian integrals in 1.1 §2.5.  64 nodes is far more than the
# smooth bounded ``tanh`` integrands need; it makes the constants deterministic.
_GH_NODES, _GH_WEIGHTS = hermegauss(64)
_GH_NORM = float(_GH_WEIGHTS.sum())  # == sqrt(2*pi)


def _expect_gaussian(fn) -> float:
    """E_{eta ~ N(0,1)}[fn(eta)] by Gauss--Hermite quadrature."""
    vals = fn(_GH_NODES)
    return float((_GH_WEIGHTS * vals).sum() / _GH_NORM)


def gate_means(u: float) -> tuple[float, float]:
    """Return ``(gbar, a_signed)`` for standardized mispricing ``u = g / sigma_s``.

    ``gbar(u) = E[tanh(|u + eta|)]`` is the (unsigned) gate mean that scales the
    toxic notional; ``a_signed(u) = E[tanh(u + eta)]`` is the signed gate mean
    entering the adverse severity ``psi`` (1.1 §2.5).  Both are 1-D Gaussian
    integrals in ``u`` alone and are independent of the half-spread ``h``.
    """
    gbar = _expect_gaussian(lambda eta: np.tanh(np.abs(u + eta)))
    a_signed = _expect_gaussian(lambda eta: np.tanh(u + eta))
    return gbar, a_signed


@dataclass
class ReferenceState:
    """Evaluated state constants at the probe reference state (1.1 A2/§2.5)."""

    rho: float  # liquidity ratio  liquidity / liq_mean
    mispricing: float  # g = fundamental - mid
    sigma_s: float  # info_signal_noise
    u: float  # standardized mispricing g / sigma_s
    gbar: float  # gate mean  E[tanh(|u + eta|)]
    a_signed: float  # signed gate mean  E[tanh(u + eta)]
    psi: float  # adverse severity  sigma_s * u * a_signed / gbar  ( > 0 )


def reference_state(
    cfg: Config,
    mispricing: Optional[float] = None,
    liquidity_ratio: float = 1.0,
) -> ReferenceState:
    """Build the :class:`ReferenceState` for ``cfg`` (1.1 §2.5).

    ``mispricing`` defaults to a one-sigma representative
    ``simulator.init_mispricing_vol``; ``liquidity_ratio`` defaults to ``1.0``
    (liquidity at its long-run mean ``liq_mean``).
    """
    sigma_s = max(float(cfg.clients.info_signal_noise), 1e-6)
    g = float(cfg.simulator.init_mispricing_vol if mispricing is None else mispricing)
    u = g / sigma_s
    gbar, a_signed = gate_means(u)
    gbar = max(gbar, 1e-9)
    psi = sigma_s * u * a_signed / gbar  # >= 0 since sign(u) == sign(a_signed)
    return ReferenceState(
        rho=float(liquidity_ratio),
        mispricing=g,
        sigma_s=sigma_s,
        u=u,
        gbar=gbar,
        a_signed=a_signed,
        psi=psi,
    )


# --------------------------------------------------------------------------- #
# The three closed-form constants and the toxic level (1.1 §2--§3)            #
# --------------------------------------------------------------------------- #
def tau(cfg: Config, h: float, ref: ReferenceState) -> float:
    """Expected informed (toxic) notional at deployed half-spread ``h`` (1.1 §2.3).

    ``tau(h) = rho * gbar * ( I_b + alpha*f*I * exp(-c_t*h) )``.
    """
    c = cfg.clients
    return ref.rho * ref.gbar * (
        c.info_base_intensity
        + c.alpha * c.toxicity_feedback * c.info_intensity * math.exp(-c.info_spread_decay * h)
    )


def epsilon(cfg: Config, h: float, ref: ReferenceState) -> float:
    """Distribution sensitivity ``epsilon(h) = |d tau / d h|`` (1.1 §3.2 (***)).

    ``epsilon(h) = rho * gbar * alpha*f*I * c_t * exp(-c_t*h)`` -- the slope of the
    toxic-flow response to the deployed spread; linear in ``alpha`` and in
    ``toxicity_feedback``.
    """
    c = cfg.clients
    return (
        ref.rho * ref.gbar * c.alpha * c.toxicity_feedback * c.info_intensity
        * c.info_spread_decay * math.exp(-c.info_spread_decay * h)
    )


def beta(cfg: Config) -> float:
    """Joint smoothness ``beta = |d^2 J / dh dT| = P = pnl_scale`` (1.1 §3.2 (**))."""
    return float(cfg.reward.pnl_scale)


def gamma(cfg: Config, h: float, ref: ReferenceState, lambda_q: Optional[float] = None) -> float:
    """Strong-convexity constant ``gamma = -d^2 J/dh^2`` (1.1 §3.2 (*)).

    ``gamma = P * [ 2*w + A*rho*k*exp(-k*h)*(2 - k*h) + lambda_q ]``.  ``lambda_q``
    is the (stabilising) inventory-risk curvature; it defaults to
    ``reward.inv_risk_weight`` (the symbol map lists it as proportional to that
    weight, 1.1 §7).  Note the GLFT term ``A*rho*k*exp(-k*h)*(2 - k*h)`` turns
    negative once ``k*h > 2`` -- the defensive-widening low-curvature region of
    1.1 §6.3.
    """
    c = cfg.clients
    P = float(cfg.reward.pnl_scale)
    w = float(cfg.reward.quote_anchor_weight)
    A = float(c.base_arrival_rate)
    k = float(c.demand_elasticity)
    lq = float(cfg.reward.inv_risk_weight if lambda_q is None else lambda_q)
    glft = A * ref.rho * k * math.exp(-k * h) * (2.0 - k * h)
    return P * (2.0 * w + glft + lq)


def blind_gradient(cfg: Config, h: float, ref: ReferenceState) -> float:
    """Self-consistent blind gradient ``G(h) = d1 J(h; tau(h))`` (1.2 §1).

    ``G(h) = P * [ A*rho*exp(-k*h)*(1 - k*h) + tau(h) - 2*w*(h - h_ref) ]``.  Its
    root is the stable point / RRM fixed point ``h_SP = h*`` of 1.1 §4.
    """
    c = cfg.clients
    P = float(cfg.reward.pnl_scale)
    w = float(cfg.reward.quote_anchor_weight)
    h_ref = float(cfg.reward.quote_anchor_ref)
    A = float(c.base_arrival_rate)
    k = float(c.demand_elasticity)
    uninformed = A * ref.rho * math.exp(-k * h) * (1.0 - k * h)
    return P * (uninformed + tau(cfg, h, ref) - 2.0 * w * (h - h_ref))


def frozen_gradient(cfg: Config, h: float, toxic_level: float, ref: ReferenceState) -> float:
    """Best-response gradient ``F(h; T) = dJ/dh`` at a *frozen* toxic level ``T``.

    ``F(h; T) = P * [ A*rho*exp(-k*h)*(1 - k*h) + T - 2*w*(h - h_ref) ]`` (1.1
    §3.2).  A dealer's frozen-environment best response is the root of ``F`` in
    ``h``; :func:`best_response` solves it.  Setting ``T = tau(h)`` recovers
    :func:`blind_gradient`.
    """
    c = cfg.clients
    P = float(cfg.reward.pnl_scale)
    w = float(cfg.reward.quote_anchor_weight)
    h_ref = float(cfg.reward.quote_anchor_ref)
    A = float(c.base_arrival_rate)
    k = float(c.demand_elasticity)
    uninformed = A * ref.rho * math.exp(-k * h) * (1.0 - k * h)
    return P * (uninformed + toxic_level - 2.0 * w * (h - h_ref))


# --------------------------------------------------------------------------- #
# Root finding                                                               #
# --------------------------------------------------------------------------- #
def _bracketed_root(fn, lo: float, hi: float, n_grid: int = 256) -> float:
    """Return a root of ``fn`` on ``[lo, hi]``.

    Scans a grid for the first sign change and refines with scipy's Brent method;
    if no sign change is found, returns the grid point of least ``|fn|`` (the
    boundary of the operating range).  Robust for the smooth, single-crossing
    gradients of this model.
    """
    from scipy.optimize import brentq

    xs = np.linspace(lo, hi, n_grid)
    vals = np.array([fn(float(x)) for x in xs])
    sign_change = np.where(np.sign(vals[:-1]) * np.sign(vals[1:]) < 0)[0]
    if sign_change.size:
        i = int(sign_change[0])
        return float(brentq(fn, xs[i], xs[i + 1], xtol=1e-10, rtol=1e-12))
    return float(xs[int(np.argmin(np.abs(vals)))])


def best_response(cfg: Config, toxic_level: float, ref: ReferenceState) -> float:
    """Frozen-environment best response ``argmax_h J(h; T)`` (root of ``F(.; T)``)."""
    hi = float(cfg.policy.max_half_spread)
    return _bracketed_root(lambda h: frozen_gradient(cfg, h, toxic_level, ref), 0.0, hi)


def solve_fixed_point(cfg: Config, ref: ReferenceState) -> float:
    """Locate the self-consistent fixed point ``h*`` (root of ``G``; 1.1 §4)."""
    hi = float(cfg.policy.max_half_spread)
    return _bracketed_root(lambda h: blind_gradient(cfg, h, ref), 0.0, hi)


# --------------------------------------------------------------------------- #
# Assembled result                                                           #
# --------------------------------------------------------------------------- #
@dataclass
class AnalyticBoundary:
    """Closed-form stability summary at the fixed point ``h*`` (1.1 §3.4)."""

    h_star: float  # self-consistent fixed-point half-spread
    gamma: float  # strong convexity at h*
    beta: float  # joint smoothness ( = pnl_scale )
    epsilon: float  # distribution sensitivity at h*
    modulus: float  # m = epsilon*beta/gamma  ( < 1  =>  stable )
    boundary_epsilon: float  # gamma/beta : the largest stable epsilon
    stable: bool  # modulus < 1
    tau: float  # toxic level at h*
    psi: float  # adverse severity (echo-chamber level; not in the boundary)
    ref: ReferenceState


def analytic_boundary(
    cfg: Config,
    mispricing: Optional[float] = None,
    liquidity_ratio: float = 1.0,
    lambda_q: Optional[float] = None,
) -> AnalyticBoundary:
    """Compute the closed-form stability boundary for ``cfg`` (1.1 §3--§4).

    Returns the fixed point ``h*``, the three constants evaluated there, the
    contraction modulus ``m = epsilon*beta/gamma`` and the boundary
    ``epsilon < gamma/beta``.  This is the ``m_pred`` of the predict-then-verify
    protocol (1.1 §8), directly comparable to
    :func:`measure_response_modulus`'s ``m_meas``.
    """
    ref = reference_state(cfg, mispricing=mispricing, liquidity_ratio=liquidity_ratio)
    h_star = solve_fixed_point(cfg, ref)
    g = gamma(cfg, h_star, ref, lambda_q=lambda_q)
    b = beta(cfg)
    eps = epsilon(cfg, h_star, ref)
    m = eps * b / g if g != 0.0 else math.inf
    return AnalyticBoundary(
        h_star=h_star,
        gamma=g,
        beta=b,
        epsilon=eps,
        modulus=m,
        boundary_epsilon=(g / b if b != 0.0 else math.inf),
        stable=m < 1.0,
        tau=tau(cfg, h_star, ref),
        psi=ref.psi,
        ref=ref,
    )
