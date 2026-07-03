"""Multi-dealer competition and systemic performative instability (math-theory 1.3).

Lifts the single-dealer boundary of
``new-methodology/math-theory/01-analytic-stability-boundary.md`` to ``N``
symmetric dealers who share one informed-flow pool with cross-dealer spillover
``kappa in [0, 1]``.  When *any* dealer tightens it summons toxic flow that picks
off *all* of them, so the joint best-response Jacobian is a rank-one common-mode
coupling

    J_BR = -m_1 * [ (1 - kappa)*I + kappa * 1 1^T ] ,      m_1 = epsilon*beta/gamma,

whose unstable (common-mode) eigenvalue is ``lambda_+ = -m_1*N_eff`` with the
**effective dealer count** ``N_eff = 1 + kappa*(N-1)``.  Hence the PSNE stability
boundary (Theorem 1, 1.3 §4):

    m_N = N_eff * epsilon * beta / gamma < 1
      <=>  epsilon < gamma / (N_eff * beta)                      (kappa=1: gamma/(N*beta)).

Every constant is one of 1.1's closed forms, so this requires **no new
estimation** -- only the ``N``-dealer linear algebra.  The key computational
observation is that the symmetric common mode reduces to a *single-dealer problem
with the toxic slope amplified by ``N_eff``*: at an in-phase profile
``h^dep = h * 1``, dealer ``i`` sees toxic level
``rho*gbar*(I_b + N_eff*alpha*f*I*exp(-c_t*h))`` -- i.e. the single-dealer
environment with ``info_intensity`` scaled by ``N_eff`` (:func:`effective_config`).
This lets the existing single-dealer machinery -- ``solve_fixed_point`` for the
PSNE and :func:`measure_response_modulus` for the empirical common-mode probe --
be reused verbatim, exactly as 1.3 §10 prescribes.

Provided here:

* :func:`n_eff`, :func:`multi_dealer_boundary` -- the boundary, joint modulus
  ``m_N``, joint curvature ``gamma_joint``, and critical dealer count ``N_c``;
* :func:`joint_jacobian` -- the ``N x N`` matrix and its explicit spectrum;
* :func:`run_joint_rrm` -- the genuine ``N``-dimensional best-response cobweb with
  coupled ``tau_i`` (demonstrates common-mode instability and critical slowing
  down without any simulator surgery);
* :func:`measure_common_mode_modulus` / :func:`measure_differential_modulus` /
  :func:`identify_kappa` -- the in-phase and anti-phase empirical probes through
  the full deploy->collect->fit->optimize pipeline, and the free calibration of
  ``kappa`` from their ratio;
* :func:`mean_field_boundary` -- the ``N -> inf`` limits (§7); and
* :func:`sweep_dealer_count` -- the linear-in-``N`` phase-diagram row (§8.2).
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from ..config import Config
from .analytic_boundary import (
    ReferenceState,
    beta as beta_of,
    best_response,
    epsilon as epsilon_of,
    gamma as gamma_of,
    reference_state,
    solve_fixed_point,
    tau as tau_of,
)
from .response_modulus import measure_response_modulus


def n_eff(n_dealers: int, kappa: float) -> float:
    """Effective dealer count ``N_eff = 1 + kappa*(N-1)`` (1.3 §4).

    ``N_eff = 1`` at ``kappa = 0`` (decoupled single-dealer loops) and ``N_eff = N``
    at ``kappa = 1`` (full spillover).
    """
    return 1.0 + float(kappa) * (int(n_dealers) - 1)


def effective_config(cfg: Config, intensity_factor: float) -> Config:
    """Copy ``cfg`` with the toxic *slope* scale ``info_intensity`` multiplied.

    Scaling ``info_intensity`` by ``N_eff`` reproduces the common-mode toxic
    environment exactly: only the spread-responsive slope term
    ``alpha*f*I*exp(-c_t*h)`` is amplified, while the baseline ``I_b`` is untouched
    -- precisely the ``tau_i`` of 1.3 §2.1 at a symmetric in-phase profile.  Scaling
    by ``(1 - kappa)`` gives the anti-phase (differential-mode) environment.
    """
    out = copy.deepcopy(cfg)
    out.clients.info_intensity = float(cfg.clients.info_intensity) * float(intensity_factor)
    return out


def joint_jacobian(m_1: float, n_dealers: int, kappa: float) -> np.ndarray:
    """The ``N x N`` joint best-response Jacobian ``J_BR`` (1.3 §3.2).

    ``J_BR = -m_1 * [ (1 - kappa)*I + kappa * ones(N, N) ]`` -- diagonal ``-m_1``
    (own-deployment self-coupling), off-diagonal ``-kappa*m_1`` (competitor
    coupling through the shared pool).  Symmetric, so its spectral radius equals
    its operator 2-norm.
    """
    N = int(n_dealers)
    eye = np.eye(N)
    ones = np.ones((N, N))
    return -float(m_1) * ((1.0 - float(kappa)) * eye + float(kappa) * ones)


@dataclass
class MultiDealerBoundary:
    """Closed-form ``N``-dealer stability summary at the symmetric PSNE (1.3 §4)."""

    n_dealers: int
    kappa: float
    n_eff: float  # effective dealer count 1 + kappa*(N-1)
    h_star: float  # symmetric PSNE half-spread
    gamma: float  # strong convexity at h*
    beta: float  # joint smoothness
    epsilon: float  # distribution sensitivity at h*
    m_1: float  # single-dealer modulus epsilon*beta/gamma at h*
    m_N: float  # joint modulus N_eff*m_1  ( = spectral radius of J_BR )
    gamma_joint: float  # gamma - N_eff*epsilon*beta (joint effective curvature)
    n_critical: float  # critical dealer count N_c = 1/m_1
    boundary_epsilon: float  # gamma/(N_eff*beta) : the largest stable epsilon
    stable: bool  # m_N < 1
    lambda_plus: float  # common-mode eigenvalue -m_N (unstable direction)
    lambda_minus: float  # differential eigenvalue -m_1*(1-kappa) (multiplicity N-1)


def multi_dealer_boundary(
    cfg: Config,
    n_dealers: Optional[int] = None,
    kappa: Optional[float] = None,
    mispricing: Optional[float] = None,
    liquidity_ratio: float = 1.0,
    lambda_q: Optional[float] = None,
    reference_spread: Optional[float] = None,
) -> MultiDealerBoundary:
    """Compute the ``N``-dealer PSNE boundary for ``cfg`` (1.3 §4--§6, §8.1).

    ``n_dealers`` and ``kappa`` default to ``cfg.clients.n_dealers`` and
    ``cfg.clients.toxic_spillover``.

    Two operating-point conventions are supported:

    * ``reference_spread=None`` (default): the **self-consistent PSNE** ``h*`` is
      solved on the ``N_eff``-amplified effective config (1.3 §5/§10) -- the honest
      value in which competition widens the equilibrium spread, so ``m_1``
      (evaluated at the ``N``-dependent ``h*``) *decays* with ``N`` and ``m_N``
      grows *sub-linearly* (the defensive-widening effect of 1.1 §6.3).
    * ``reference_spread=h0``: the **fixed structural regime** of 1.3 §8.2 --
      ``gamma``, ``epsilon`` and ``m_1`` are evaluated at the held spread ``h0``
      (e.g. the single-dealer PSNE), so ``m_N = N_eff*m_1`` is *exactly linear* in
      ``N`` through the origin and the critical count ``N_c = 1/m_1`` is a clean
      integer crossing -- the falsifiable straight-line prediction.

    In both cases ``m_1``, ``gamma`` and ``epsilon`` come from 1.1's closed forms
    and ``m_N = N_eff*m_1``.
    """
    N = int(cfg.clients.n_dealers if n_dealers is None else n_dealers)
    kap = float(cfg.clients.toxic_spillover if kappa is None else kappa)
    Ne = n_eff(N, kap)

    ref = reference_state(cfg, mispricing=mispricing, liquidity_ratio=liquidity_ratio)
    if reference_spread is None:
        # Symmetric PSNE: 1.1-shaped scalar equation with the toxic slope amplified
        # by N_eff (1.3 §5) -- solved by reusing the single-dealer fixed-point
        # routine on the effective config.
        cfg_eff = effective_config(cfg, Ne)
        ref_eff = reference_state(cfg_eff, mispricing=mispricing, liquidity_ratio=liquidity_ratio)
        h_star = solve_fixed_point(cfg_eff, ref_eff)
    else:
        h_star = float(reference_spread)

    g = gamma_of(cfg, h_star, ref, lambda_q=lambda_q)
    b = beta_of(cfg)
    eps = epsilon_of(cfg, h_star, ref)  # single-dealer epsilon at the PSNE
    m_1 = eps * b / g if g != 0.0 else math.inf
    m_N = Ne * m_1
    return MultiDealerBoundary(
        n_dealers=N,
        kappa=kap,
        n_eff=Ne,
        h_star=h_star,
        gamma=g,
        beta=b,
        epsilon=eps,
        m_1=m_1,
        m_N=m_N,
        gamma_joint=g - Ne * eps * b,
        n_critical=(1.0 / m_1 if m_1 > 0.0 else math.inf),
        boundary_epsilon=(g / (Ne * b) if b != 0.0 else math.inf),
        stable=m_N < 1.0,
        lambda_plus=-m_N,
        lambda_minus=-m_1 * (1.0 - kap),
    )


# --------------------------------------------------------------------------- #
# Mean-field limits (1.3 §7)                                                  #
# --------------------------------------------------------------------------- #
@dataclass
class MeanFieldLimit:
    """The ``N -> inf`` stability boundary under a given spillover scaling (1.3 §7)."""

    regime: str  # "strong" (kappa fixed) or "mean_field" (kappa = c/N)
    n_eff_limit: float  # limiting N_eff  (inf for strong, 1+c for mean-field)
    boundary_epsilon: float  # limiting stable-epsilon bound (0 for strong collapse)


def mean_field_boundary(
    cfg: Config,
    c: float,
    mispricing: Optional[float] = None,
    liquidity_ratio: float = 1.0,
    lambda_q: Optional[float] = None,
) -> MeanFieldLimit:
    """Mean-field (``kappa = c/N``) boundary as ``N -> inf`` (1.3 §7.2).

    Here ``N_eff -> 1 + c`` and the boundary converges to the finite limit
    ``epsilon < gamma/((1+c)*beta)`` -- the well-posed continuum boundary that
    licenses scaling the phase diagram to a large market.  (The complementary
    fixed-``kappa`` "strong" regime of §7.1, where ``N_eff -> inf`` and the
    boundary collapses to ``0``, is returned by :func:`strong_coupling_limit`.)  The
    curvature ``gamma`` and ``epsilon`` are evaluated at the ``N_eff = 1+c``
    effective PSNE.
    """
    Ne = 1.0 + float(c)
    ref = reference_state(cfg, mispricing=mispricing, liquidity_ratio=liquidity_ratio)
    cfg_eff = effective_config(cfg, Ne)
    ref_eff = reference_state(cfg_eff, mispricing=mispricing, liquidity_ratio=liquidity_ratio)
    h_star = solve_fixed_point(cfg_eff, ref_eff)
    g = gamma_of(cfg, h_star, ref, lambda_q=lambda_q)
    b = beta_of(cfg)
    return MeanFieldLimit(
        regime="mean_field",
        n_eff_limit=Ne,
        boundary_epsilon=(g / (Ne * b) if b != 0.0 else math.inf),
    )


def strong_coupling_limit() -> MeanFieldLimit:
    """Fixed-``kappa`` strong-coupling limit ``N -> inf`` (1.3 §7.1): boundary -> 0."""
    return MeanFieldLimit(regime="strong", n_eff_limit=math.inf, boundary_epsilon=0.0)


# --------------------------------------------------------------------------- #
# The genuine N-dimensional joint cobweb (1.3 §3, §8.3)                        #
# --------------------------------------------------------------------------- #
def _coupled_toxic_levels(cfg: Config, h_dep: np.ndarray, kappa: float, ref: ReferenceState) -> np.ndarray:
    """Frozen toxic level ``T_i = tau_i(h^dep)`` at each dealer (1.3 §2.1).

    ``T_i = rho*gbar*( I_b + alpha*f*I*[ exp(-c_t*h_i) + kappa*sum_{j!=i} exp(-c_t*h_j) ] )``.
    """
    c = cfg.clients
    resp = np.exp(-float(c.info_spread_decay) * np.asarray(h_dep, dtype=float))
    total = resp.sum()
    # own term (weight 1) + kappa * (sum over others) = resp + kappa*(total - resp)
    coupled = resp + float(kappa) * (total - resp)
    slope = c.alpha * c.toxicity_feedback * c.info_intensity
    return ref.rho * ref.gbar * (c.info_base_intensity + slope * coupled)


def run_joint_rrm(
    cfg: Config,
    n_dealers: int,
    kappa: float,
    h0: np.ndarray | float,
    n_steps: int = 60,
    mispricing: Optional[float] = None,
    liquidity_ratio: float = 1.0,
    tol: float = 1e-9,
) -> np.ndarray:
    """Genuine ``N``-dimensional joint best-response cobweb (1.3 §3).

    Iterates ``h^{t+1} = BR(h^t)`` in ``R^N``: each dealer best-responds to the
    coupled toxic level its whole deployed profile induces.  Contracts iff
    ``rho(J_BR) = m_N < 1`` and, when it converges, does so at the joint modulus
    ``m_N`` -- so it slows down (critical slowing down, §8.3) as the market
    approaches the boundary.  Returns an array of shape ``[T+1, N]``.

    This is analytic (it uses 1.1's closed forms, no learned operator), but it is a
    *bona fide* ``N``-body iteration -- it does not collapse to the single-dealer
    reduction -- so it validates the common-mode eigenstructure and the integer
    ``N_c`` crossing directly.
    """
    N = int(n_dealers)
    ref = reference_state(cfg, mispricing=mispricing, liquidity_ratio=liquidity_ratio)
    hi = float(cfg.policy.max_half_spread)
    h = np.full(N, float(h0), dtype=float) if np.isscalar(h0) else np.array(h0, dtype=float)
    traj = [h.copy()]
    for _ in range(int(n_steps)):
        T = _coupled_toxic_levels(cfg, h, kappa, ref)
        h_next = np.array([best_response(cfg, float(T[i]), ref) for i in range(N)], dtype=float)
        traj.append(h_next.copy())
        if not np.all(np.isfinite(h_next)) or np.any(h_next <= 0.0) or np.any(h_next >= hi):
            break
        if float(np.max(np.abs(h_next - h))) < tol:
            break
        h = h_next
    return np.stack(traj, axis=0)


# --------------------------------------------------------------------------- #
# Empirical probes through the full pipeline (1.3 §10)                         #
# --------------------------------------------------------------------------- #
@dataclass
class CommonModeProbe:
    """Empirical in-phase / anti-phase modulus measurements (1.3 §10)."""

    m_N_measured: float  # in-phase (common-mode) BR slope  ~ N_eff*m_1
    m_diff_measured: float  # anti-phase (differential) BR slope  ~ m_1*(1-kappa)
    kappa_identified: float  # kappa recovered from the ratio of the two
    h_ref: float


def measure_common_mode_modulus(
    cfg: Config,
    n_dealers: int,
    kappa: float,
    seed: int = 0,
    h_ref: Optional[float] = None,
    delta: float = 0.25,
    mispricing: Optional[float] = None,
    liquidity_ratio: float = 1.0,
) -> float:
    """Empirical common-mode joint modulus ``m_N`` (1.3 §10, in-phase probe).

    Perturbing all ``N`` deployed spreads in phase and reading the joint best
    response is identical to the single-dealer BR-slope probe on the
    ``N_eff``-amplified effective config, so we run :func:`measure_response_modulus`
    there.  ``h_ref`` defaults to the symmetric PSNE ``h*``.
    """
    Ne = n_eff(n_dealers, kappa)
    cfg_eff = effective_config(cfg, Ne)
    if h_ref is None:
        ref_eff = reference_state(cfg_eff, mispricing=mispricing, liquidity_ratio=liquidity_ratio)
        h_ref = solve_fixed_point(cfg_eff, ref_eff)
    return measure_response_modulus(cfg_eff, seed=seed, h_ref=h_ref, delta=delta).modulus


def measure_differential_modulus(
    cfg: Config,
    n_dealers: int,
    kappa: float,
    seed: int = 0,
    h_ref: Optional[float] = None,
    delta: float = 0.25,
    mispricing: Optional[float] = None,
    liquidity_ratio: float = 1.0,
) -> float:
    """Empirical differential-mode modulus ``m_1*(1-kappa)`` (1.3 §10, anti-phase).

    Perturbing along an eigenvector orthogonal to ``1`` amplifies the toxic slope
    by ``(1 - kappa)`` instead of ``N_eff``; measured at the *same* PSNE ``h*`` as
    the common-mode probe so their ratio identifies ``kappa``.
    """
    Ne = n_eff(n_dealers, kappa)
    if h_ref is None:
        cfg_eff = effective_config(cfg, Ne)
        ref_eff = reference_state(cfg_eff, mispricing=mispricing, liquidity_ratio=liquidity_ratio)
        h_ref = solve_fixed_point(cfg_eff, ref_eff)
    cfg_diff = effective_config(cfg, 1.0 - float(kappa))
    return measure_response_modulus(cfg_diff, seed=seed, h_ref=h_ref, delta=delta).modulus


def identify_kappa(m_common: float, m_diff: float, n_dealers: int) -> float:
    """Recover ``kappa`` from the in-phase / anti-phase modulus ratio (1.3 §10).

    With ``m_common = N_eff*m_1`` and ``m_diff = (1-kappa)*m_1``, the ratio
    ``r = m_diff/m_common = (1-kappa)/(1 + kappa*(N-1))`` inverts to
    ``kappa = (1 - r)/(1 + r*(N-1))`` -- a free calibration of the spillover from
    two measured slopes.
    """
    if m_common <= 0.0:
        return float("nan")
    r = m_diff / m_common
    N = int(n_dealers)
    denom = 1.0 + r * (N - 1)
    return (1.0 - r) / denom if denom != 0.0 else float("nan")


def common_mode_probe(
    cfg: Config,
    n_dealers: int,
    kappa: float,
    seed: int = 0,
    delta: float = 0.25,
    mispricing: Optional[float] = None,
    liquidity_ratio: float = 1.0,
) -> CommonModeProbe:
    """Run both empirical probes and identify ``kappa`` (1.3 §10)."""
    Ne = n_eff(n_dealers, kappa)
    cfg_eff = effective_config(cfg, Ne)
    ref_eff = reference_state(cfg_eff, mispricing=mispricing, liquidity_ratio=liquidity_ratio)
    h_ref = solve_fixed_point(cfg_eff, ref_eff)
    m_common = measure_common_mode_modulus(
        cfg, n_dealers, kappa, seed=seed, h_ref=h_ref, delta=delta,
        mispricing=mispricing, liquidity_ratio=liquidity_ratio,
    )
    m_diff = measure_differential_modulus(
        cfg, n_dealers, kappa, seed=seed, h_ref=h_ref, delta=delta,
        mispricing=mispricing, liquidity_ratio=liquidity_ratio,
    )
    return CommonModeProbe(
        m_N_measured=m_common,
        m_diff_measured=m_diff,
        kappa_identified=identify_kappa(m_common, m_diff, n_dealers),
        h_ref=h_ref,
    )


# --------------------------------------------------------------------------- #
# Phase-diagram row (1.3 §8.2)                                                 #
# --------------------------------------------------------------------------- #
def sweep_dealer_count(
    cfg: Config,
    n_values: List[int],
    kappa: Optional[float] = None,
    mispricing: Optional[float] = None,
    liquidity_ratio: float = 1.0,
    lambda_q: Optional[float] = None,
    reference_spread: Optional[float] = None,
) -> List[MultiDealerBoundary]:
    """Closed-form ``m_N`` vs ``N`` (the systemic phase-diagram row, 1.3 §8.2).

    With ``reference_spread`` held fixed (the structural-regime reading) the joint
    modulus ``m_N = N_eff*m_1`` is exactly linear in ``N`` through the origin with
    slope ``m_1`` -- a one-parameter check of the whole construction -- and crosses
    ``1`` at the integer ``N_c = 1/m_1`` (§8.1).  Left free (``None``), each ``N``
    uses its self-consistent PSNE, so the line bends sub-linearly as competition
    widens the equilibrium spread (defensive widening).
    """
    return [
        multi_dealer_boundary(
            cfg, n_dealers=int(N), kappa=kappa,
            mispricing=mispricing, liquidity_ratio=liquidity_ratio, lambda_q=lambda_q,
            reference_spread=reference_spread,
        )
        for N in n_values
    ]
