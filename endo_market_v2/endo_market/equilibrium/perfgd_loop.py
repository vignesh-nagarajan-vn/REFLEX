"""The PerfGD-corrected policy loop -- un-blinding the operator (math-theory 1.2).

Blind repeated retraining (RRM) finds the **stable point** ``h_SP``: the spread
that is its own best response *given* the toxic flow it induces.  It is blind to
the fact that its own quoting *moves* that flow, so it is not the maximiser of the
true objective.  PerfGD (Izzo et al., 2021) adds exactly the missing total-
derivative term.  Because
``research/math-theory/01-analytic-stability-boundary.md`` supplies the
distribution response ``d tau/d h = -epsilon(h)`` in closed form, the correction

    Delta(h) = dJ/dT * (d tau/d h) = -beta * (h - psi) * epsilon(h)         (1.2 §2)

is analytic -- **no estimation of ``dD/dphi`` is required**, in contrast to the
finite-difference PerfGD of Izzo et al.  The corrected ascent direction is

    Phi'(h) = G(h) + Delta(h)                                              (1.2 §2)

with ``G`` the blind gradient from :mod:`analytic_boundary`.  Its root is the
**performative optimum** ``h_PO``.

The headline theorem (1.2 §5): PerfGD's stability is governed by the objective
curvature ``gamma_PO = gamma + beta*epsilon*(2 + c_t*psi - c_t*h_PO)``, not by the
cobweb modulus ``m = epsilon*beta/gamma``.  In the operating regime
``c_t*h_PO < 2 + c_t*psi`` the correction only *adds* curvature, so
``gamma_PO > gamma > 0`` for every ``epsilon`` and **PerfGD converges even for
``epsilon`` beyond the RRM boundary ``epsilon* = gamma/beta`` where the blind
cobweb diverges**.

This module implements the correction as exact 1-D dynamics on the central
half-spread ``h`` (the dominant coordinate of the iterate map; 1.1 §1) -- the
concrete realisation of "PerfGD run with the exact analytic ``dD/dphi``".  It
provides:

* :func:`perfgd_correction` / :func:`perfgd_gradient` -- the scalar correction and
  the corrected ascent direction, ready to add to any policy-gradient step;
* :func:`solve_performative_optimum` -- ``h_PO`` (root of ``Phi'``);
* :func:`echo_chamber_gap` -- the decision gap (6a, ``O(epsilon)``) and value gap
  (6b, ``O(epsilon^2)``) between the stable point and the optimum;
* :func:`run_rrm_cobweb` / :func:`run_perfgd` -- the blind best-response cobweb and
  the corrected gradient loop, so the "stable beyond ``epsilon*``" claim is
  directly demonstrable; and
* :func:`analyze_perfgd` -- one call assembling all of the above.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional

from ..analysis.analytic_boundary import (
    AnalyticBoundary,
    ReferenceState,
    analytic_boundary,
    best_response,
    beta as beta_of,
    blind_gradient,
    epsilon as epsilon_of,
    gamma as gamma_of,
    reference_state,
    solve_fixed_point,
    tau as tau_of,
)
from ..config import Config


# --------------------------------------------------------------------------- #
# The analytic PerfGD correction (1.2 §2)                                     #
# --------------------------------------------------------------------------- #
def perfgd_correction(cfg: Config, h: float, ref: ReferenceState) -> float:
    """Performative correction ``Delta(h) = -beta*(h - psi)*epsilon(h)`` (1.2 §2).

    Positive toxic flow is worth the half-spread ``h`` but costs ``psi`` to adverse
    selection, so ``dJ/dT = beta*(h - psi)``; multiplying by the flow response
    ``d tau/d h = -epsilon(h)`` gives the correction.  It vanishes at ``h = psi``
    (toxic flow value-neutral) and flips sign there: it pulls toward *tighter*
    quotes when ``h > psi`` (chase profitable summoned flow) and *wider* quotes
    when ``h < psi`` (suppress toxic summoned flow) -- corollary 1.2 §7.1.
    """
    return -beta_of(cfg) * (h - ref.psi) * epsilon_of(cfg, h, ref)


def perfgd_gradient(cfg: Config, h: float, ref: ReferenceState) -> float:
    """Corrected ascent direction ``Phi'(h) = G(h) + Delta(h)`` (1.2 §2)."""
    return blind_gradient(cfg, h, ref) + perfgd_correction(cfg, h, ref)


def solve_performative_optimum(cfg: Config, ref: ReferenceState) -> float:
    """Locate the performative optimum ``h_PO`` (root of ``Phi'``; 1.2 §1)."""
    from ..analysis.analytic_boundary import _bracketed_root

    hi = float(cfg.policy.max_half_spread)
    return _bracketed_root(lambda h: perfgd_gradient(cfg, h, ref), 0.0, hi)


def gamma_po(cfg: Config, h_po: float, ref: ReferenceState, lambda_q: Optional[float] = None) -> float:
    """Objective curvature at the optimum ``gamma_PO`` (1.2 §4.1).

    ``gamma_PO = gamma + beta*epsilon(h_PO)*(2 + c_t*psi - c_t*h_PO)``.  This -- not
    the cobweb modulus -- governs PerfGD convergence.  It exceeds ``gamma`` (the
    correction is purely stabilising) whenever ``c_t*h_PO < 2 + c_t*psi``.
    """
    c_t = float(cfg.clients.info_spread_decay)
    g = gamma_of(cfg, h_po, ref, lambda_q=lambda_q)
    b = beta_of(cfg)
    eps = epsilon_of(cfg, h_po, ref)
    return g + b * eps * (2.0 + c_t * ref.psi - c_t * h_po)


# --------------------------------------------------------------------------- #
# The echo-chamber gap (1.2 §6)                                               #
# --------------------------------------------------------------------------- #
@dataclass
class EchoChamberGap:
    """Stable-vs-optimal quote and value gaps (1.2 §6)."""

    h_sp: float  # stable point (RRM fixed point)
    h_po: float  # performative optimum
    decision_gap: float  # h_SP - h_PO, leading order O(epsilon)   (6a)
    decision_gap_exact: float  # h_SP - h_PO from the two exact root-finds
    value_gap: float  # Phi(h_PO) - Phi(h_SP), O(epsilon^2)        (6b)


def echo_chamber_gap(
    cfg: Config,
    ref: ReferenceState,
    h_sp: Optional[float] = None,
    h_po: Optional[float] = None,
    lambda_q: Optional[float] = None,
) -> EchoChamberGap:
    """Compute the decision gap (6a) and value gap (6b) of 1.2 §6.

    The decision gap is reported both as the leading-order linearisation
    ``beta*epsilon*(h_SP - psi)/(gamma + beta*epsilon)`` (``O(epsilon)``) and as the
    exact difference of the two root-finds; the value gap uses the exact decision
    gap and ``gamma_PO`` (``O(epsilon^2)``).  Where toxic flow is net profitable at
    the stable spread (``h_SP > psi``) the stable point quotes *wider* than the
    optimum -- the dealer over-defends because it never credits itself for the flow
    its own tightening would summon.
    """
    if h_sp is None:
        h_sp = solve_fixed_point(cfg, ref)
    if h_po is None:
        h_po = solve_performative_optimum(cfg, ref)
    g = gamma_of(cfg, h_sp, ref, lambda_q=lambda_q)
    b = beta_of(cfg)
    eps = epsilon_of(cfg, h_sp, ref)
    linear = b * eps * (h_sp - ref.psi) / (g + b * eps)
    exact = h_sp - h_po
    g_po = gamma_po(cfg, h_po, ref, lambda_q=lambda_q)
    value = 0.5 * g_po * exact * exact
    return EchoChamberGap(
        h_sp=h_sp,
        h_po=h_po,
        decision_gap=linear,
        decision_gap_exact=exact,
        value_gap=value,
    )


# --------------------------------------------------------------------------- #
# The two loops (1.2 §3, §5)                                                  #
# --------------------------------------------------------------------------- #
def run_rrm_cobweb(
    cfg: Config,
    ref: ReferenceState,
    h0: float,
    n_steps: int = 60,
    tol: float = 1e-8,
) -> List[float]:
    """Blind best-response cobweb ``h_{k+1} = BR(h_k)`` (RRM; 1.1 §3, 1.2 §3).

    Each step best-responds to the toxic level induced by the *previous*
    deployment, ``BR(h_k) = argmax_h J(h; tau(h_k))``.  The iteration contracts iff
    ``m = epsilon*beta/gamma < 1`` and oscillates outward (diverges) past the
    boundary.  Returns the trajectory ``[h0, h1, ...]``, stopping early on
    convergence or on leaving the operating range ``[0, max_half_spread]``.
    """
    hi = float(cfg.policy.max_half_spread)
    traj = [float(h0)]
    h = float(h0)
    for _ in range(int(n_steps)):
        h_next = best_response(cfg, tau_of(cfg, h, ref), ref)
        traj.append(h_next)
        if not math.isfinite(h_next) or h_next <= 0.0 or h_next >= hi:
            break
        if abs(h_next - h) < tol:
            break
        h = h_next
    return traj


def run_perfgd(
    cfg: Config,
    ref: ReferenceState,
    h0: float,
    eta: Optional[float] = None,
    n_steps: int = 200,
    tol: float = 1e-8,
    lambda_q: Optional[float] = None,
) -> List[float]:
    """Corrected gradient ascent ``h_{k+1} = h_k + eta*Phi'(h_k)`` (PerfGD; 1.2 §3).

    Ascends the *true* objective ``Phi``.  If ``eta`` is ``None`` it defaults to
    ``1/gamma_PO`` (the near-optimal step for the local curvature; 1.2 §4.2), which
    keeps the loop stable in the operating regime for arbitrarily large
    ``epsilon``.  Returns the trajectory ``[h0, h1, ...]``, clamped to
    ``[0, max_half_spread]``.
    """
    hi = float(cfg.policy.max_half_spread)
    if eta is None:
        h_po = solve_performative_optimum(cfg, ref)
        g_po = gamma_po(cfg, h_po, ref, lambda_q=lambda_q)
        eta = 1.0 / g_po if g_po > 1e-9 else 1e-2
    traj = [float(h0)]
    h = float(h0)
    for _ in range(int(n_steps)):
        h_next = h + eta * perfgd_gradient(cfg, h, ref)
        h_next = min(max(h_next, 0.0), hi)
        traj.append(h_next)
        if abs(h_next - h) < tol:
            break
        h = h_next
    return traj


# --------------------------------------------------------------------------- #
# Assembled analysis                                                          #
# --------------------------------------------------------------------------- #
@dataclass
class PerfGDResult:
    """Full PerfGD analysis for one config (math-theory 1.2)."""

    boundary: AnalyticBoundary  # the blind 1.1 boundary (h_SP, gamma, beta, m, ...)
    h_sp: float  # stable point
    h_po: float  # performative optimum
    gamma: float  # strong convexity at h_SP
    beta: float  # joint smoothness
    gamma_po: float  # objective curvature at h_PO (governs PerfGD stability)
    modulus_rrm: float  # m = epsilon*beta/gamma at h_SP (governs RRM stability)
    epsilon_star: float  # RRM boundary  epsilon* = gamma/beta
    rrm_stable: bool  # m < 1
    perfgd_strongly_concave: bool  # gamma_PO > 0
    gap: EchoChamberGap  # echo-chamber decision/value gaps
    rrm_trajectory: List[float] = field(default_factory=list)
    perfgd_trajectory: List[float] = field(default_factory=list)
    rrm_converged: bool = False
    perfgd_converged: bool = False


def analyze_perfgd(
    cfg: Config,
    mispricing: Optional[float] = None,
    liquidity_ratio: float = 1.0,
    lambda_q: Optional[float] = None,
    run_loops: bool = True,
    h0: Optional[float] = None,
    n_steps: int = 120,
    eta: Optional[float] = None,
    conv_tol: float = 1e-4,
) -> PerfGDResult:
    """Assemble the full 1.2 analysis for ``cfg``.

    Computes the stable point, the performative optimum, the two curvatures
    (``gamma`` vs ``gamma_PO``), the RRM modulus and boundary, and the echo-chamber
    gap.  When ``run_loops`` is ``True`` it also runs the blind cobweb and the
    corrected loop from a common start ``h0`` (default: the RRM boundary ``h_SP``
    plus a small kick), so a caller can confirm the headline claim -- RRM diverges
    and PerfGD converges -- for ``epsilon`` beyond ``epsilon*``.
    """
    ref = reference_state(cfg, mispricing=mispricing, liquidity_ratio=liquidity_ratio)
    boundary = analytic_boundary(
        cfg, mispricing=mispricing, liquidity_ratio=liquidity_ratio, lambda_q=lambda_q
    )
    h_sp = boundary.h_star
    h_po = solve_performative_optimum(cfg, ref)
    g = boundary.gamma
    b = boundary.beta
    g_po = gamma_po(cfg, h_po, ref, lambda_q=lambda_q)
    gap = echo_chamber_gap(cfg, ref, h_sp=h_sp, h_po=h_po, lambda_q=lambda_q)

    rrm_traj: List[float] = []
    perf_traj: List[float] = []
    rrm_conv = False
    perf_conv = False
    if run_loops:
        start = float(h_sp + 0.1 * max(boundary.ref.sigma_s, 0.1)) if h0 is None else float(h0)
        rrm_traj = run_rrm_cobweb(cfg, ref, start, n_steps=n_steps)
        perf_traj = run_perfgd(cfg, ref, start, eta=eta, n_steps=n_steps, lambda_q=lambda_q)
        rrm_conv = len(rrm_traj) >= 2 and abs(rrm_traj[-1] - h_sp) < conv_tol
        perf_conv = len(perf_traj) >= 2 and abs(perf_traj[-1] - h_po) < conv_tol

    return PerfGDResult(
        boundary=boundary,
        h_sp=h_sp,
        h_po=h_po,
        gamma=g,
        beta=b,
        gamma_po=g_po,
        modulus_rrm=boundary.modulus,
        epsilon_star=boundary.boundary_epsilon,
        rrm_stable=boundary.stable,
        perfgd_strongly_concave=g_po > 0.0,
        gap=gap,
        rrm_trajectory=rrm_traj,
        perfgd_trajectory=perf_traj,
        rrm_converged=rrm_conv,
        perfgd_converged=perf_conv,
    )
