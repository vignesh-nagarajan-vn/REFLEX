"""Numerical proof certificates for theory 1.1-1.6 (v4 verification layer).

Each ``certify_*`` function re-derives the load-bearing content of one
derivation *numerically* -- finite-difference slopes against closed-form
slopes, spectral identities against eigensolves, Monte-Carlo rates against
the claimed exponents, dynamical claims by running the dynamics -- and
returns a :class:`Certificate` listing every named check with its residual
and tolerance.  A certificate passes only if every check does.

These are not unit tests of code paths (the test suite does that); they are
machine-checkable *justifications of the mathematics as implemented*: if a
derivation document claims ``BR'(h*) = -epsilon*beta/gamma``, the certificate
measures ``BR'`` with no reference to ``epsilon``, ``beta`` or ``gamma`` and
compares.  Run them on any config via ``experiments/run_certificates.py``.

The formal (Lean 4) companions of the *logical* skeletons live in
``endo_market_v4/lean/`` -- see its README for scope and compile status.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from ..config import Config
from ..theory.analytic_boundary import (
    analytic_boundary,
    best_response,
    blind_gradient,
    epsilon as epsilon_of,
    gamma as gamma_of,
    reference_state,
    solve_fixed_point,
    tau as tau_of,
)
from ..theory.lazy_deploy import (
    deadbeat_k,
    equal_modulus_k,
    gamma_eff,
    k_step_slope,
    lazy_weight,
    max_stable_k,
)
from ..theory.multi_dealer import joint_jacobian, multi_dealer_boundary, n_eff
from ..theory.perfgd import (
    gamma_po,
    perfgd_correction,
    perfgd_gradient,
    run_perfgd,
    run_rrm_cobweb,
    solve_performative_optimum,
)
from ..theory.robust import (
    calibrate_radius,
    empirical_radius,
    loglog_rate,
    robust_certificate,
    sample_complexity,
)


@dataclass
class Check:
    """One named check inside a certificate."""

    name: str
    value: float  # measured residual / quantity
    tolerance: float  # pass iff |value| <= tolerance (or bool-as-0/1)
    passed: bool


@dataclass
class Certificate:
    """Verdict for one derivation on one config."""

    name: str
    passed: bool
    checks: List[Check] = field(default_factory=list)

    def summary(self) -> str:
        mark = "PASS" if self.passed else "FAIL"
        worst = max(self.checks, key=lambda c: (not c.passed, abs(c.value)))
        return f"[{mark}] {self.name} ({len(self.checks)} checks; worst: {worst.name})"


def _mk(name: str, checks: List[Check]) -> Certificate:
    return Certificate(name=name, passed=all(c.passed for c in checks), checks=checks)


def _resid(name: str, value: float, tol: float) -> Check:
    v = float(value)
    return Check(name=name, value=v, tolerance=float(tol), passed=abs(v) <= tol)


def _flag(name: str, ok: bool) -> Check:
    return Check(name=name, value=0.0 if ok else 1.0, tolerance=0.5, passed=bool(ok))


# --------------------------------------------------------------------------- #
# 1.1 -- the analytic stability boundary                                       #
# --------------------------------------------------------------------------- #
def certify_boundary(cfg: Config) -> Certificate:
    """1.1: the modulus *is* the best-response slope, measured independently.

    The theorem's content is ``BR'(h*) = -epsilon*beta/gamma``: the certificate
    measures the left side by central finite differences of the frozen-``T``
    best response through :func:`best_response` (no reference to the three
    constants) and compares against the assembled ``-m``.  Plus: ``h*`` zeroes
    the blind gradient, ``gamma > 0`` in the operating regime, and one linear
    cobweb step contracts by ``m`` to first order.
    """
    ref = reference_state(cfg)
    ab = analytic_boundary(cfg)
    h = ab.h_star
    scale = max(abs(blind_gradient(cfg, 0.5 * h, ref)), 1e-6)

    checks = [
        _resid("fixed_point_zeroes_blind_gradient",
               blind_gradient(cfg, h, ref) / scale, 1e-6),
        _resid("modulus_identity_m_eq_eps_beta_over_gamma",
               ab.modulus - ab.epsilon * ab.beta / ab.gamma, 1e-12),
        _flag("gamma_positive_at_fixed_point", ab.gamma > 0.0),
    ]

    # BR'(h*) by central differences of the *composed* map h -> BR(tau(h)).
    # Convention note (a finding of this certificate, cf. 1.1 §7): the 1-D
    # frozen-gradient helper omits the inventory-risk term, so the slope of
    # *this* map is -eps*beta/(gamma - P*lambda_q); the full-pipeline modulus
    # m = eps*beta/gamma includes lambda_q.  Both sides of the identity must
    # use the same convention -- here lambda_q = 0.
    m0 = ab.epsilon * ab.beta / gamma_of(cfg, h, ref, lambda_q=0.0)
    d = 1e-3 * max(h, 1.0)
    br_plus = best_response(cfg, tau_of(cfg, h + d, ref), ref)
    br_minus = best_response(cfg, tau_of(cfg, h - d, ref), ref)
    br_slope = (br_plus - br_minus) / (2.0 * d)
    checks.append(_resid("br_slope_numeric_vs_minus_m_lq0",
                         (br_slope + m0) / max(m0, 1e-3), 3e-2))

    # One cobweb step from a small kick contracts by ~m0 (signed).
    kick = 0.05 * max(h, 1.0)
    h1 = best_response(cfg, tau_of(cfg, h + kick, ref), ref)
    lin = (h1 - h) / (-m0 * kick) - 1.0 if m0 > 1e-6 else 0.0
    checks.append(_resid("cobweb_step_linearisation", lin, 8e-2))
    return _mk("1.1 analytic boundary", checks)


# --------------------------------------------------------------------------- #
# 1.2 -- the PerfGD correction                                                 #
# --------------------------------------------------------------------------- #
def _unstable_demo(cfg: Config) -> Config:
    """The genuinely RRM-unstable regime used across the 1.2 artifacts."""
    import copy

    demo = copy.deepcopy(cfg)
    demo.clients.alpha = 1.0
    demo.clients.toxicity_feedback = 6.0
    demo.clients.info_intensity = 3.0
    demo.clients.info_spread_decay = 0.8
    demo.reward.quote_anchor_weight = 0.15
    return demo


def certify_perfgd(cfg: Config) -> Certificate:
    """1.2 identities (config-generic): stationarity of ``h_PO``, the sign
    flip of the correction at ``psi`` (probed at a *scale-relative* offset),
    and the curvature identity ``gamma_PO = -Phi''(h_PO)`` (measured by
    differencing ``Phi'``)."""
    ref = reference_state(cfg)
    h_po = solve_performative_optimum(cfg, ref)
    scale = max(abs(perfgd_gradient(cfg, 0.5 * h_po, ref)), 1e-6)
    off = 0.2 * max(ref.psi, 0.1 * h_po)  # scale-relative probe offset
    checks = [
        _resid("optimum_zeroes_corrected_gradient",
               perfgd_gradient(cfg, h_po, ref) / scale, 1e-6),
        _resid("correction_vanishes_at_psi",
               perfgd_correction(cfg, ref.psi, ref) / scale, 1e-9),
        _flag("correction_pulls_tighter_above_psi",
              perfgd_correction(cfg, ref.psi + off, ref) < 0.0
              < perfgd_correction(cfg, max(ref.psi - off, 1e-6), ref)),
    ]
    # Same lambda_q convention note as certify_boundary: Phi' is built on the
    # 1-D frozen gradient (no inventory term), so its curvature identity holds
    # with gamma evaluated at lambda_q = 0.
    d = 1e-3 * max(h_po, 1.0)
    curv = -(perfgd_gradient(cfg, h_po + d, ref) - perfgd_gradient(cfg, h_po - d, ref)) / (2 * d)
    g_po = gamma_po(cfg, h_po, ref, lambda_q=0.0)
    checks.append(_resid("gamma_po_curvature_identity_lq0",
                         (curv - g_po) / max(abs(g_po), 1e-6), 2e-2))
    return _mk("1.2 PerfGD identities", checks)


def certify_perfgd_dynamics(cfg: Config) -> Certificate:
    """1.2 beyond-boundary dynamics, in the *canonical raw-unit demo regime*.

    The demo constants (feedback 6, intensity 3, decay 0.8, anchor weight
    0.15) are absolute raw-unit values -- the regime every 1.2 artifact and
    test uses -- so this certificate is only meaningful on raw-unit configs
    and is excluded from calibrated (real-unit) runs by
    :func:`run_all_certificates` (imposing absolute spread constants on a
    calibrated config is exactly the unit bug the repo conventions forbid).
    Checks: the demo fixed point is genuinely RRM-unstable, the cobweb fails
    to settle there, the corrected ascent reaches ``h_PO``, and the
    echo-chamber direction ``h_SP > h_PO`` holds.
    """
    demo = _unstable_demo(cfg)
    ref_d = reference_state(demo)
    ab_d = analytic_boundary(demo)
    h_po_d = solve_performative_optimum(demo, ref_d)
    start = ab_d.h_star + 0.1
    cob = run_rrm_cobweb(demo, ref_d, start, n_steps=80)
    corr = run_perfgd(demo, ref_d, start, n_steps=200)
    checks = [
        _flag("demo_regime_is_beyond_boundary", ab_d.modulus > 1.0),
        _flag("cobweb_diverges_beyond_boundary",
              abs(cob[-1] - ab_d.h_star) > abs(cob[0] - ab_d.h_star)
              or len(cob) < 80),
        _resid("corrected_ascent_reaches_h_po",
               (corr[-1] - h_po_d) / max(h_po_d, 1e-6), 1e-3),
        _flag("echo_chamber_direction", ab_d.h_star > h_po_d),
    ]
    return _mk("1.2 PerfGD dynamics (raw-unit demo)", checks)


# --------------------------------------------------------------------------- #
# 1.3 -- multi-dealer systemic risk                                            #
# --------------------------------------------------------------------------- #
def certify_multi_dealer(cfg: Config, n_dealers: int = 3, kappa: float = 0.6) -> Certificate:
    """1.3: the ``N_eff`` formula, the common-mode eigen-identity of the joint
    Jacobian (measured by an eigensolve, compared against ``-m_1*N_eff``),
    the ``N = 1`` reduction to 1.1, and the boundary identity."""
    checks = [
        _resid("n_eff_formula", n_eff(n_dealers, kappa) - (1.0 + kappa * (n_dealers - 1)), 1e-12),
    ]
    m1 = 0.4
    J = joint_jacobian(m1, n_dealers, kappa)
    ones = np.ones(n_dealers)
    resid_vec = J @ ones - (-m1 * n_eff(n_dealers, kappa)) * ones
    checks.append(_resid("jacobian_common_mode_eigenvalue",
                         float(np.abs(resid_vec).max()), 1e-12))
    eigs = np.linalg.eigvals(J)
    checks.append(_resid("spectral_radius_is_common_mode",
                         float(np.max(np.abs(eigs))) - m1 * n_eff(n_dealers, kappa), 1e-9))

    ab = analytic_boundary(cfg)
    mdb1 = multi_dealer_boundary(cfg, n_dealers=1, kappa=0.0)
    checks.append(_resid("single_dealer_reduction",
                         (mdb1.m_N - ab.modulus) / max(ab.modulus, 1e-9), 1e-9))
    mdb = multi_dealer_boundary(cfg, n_dealers=n_dealers, kappa=kappa)
    checks.append(_resid("boundary_identity_eps_star",
                         (mdb.boundary_epsilon - mdb.gamma / (mdb.n_eff * mdb.beta))
                         / max(mdb.boundary_epsilon, 1e-9), 1e-9))
    checks.append(_resid("amplification_identity_mN",
                         (mdb.m_N - mdb.n_eff * mdb.m_1) / max(mdb.m_N, 1e-9), 1e-9))
    return _mk("1.3 multi-dealer", checks)


# --------------------------------------------------------------------------- #
# 1.4 -- robust boundary                                                       #
# --------------------------------------------------------------------------- #
def certify_robust(seed: int = 0) -> Certificate:
    """1.4: the radius formula, the parametric Monte-Carlo rate (log-log slope
    ``-1/2`` for a mean-of-``n`` estimator -- the rate the CRN coupling buys),
    certificate trichotomy on constructed cases, sample-complexity divergence,
    and the calibrated radius floor."""
    rng = np.random.default_rng(seed)
    est = rng.normal(0.5, 0.1, size=64)
    mean, std, rad = empirical_radius(est, confidence=0.95)
    from scipy.stats import norm

    checks = [
        _resid("radius_formula_z_times_s",
               rad - float(norm.ppf(0.95)) * std, 1e-12),
    ]
    ns = [64, 256, 1024, 4096]
    stds = []
    for n in ns:
        reps = np.array([rng.normal(0.0, 1.0, size=n).mean() for _ in range(200)])
        stds.append(float(reps.std(ddof=1)))
    slope = loglog_rate(ns, stds)
    checks.append(_resid("parametric_rate_slope_minus_half", slope + 0.5, 0.1))

    checks.append(_flag("trichotomy_stable",
                        robust_certificate(0.5, 0.1).verdict == "stable"))
    checks.append(_flag("trichotomy_undecided",
                        robust_certificate(0.95, 0.1).verdict == "undecided"))
    checks.append(_flag("trichotomy_unstable",
                        robust_certificate(1.5, 0.1).verdict == "unstable"))
    checks.append(_flag("sample_complexity_diverges",
                        sample_complexity(0.1, 1e-9) > sample_complexity(0.1, 0.1) > 0.0
                        and math.isinf(sample_complexity(0.1, 0.0))))
    cal = calibrate_radius(est, confidence=0.95, n_boot=400, seed=seed)
    checks.append(_flag("calibrated_radius_never_tighter_than_z_s",
                        cal.calibrated_radius >= cal.z_radius - 1e-12))
    return _mk("1.4 robust boundary", checks)


# --------------------------------------------------------------------------- #
# 1.5 -- factor-model scaling                                                  #
# --------------------------------------------------------------------------- #
def certify_factor_scaling(cfg: Config, k_max: int = 4) -> Certificate:
    """1.5: the Woodbury spectral radius equals the dense eigensolve, the
    truncation bound dominates the measured truncation error for every
    ``k``, and the residual variance decreases in ``k``."""
    import torch

    from ..env.bonds import BondUniverse
    from ..theory.factor_scaling import (
        factor_modulus,
        modulus_matrix,
        per_bond_constants,
        spectral_radius,
        truncation_error_bound,
    )

    torch.manual_seed(0)
    universe = BondUniverse(cfg.bonds, seed=0)
    fm = factor_modulus(cfg, universe=universe)
    pbc = per_bond_constants(cfg, universe)
    M = modulus_matrix(cfg, pbc, universe)
    rho_dense = spectral_radius(M)
    checks = [
        _resid("assembled_rho_equals_dense_eigensolve",
               (fm.rho - rho_dense) / max(rho_dense, 1e-12), 1e-9),
    ]
    ks = list(range(1, min(int(k_max), int(cfg.bonds.n_bonds) - 1) + 1))
    prev_res = float("inf")
    bound_holds = True
    res_monotone = True
    for k in ks:
        tb = truncation_error_bound(cfg, universe, k=k)
        if tb.rho_error_measured > tb.m_error_bound + 1e-12:
            bound_holds = False
        if tb.residual_variance > prev_res + 1e-12:
            res_monotone = False
        prev_res = tb.residual_variance
    checks.append(_flag("truncation_bound_dominates_measured_error", bound_holds))
    checks.append(_flag("residual_variance_monotone_in_k", res_monotone))
    return _mk("1.5 factor scaling", checks)


# --------------------------------------------------------------------------- #
# 1.6 -- lazy deployment                                                       #
# --------------------------------------------------------------------------- #
def certify_lazy_deploy() -> Certificate:
    """1.6: the boundary values, the interpolation identity, the deadbeat and
    max-stable roots, and the two-branch structure of ``gamma_eff``."""
    m, c = 0.6, 0.75
    checks = [
        _resid("mu_at_zero_is_one", k_step_slope(m, c, 0) - 1.0, 1e-12),
        _resid("mu_limit_is_minus_m", k_step_slope(m, c, 500) + m, 1e-9),
        _resid("interpolation_identity",
               k_step_slope(m, c, 7)
               - ((1.0 - lazy_weight(c, 7)) * 1.0 + lazy_weight(c, 7) * (-m)), 1e-12),
        _resid("deadbeat_zeroes_mu", k_step_slope(m, c, deadbeat_k(m, c)), 1e-12),
    ]
    m_u = 2.0
    kmax = max_stable_k(m_u, c)
    checks.append(_flag("max_stable_k_boundary",
                        abs(k_step_slope(m_u, c, kmax * 0.98)) < 1.0
                        < abs(k_step_slope(m_u, c, kmax * 1.02))))
    g = 3.0
    keq = equal_modulus_k(m, c)
    checks.append(_flag("gamma_eff_two_branches",
                        gamma_eff(g, m, c, 0.5 * keq) < g < gamma_eff(
                            g, m, c, 0.5 * (keq + deadbeat_k(m, c)))))
    return _mk("1.6 lazy deployment", checks)


# --------------------------------------------------------------------------- #
# Assembly                                                                     #
# --------------------------------------------------------------------------- #
def run_all_certificates(
    cfg: Config, seed: int = 0, include_dynamics: Optional[bool] = None
) -> List[Certificate]:
    """Run every certificate against ``cfg`` (1.4/1.6 are config-free).

    ``include_dynamics`` controls the raw-unit beyond-boundary demo
    (:func:`certify_perfgd_dynamics`); the default ``None`` includes it
    exactly when the config is *not* calibrated -- its demo constants are
    absolute raw-unit values that must not be imposed on real-unit configs.
    """
    if include_dynamics is None:
        include_dynamics = not bool(cfg.calibration.enabled)
    certs = [
        certify_boundary(cfg),
        certify_perfgd(cfg),
        certify_multi_dealer(cfg),
        certify_robust(seed=seed),
        certify_factor_scaling(cfg),
        certify_lazy_deploy(),
    ]
    if include_dynamics:
        certs.insert(2, certify_perfgd_dynamics(cfg))
    return certs
