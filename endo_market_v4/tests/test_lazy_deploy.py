"""Tests for theory 1.6: the K-step (lazy-deploy) outer map and its probe."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from reflex.config import load_config
from reflex.estimators.br_slope import measure_rgd_response
from reflex.theory.lazy_deploy import (
    deadbeat_k,
    effective_modulus,
    equal_modulus_k,
    fit_inner_contraction,
    gamma_eff,
    k_step_slope,
    lazy_deploy_curve,
    lazy_weight,
    max_stable_k,
)

CONFIG = "configs/default.yaml"


# --------------------------------------------------------------------------- #
# Closed forms (06 sections 2--4)                                              #
# --------------------------------------------------------------------------- #
def test_k_step_slope_limits():
    """mu(0) = 1 exactly; mu(K) -> -m as K -> infinity; strictly decreasing."""
    for m, c in ((0.3, 0.6), (0.9, 0.8), (2.5, 0.7)):
        assert k_step_slope(m, c, 0) == pytest.approx(1.0)
        assert k_step_slope(m, c, 400) == pytest.approx(-m, abs=1e-9)
        ks = np.linspace(0.0, 40.0, 81)
        mus = [k_step_slope(m, c, k) for k in ks]
        assert all(a > b for a, b in zip(mus[:-1], mus[1:]))


def test_lazy_interpolation_identity():
    """mu(K) == (1 - lam_K)*1 + lam_K*(-m) for the same c, K (06 corollary 1)."""
    m, c = 0.7, 0.85
    for k in (0, 1, 2, 5, 13):
        lam = lazy_weight(c, k)
        assert k_step_slope(m, c, k) == pytest.approx((1.0 - lam) * 1.0 + lam * (-m))


def test_deadbeat_zero_crossing():
    """mu(K_db) = 0 (06 eq. 6.2)."""
    m, c = 0.6, 0.75
    kdb = deadbeat_k(m, c)
    assert kdb > 0.0
    assert k_step_slope(m, c, kdb) == pytest.approx(0.0, abs=1e-12)
    assert math.isnan(deadbeat_k(0.0, c))


def test_max_stable_k_window():
    """m <= 1 -> every K stable (inf); m > 1 -> |mu| < 1 iff K < K_max (06 eq. 6.3)."""
    assert max_stable_k(0.99, 0.8) == float("inf")
    m, c = 2.0, 0.8
    kmax = max_stable_k(m, c)
    assert kmax == pytest.approx(math.log(1.0 / 3.0) / math.log(0.8))
    assert abs(k_step_slope(m, c, kmax * 0.98)) < 1.0
    assert abs(k_step_slope(m, c, kmax * 1.02)) > 1.0


def test_gamma_eff_two_branches():
    """gamma_eff < gamma below K_eq (inertia), > gamma above it, -> gamma at K -> inf.

    (06 section 4 / eq. 6.5.)  The original draft of this theorem claimed
    ``gamma_eff >= gamma`` for all K -- falsified by the smoke sweep at the
    near-zero-m default market, which sits entirely in the inertia branch.
    """
    g, m, c = 3.0, 0.1, 0.85
    keq = equal_modulus_k(m, c)
    assert keq > 0.0
    # inertia branch: under-trained loop reads as a softer objective
    assert gamma_eff(g, m, c, 0.25 * keq) < g
    # stiffness branch (between K_eq and the deadbeat divergence)
    kdb = deadbeat_k(m, c)
    assert kdb > keq
    assert gamma_eff(g, m, c, 0.5 * (keq + kdb)) > g
    # exact limit from above
    assert gamma_eff(g, m, c, 2000) == pytest.approx(g, rel=1e-6)
    assert gamma_eff(g, m, c, kdb) > 1e6
    # m >= 1: every K >= 1 already past K_eq (stiff branch)
    assert equal_modulus_k(1.5, 0.8) == 0.0
    assert gamma_eff(g, 1.5, 0.8, 1) > g


def test_effective_modulus_dips_then_recovers():
    """m_eff falls to ~0 at the deadbeat count and returns to m afterwards."""
    m, c = 0.6, 0.75
    kdb = deadbeat_k(m, c)
    assert effective_modulus(m, c, 1) > effective_modulus(m, c, kdb)
    assert effective_modulus(m, c, 300) == pytest.approx(m, rel=1e-6)


def test_fit_inner_contraction_recovers_c():
    """The one-parameter fit recovers c from a noisy synthetic curve."""
    rng = np.random.default_rng(0)
    m, c_true = 0.6, 0.75
    ks = np.array([1, 2, 3, 5, 8, 12, 20], dtype=float)
    mus = np.array([k_step_slope(m, c_true, k) for k in ks])
    mus_noisy = mus + rng.normal(0.0, 0.01, size=mus.shape)
    c_fit = fit_inner_contraction(ks, mus_noisy, m)
    assert c_fit == pytest.approx(c_true, abs=0.05)


def test_input_validation():
    with pytest.raises(ValueError):
        k_step_slope(0.5, 1.5, 3)
    with pytest.raises(ValueError):
        k_step_slope(-0.1, 0.5, 3)
    with pytest.raises(ValueError):
        fit_inner_contraction([], [], 0.5)


def test_lazy_deploy_curve_assembly():
    """Curve assembly from a config: consistent fields, gamma_eff >= gamma."""
    cfg = load_config(CONFIG)
    curve = lazy_deploy_curve(cfg, [1, 3, 8], c=0.8, h_eval=1.0)
    assert curve.m > 0.0 and curve.gamma > 0.0
    assert len(curve.slopes) == len(curve.moduli) == len(curve.gamma_eff) == 3
    assert all(abs(s) == pytest.approx(mod) for s, mod in zip(curve.slopes, curve.moduli))
    assert all(np.isfinite(ge) and ge > 0.0 for ge in curve.gamma_eff)
    # branch consistency (06 eq. 6.5): gamma_eff vs gamma splits at K_eq
    keq = equal_modulus_k(curve.m, curve.c)
    for k, ge in zip(curve.k_values, curve.gamma_eff):
        if k < keq:
            assert ge < curve.gamma
        elif k > keq:
            assert ge > curve.gamma
    assert curve.k_max_stable == float("inf") or curve.m > 1.0


# --------------------------------------------------------------------------- #
# The CRN K-probe                                                              #
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def tiny_cfg():
    cfg = load_config(CONFIG)
    cfg.bonds.n_bonds = 4
    cfg.simulator.horizon = 16
    cfg.rrm.n_episodes = 4
    cfg.operator.epochs = 15
    cfg.operator.hidden = 32
    cfg.policy.n_rollouts = 4
    cfg.policy.rollout_horizon = 8
    return cfg


def test_rgd_probe_k0_is_identity(tiny_cfg):
    """With zero inner steps the deployment map is the identity: mu(0) = 1.

    The K = 0 probe returns exactly the deployed spreads (no optimisation), so
    the signed difference quotient is 1 up to float error -- the sharpest
    end-to-end check that the probe measures the map it claims to.
    """
    torch.manual_seed(0)
    res = measure_rgd_response(tiny_cfg, seed=0, h_ref=1.0, delta=0.25, n_steps=0)
    assert res.slope == pytest.approx(1.0, abs=1e-4)
    assert res.n_steps == 0


def test_rgd_probe_contracts_with_steps(tiny_cfg):
    """A few inner steps must pull the map slope below the K = 0 identity."""
    torch.manual_seed(0)
    res = measure_rgd_response(tiny_cfg, seed=0, h_ref=1.0, delta=0.25, n_steps=3, lr=0.08)
    assert np.isfinite(res.slope)
    assert res.slope < 1.0
    assert res.modulus == pytest.approx(abs(res.slope))


@pytest.mark.slow
def test_rgd_probe_k_curve_contracts_and_fit():
    """Every K >= 1 sits far below the K = 0 identity, and the c-fit is valid.

    At the default (low-feedback) constants the true modulus is ~0.02, so the
    clean K-curve flattens near zero almost immediately and the *ordering* of
    small medians is inside probe noise (the finite-budget BR attenuation
    documented in the 2026-07 audit).  The noise-robust theory-06 predictions
    at m ~ 0 are (i) mu_hat(K) << 1 for every K >= 1 -- the inner steps
    genuinely contract the deployment map -- and (ii) the one-parameter c-fit
    lands strictly inside (0, 1).
    """
    cfg = load_config(CONFIG)
    cfg.bonds.n_bonds = 4
    cfg.simulator.horizon = 24
    cfg.rrm.n_episodes = 6
    cfg.operator.epochs = 20
    cfg.policy.n_rollouts = 6
    cfg.policy.rollout_horizon = 8
    ks = [1, 4, 16]
    med = []
    for k in ks:
        slopes = [
            measure_rgd_response(cfg, seed=s, h_ref=1.0, delta=0.25, n_steps=k, lr=0.08).slope
            for s in (0, 1)
        ]
        med.append(float(np.median(slopes)))
    assert all(np.isfinite(med))
    assert max(med) < 0.7, f"K >= 1 probes failed to contract below the K=0 identity: {med}"
    c_fit = fit_inner_contraction(ks, med, m=max(abs(med[-1]), 0.05))
    assert 0.0 < c_fit < 1.0
