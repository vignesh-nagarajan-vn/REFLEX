"""Tests for the v4 estimator tuning: Sinkhorn blur + robust ambiguity radius."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from reflex.config import load_config
from reflex.estimators.sinkhorn import (
    TUNED_REL_REG,
    estimate_epsilon_sinkhorn,
    quantile_w1,
    sinkhorn_divergence,
    tune_sinkhorn_reg,
)
from reflex.theory.robust import calibrate_radius, robust_boundary

CONFIG = "configs/default.yaml"


# --------------------------------------------------------------------------- #
# Sinkhorn blur tuning                                                         #
# --------------------------------------------------------------------------- #
def test_tune_sinkhorn_synthetic_recovers_w1():
    """On a location-shift pair the tuned blur reproduces the exact W1 closely,
    and the largest blur on the grid is measurably worse."""
    rng = np.random.default_rng(0)
    x = rng.normal(0.0, 1.0, size=1000)
    y = rng.normal(0.3, 1.0, size=1000)
    tun = tune_sinkhorn_reg(x, y)
    assert tun.w1_exact == pytest.approx(0.3, abs=0.08)
    assert tun.best_rel_bias < 0.10
    assert tun.rel_bias[-1] > tun.best_rel_bias
    assert tun.best_abs_reg == pytest.approx(tun.best_rel_reg * tun.scale)


def test_tune_sinkhorn_scale_invariance():
    """The relative-blur bias curve is invariant to rescaling the samples.

    This is the property that makes a *relative* tuned default legitimate on
    calibrated real-unit configs: the entropic OT problem is positively
    homogeneous, so blur/scale is the right coordinate.
    """
    rng = np.random.default_rng(1)
    x = rng.normal(0.0, 1.0, size=400)
    y = rng.normal(0.25, 1.0, size=400)
    grid = [0.05, 0.2, 0.5, 1.0]
    a = tune_sinkhorn_reg(x, y, rel_grid=grid)
    b = tune_sinkhorn_reg(1000.0 * x, 1000.0 * y, rel_grid=grid)
    assert np.allclose(a.rel_bias, b.rel_bias, rtol=1e-6, atol=1e-9)
    assert a.best_rel_reg == b.best_rel_reg


def test_tuned_default_matches_synthetic_optimum_region():
    """The baked-in TUNED_REL_REG sits in the low-bias region of the synthetic
    curve (within 2x of the measured minimum bias)."""
    rng = np.random.default_rng(2)
    x = rng.normal(0.0, 1.0, size=1000)
    y = rng.normal(0.3, 1.0, size=1000)
    tun = tune_sinkhorn_reg(x, y)
    assert TUNED_REL_REG in tun.rel_grid
    i = tun.rel_grid.index(TUNED_REL_REG)
    assert tun.rel_bias[i] <= max(2.0 * tun.best_rel_bias, 0.05)


def test_estimate_epsilon_sinkhorn_auto_reg():
    """The reg='auto' Sinkhorn leg agrees with the exact quantile leg on the
    same CRN samples (tiny config)."""
    cfg = load_config(CONFIG)
    cfg.bonds.n_bonds = 4
    cfg.simulator.horizon = 16
    torch.manual_seed(0)
    q = estimate_epsilon_sinkhorn(cfg, h_ref=1.0, seed=0, n_episodes=3, method="quantile")
    s = estimate_epsilon_sinkhorn(cfg, h_ref=1.0, seed=0, n_episodes=3, method="sinkhorn",
                                  reg="auto")
    assert np.isfinite(s.epsilon_hat) and s.epsilon_hat >= 0.0
    if q.epsilon_hat > 1e-6:
        assert 0.4 < s.epsilon_hat / q.epsilon_hat < 2.5


def test_sinkhorn_divergence_absolute_reg_unchanged():
    """The pre-v4 absolute-reg call path still works (backward compatibility)."""
    rng = np.random.default_rng(3)
    x = rng.normal(0.0, 1.0, size=200)
    y = rng.normal(0.4, 1.0, size=200)
    d = sinkhorn_divergence(x, y, reg=0.05)
    assert np.isfinite(d) and d >= 0.0
    assert d == pytest.approx(quantile_w1(x, y), rel=0.5)


# --------------------------------------------------------------------------- #
# Robust ambiguity radius calibration                                          #
# --------------------------------------------------------------------------- #
def test_calibrate_radius_normal_multiplier_near_one():
    """For ~normal estimates the empirical-quantile radius matches z*s."""
    rng = np.random.default_rng(0)
    est = rng.normal(0.5, 0.1, size=400)
    cal = calibrate_radius(est, confidence=0.95, n_boot=800, seed=0)
    assert cal.multiplier == pytest.approx(1.0, abs=0.15)
    assert cal.calibrated_radius >= cal.z_radius
    assert 0.90 <= cal.coverage_normal <= 1.0
    assert cal.coverage_calibrated >= cal.coverage_normal


def test_calibrate_radius_symmetric_heavy_tails_stay_conservative():
    """Symmetric heavy tails (Student-t) inflate the std *faster* than the 95%
    quantile, so z*s stays conservative and the multiplier is below 1 -- the
    calibrated radius must then fall back to z*s exactly."""
    rng = np.random.default_rng(1)
    est = 0.5 + 0.1 * rng.standard_t(2.0, size=400)
    cal = calibrate_radius(est, confidence=0.95, n_boot=800, seed=0)
    assert cal.multiplier < 1.0
    assert cal.calibrated_radius == pytest.approx(cal.z_radius)


def test_calibrate_radius_contamination_inflates():
    """Rare far-outlier estimates -- the railed-probe / beyond-boundary
    bifurcation pattern -- are the regime the quantile radius exists for:
    ~6% mass at ~12 sigma puts the 95% quantile far above z*s (the
    Cantelli-extremal direction), and the calibrated radius must follow it."""
    rng = np.random.default_rng(2)
    est = rng.normal(0.5, 0.05, size=400)
    outliers = rng.random(size=400) < 0.06
    est[outliers] = 0.5 + 0.6
    cal = calibrate_radius(est, confidence=0.95, n_boot=800, seed=0)
    assert cal.multiplier > 1.5
    assert cal.calibrated_radius == pytest.approx(cal.quantile_radius)
    assert cal.coverage_calibrated >= cal.coverage_normal


def test_calibrate_radius_input_validation():
    with pytest.raises(ValueError):
        calibrate_radius([0.5])


def test_robust_boundary_rejects_unknown_radius_method():
    cfg = load_config(CONFIG)
    with pytest.raises(ValueError):
        robust_boundary(cfg, seeds=[0, 1], radius_method="bayes_optimal_vibes")


@pytest.mark.slow
def test_robust_boundary_calibrated_end_to_end():
    """radius_method='calibrated' runs the full probe pipeline and can only
    widen the ball relative to the normal-approximation radius."""
    cfg = load_config(CONFIG)
    cfg.bonds.n_bonds = 4
    cfg.simulator.horizon = 16
    cfg.rrm.n_episodes = 4
    cfg.operator.epochs = 15
    cfg.policy.inner_steps = 10
    cfg.policy.n_rollouts = 4
    cfg.policy.rollout_horizon = 8
    seeds = [0, 1, 2]
    normal = robust_boundary(cfg, seeds=seeds, radius_method="normal")
    calibr = robust_boundary(cfg, seeds=seeds, radius_method="calibrated")
    assert normal.modulus.mean == pytest.approx(calibr.modulus.mean)
    assert calibr.modulus.radius >= normal.modulus.radius - 1e-12
    assert calibr.modulus.verdict in ("stable", "unstable", "undecided")
