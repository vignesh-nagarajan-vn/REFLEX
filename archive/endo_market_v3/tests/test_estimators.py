"""Tests for the epsilon estimators (Sinkhorn/Wasserstein, CKS, triangulation)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from reflex.config import load_config
from reflex.estimators.cks import estimate_epsilon_cks
from reflex.estimators.sinkhorn import (
    estimate_epsilon_sinkhorn,
    quantile_w1,
    sinkhorn_divergence,
)
from reflex.theory.analytic_boundary import epsilon as epsilon_of
from reflex.theory.analytic_boundary import reference_state

CONFIG = "configs/default.yaml"


# ------------------------- distances (no simulation) ------------------------ #
def test_quantile_w1_recovers_location_shift():
    rng = np.random.default_rng(0)
    x = rng.normal(0.0, 1.0, 4000)
    y = rng.normal(0.75, 1.0, 4000)
    assert quantile_w1(x, y) == pytest.approx(0.75, abs=0.06)


def test_quantile_w1_unequal_sizes():
    rng = np.random.default_rng(1)
    x = rng.normal(0.0, 1.0, 6000)
    y = rng.normal(0.5, 1.0, 2500)
    assert quantile_w1(x, y) == pytest.approx(0.5, abs=0.1)


def test_sinkhorn_divergence_matches_quantile_w1():
    rng = np.random.default_rng(2)
    x = rng.normal(0.0, 0.5, 300)
    y = rng.normal(0.6, 0.5, 300)
    exact = quantile_w1(x, y)
    entropic = sinkhorn_divergence(x, y, reg=0.05, n_iters=300)
    assert entropic == pytest.approx(exact, rel=0.25)


def test_sinkhorn_divergence_zero_for_identical_clouds():
    rng = np.random.default_rng(3)
    x = rng.normal(0.0, 1.0, 200)
    assert sinkhorn_divergence(x, x.copy()) == pytest.approx(0.0, abs=1e-6)


# --------------------- CKS fit recovery (mocked flow curve) ----------------- #
def test_cks_recovers_synthetic_flow_curve(monkeypatch):
    """With a noiseless lambda(h) = C0 + C1*exp(-c*h) the fit must recover c."""
    C0, C1, c = 0.4, 2.0, 1.7

    def fake_mean_informed(cfg, simulator, h_dep, seed, n_episodes):
        return C0 + C1 * np.exp(-c * h_dep)

    import reflex.estimators.cks as cks_mod

    monkeypatch.setattr(cks_mod, "_mean_informed", fake_mean_informed)
    cfg = load_config(CONFIG)
    res = estimate_epsilon_cks(cfg, h_ref=1.0, seed=0, n_episodes=1, simulator=object())
    assert res.fit_ok
    assert res.c == pytest.approx(c, rel=0.01)
    assert res.epsilon_hat == pytest.approx(c * C1 * np.exp(-c * 1.0), rel=0.01)


# --------------------- small-simulation sanity checks ----------------------- #
@pytest.fixture(scope="module")
def small_cfg():
    cfg = load_config(CONFIG)
    cfg.simulator.horizon = 20
    cfg.bonds.n_bonds = 4
    return cfg


def test_sinkhorn_epsilon_probe_sane(small_cfg):
    """The Wasserstein probe must land within an order of magnitude of the
    closed form (it upper-bounds the mean shift; CRN keeps it tight)."""
    torch.manual_seed(0)
    res = estimate_epsilon_sinkhorn(small_cfg, h_ref=1.0, seed=0, n_episodes=4)
    ref = reference_state(small_cfg)
    eps_true = epsilon_of(small_cfg, 1.0, ref)
    assert np.isfinite(res.epsilon_hat) and res.epsilon_hat > 0
    assert 0.1 * eps_true < res.epsilon_hat < 10.0 * eps_true
    # W1 dominates the location shift by construction
    assert res.w1 >= res.mean_shift - 1e-9


def test_cks_epsilon_probe_sane(small_cfg):
    torch.manual_seed(0)
    res = estimate_epsilon_cks(small_cfg, h_ref=1.0, seed=0, n_episodes=4)
    ref = reference_state(small_cfg)
    eps_true = epsilon_of(small_cfg, 1.0, ref)
    assert np.isfinite(res.epsilon_hat) and res.epsilon_hat > 0
    assert 0.1 * eps_true < res.epsilon_hat < 10.0 * eps_true
    # the measured flow curve must be decreasing in h overall
    assert res.lambda_grid[0] > res.lambda_grid[-1]


@pytest.mark.slow
def test_triangulation_three_legs_agree(small_cfg):
    """The measured legs agree with the *realized-state* closed form.

    The comparison target is ``epsilon_analytic_realized`` (theory 1.1 §9):
    the deployment inflates the liquidity ratio well above the a-priori A2
    value (rho ~ 2 vs 1), so the frozen a-priori closed form understates the
    realized flow sensitivity.  Bands: the distribution-space legs (Sinkhorn,
    CKS) probe the environment directly and must agree within a factor 3; the
    decision-space BR leg additionally carries the finite-budget optimizer
    attenuation of the retraining map (documented protocol dependence), so it
    gets a factor-5 band.
    """
    from reflex.estimators.triangulate import triangulate_epsilon

    res = triangulate_epsilon(small_cfg, seed=0, n_episodes=6)
    target = res.epsilon_analytic_realized
    assert np.isfinite(target) and target > 0
    for leg in (res.epsilon_sinkhorn, res.epsilon_cks):
        assert np.isfinite(leg) and leg > 0
        assert target / 3.0 < leg < target * 3.0
    assert np.isfinite(res.epsilon_br) and res.epsilon_br > 0
    assert target / 5.0 < res.epsilon_br < target * 5.0
