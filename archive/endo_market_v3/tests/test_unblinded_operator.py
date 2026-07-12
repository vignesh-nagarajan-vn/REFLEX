"""Tests for the un-blinded operator: windowed fitting identifies dD/dphi."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from reflex.config import load_config
from reflex.env import StructuralSimulator
from reflex.equilibrium import DeploymentRecord, collect, fit_operator_windowed
from reflex.estimators.br_slope import _deployed_policy_at
from reflex.operator import OUT_DIM, MarketResponseOperator
from reflex.theory.analytic_boundary import epsilon as epsilon_of
from reflex.theory.analytic_boundary import reference_state
from reflex.types import policy_summary

CONFIG = "configs/default.yaml"


@pytest.fixture(scope="module")
def tiny_cfg():
    cfg = load_config(CONFIG)
    cfg.bonds.n_bonds = 4
    cfg.simulator.horizon = 16
    cfg.rrm.n_episodes = 4
    cfg.operator.epochs = 25
    cfg.operator.hidden = 32
    return cfg


@pytest.fixture(scope="module")
def windowed_operator(tiny_cfg):
    """An operator fit on three deployments at different spreads (0.7/1.0/1.3)."""
    torch.manual_seed(0)
    sim = StructuralSimulator(tiny_cfg)
    deployments = []
    for i, h in enumerate((0.7, 1.0, 1.3)):
        pol = _deployed_policy_at(tiny_cfg, h)
        gen = torch.Generator().manual_seed(100 + i)
        trans = collect(sim, pol, tiny_cfg, base_seed=100 + i, generator=gen)
        deployments.append(DeploymentRecord(transitions=trans, policy=pol))
    torch.manual_seed(0)
    op = MarketResponseOperator(tiny_cfg)
    fit = fit_operator_windowed(
        op, deployments, tiny_cfg.operator, generator=torch.Generator().manual_seed(7)
    )
    ref_state = sim.reset(seed=999).detach()
    return op, fit, ref_state


def test_windowed_fit_improves_nll(windowed_operator):
    _, fit, _ = windowed_operator
    assert fit.n_rows > 0
    assert fit.best_val_nll < fit.baseline_val_nll  # training actually learned


def test_distribution_response_shape_and_finite(windowed_operator, tiny_cfg):
    op, _, ref_state = windowed_operator
    pol = _deployed_policy_at(tiny_cfg, 1.0)
    summ = policy_summary(ref_state, pol).detach()
    resp = op.distribution_response(ref_state, pol.quote(ref_state), summ)
    assert resp.shape == (OUT_DIM,)
    assert torch.isfinite(resp).all()


def test_learned_toxic_slope_negative_and_scaled(windowed_operator, tiny_cfg):
    """The windowed operator's learned d(adverse)/dh must be negative (tighter
    quotes summon more toxic flow) and within an order of magnitude of the
    closed form -psi*epsilon(h)."""
    op, _, ref_state = windowed_operator
    pol = _deployed_policy_at(tiny_cfg, 1.0)
    summ = policy_summary(ref_state, pol).detach()
    learned = op.toxic_slope(ref_state, pol.quote(ref_state), summ)

    ref = reference_state(tiny_cfg)
    analytic = -ref.psi * epsilon_of(tiny_cfg, 1.0, ref)  # per bond, per step
    # The operator predicts the summed-over-signal-noise per-bond adverse mean;
    # compare per-bond magnitudes with generous tolerance (small net, few rows).
    assert learned < 0.0, f"learned slope {learned:.4f} not negative"
    assert 0.05 * abs(analytic) < abs(learned) < 20.0 * abs(analytic), (
        f"learned {learned:.4f} vs analytic {analytic:.4f}"
    )


def test_single_deployment_fit_still_works(tiny_cfg):
    """Window of one deployment reduces to the plain (blind) fit -- mechanics only."""
    torch.manual_seed(1)
    sim = StructuralSimulator(tiny_cfg)
    pol = _deployed_policy_at(tiny_cfg, 1.0)
    trans = collect(sim, pol, tiny_cfg, base_seed=5, generator=torch.Generator().manual_seed(5))
    op = MarketResponseOperator(tiny_cfg)
    fit = fit_operator_windowed(
        op, [DeploymentRecord(transitions=trans, policy=pol)], tiny_cfg.operator,
        generator=torch.Generator().manual_seed(6),
    )
    assert fit.n_rows == len(trans) * tiny_cfg.bonds.n_bonds
    assert np.isfinite(fit.best_val_nll)


def test_empty_window_returns_empty_fit(tiny_cfg):
    op = MarketResponseOperator(tiny_cfg)
    fit = fit_operator_windowed(op, [], tiny_cfg.operator)
    assert fit.n_rows == 0
