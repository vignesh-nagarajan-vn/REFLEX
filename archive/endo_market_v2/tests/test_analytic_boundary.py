"""Closed-form stability boundary (math-theory 1.1 foundation).

These tests pin the analytic constants (gamma, beta, epsilon, the modulus and the
fixed point) that the PerfGD (1.2) and multi-dealer (1.3) modules build on.  They
are pure closed-form checks -- deterministic and instant -- plus one slow
directional cross-check against the model-free best-response probe.
"""

from __future__ import annotations

import numpy as np
import pytest

from endo_market.analysis.analytic_boundary import (
    analytic_boundary,
    beta,
    blind_gradient,
    epsilon,
    gate_means,
    reference_state,
    solve_fixed_point,
)
from endo_market.config import load_config


def _cfg(**clients):
    cfg = load_config("configs/default.yaml")
    for k, v in clients.items():
        setattr(cfg.clients, k, v)
    return cfg


def test_gate_means_are_bounded_and_signed() -> None:
    """gbar in (0,1); the signed gate mean shares the sign of the mispricing."""
    gbar_pos, a_pos = gate_means(0.8)
    gbar_neg, a_neg = gate_means(-0.8)
    assert 0.0 < gbar_pos < 1.0
    assert a_pos > 0.0 and a_neg < 0.0
    # gbar is symmetric in u; the signed mean is odd.
    assert gbar_pos == pytest.approx(gbar_neg, abs=1e-9)
    assert a_pos == pytest.approx(-a_neg, abs=1e-9)


def test_adverse_severity_nonnegative() -> None:
    """psi = sigma_s * u * a(u) / gbar(u) >= 0 (sign(u) == sign(a))."""
    for g in (-1.0, -0.3, 0.0, 0.3, 1.0):
        ref = reference_state(_cfg(), mispricing=g)
        assert ref.psi >= -1e-12


def test_default_regime_is_stable() -> None:
    """The default config sits inside the stable region with an interior h*."""
    b = analytic_boundary(_cfg())
    assert 0.0 < b.h_star < _cfg().policy.max_half_spread
    assert b.gamma > 0.0
    assert b.beta == pytest.approx(load_config("configs/default.yaml").reward.pnl_scale)
    assert b.modulus < 1.0 and b.stable
    assert b.boundary_epsilon == pytest.approx(b.gamma / b.beta)


def test_epsilon_linear_in_feedback_gain() -> None:
    """epsilon is exactly linear in toxicity_feedback at a fixed spread (1.1 §3.2)."""
    ref1 = reference_state(_cfg(toxicity_feedback=0.2))
    ref2 = reference_state(_cfg(toxicity_feedback=0.4))
    e1 = epsilon(_cfg(toxicity_feedback=0.2), 1.3, ref1)
    e2 = epsilon(_cfg(toxicity_feedback=0.4), 1.3, ref2)
    assert e2 == pytest.approx(2.0 * e1, rel=1e-9)


def test_modulus_monotone_in_feedback() -> None:
    """Turning up the performative feedback raises the closed-form modulus."""
    ms = [analytic_boundary(_cfg(toxicity_feedback=f)).modulus for f in (0.1, 0.5, 2.0, 6.0)]
    assert all(np.diff(ms) > 0.0)


def test_beta_is_pnl_scale() -> None:
    cfg = _cfg()
    cfg.reward.pnl_scale = 2.5
    assert beta(cfg) == pytest.approx(2.5)


def test_fixed_point_zeroes_the_blind_gradient() -> None:
    """solve_fixed_point returns a genuine root of the self-consistent FOC G."""
    cfg = _cfg(toxicity_feedback=1.0, alpha=0.5)
    ref = reference_state(cfg)
    h_star = solve_fixed_point(cfg, ref)
    assert blind_gradient(cfg, h_star, ref) == pytest.approx(0.0, abs=1e-6)


@pytest.mark.slow
def test_predicted_modulus_tracks_measured_direction() -> None:
    """The closed-form modulus and the model-free probe move together with epsilon.

    Absolute agreement is not expected outside the responsive regime (the modulus
    saturates / attenuates through the learned operator -- a documented feature);
    the robust, in-regime claim is that both rise as the feedback gain rises.
    """
    from endo_market.analysis.response_modulus import measure_response_modulus

    def cfg(f):
        c = load_config("configs/default.yaml")
        c.bonds.n_bonds = 6
        c.clients.alpha = 0.5
        c.clients.toxicity_feedback = f
        c.simulator.horizon = 24
        c.rrm.n_episodes = 8
        c.policy.inner_steps = 30
        c.policy.n_rollouts = 12
        c.operator.epochs = 25
        return c

    seeds = [0, 1, 2]
    m_pred_lo = analytic_boundary(cfg(0.5)).modulus
    m_pred_hi = analytic_boundary(cfg(6.0)).modulus
    m_meas_lo = float(np.median([measure_response_modulus(cfg(0.5), seed=s).modulus for s in seeds]))
    m_meas_hi = float(np.median([measure_response_modulus(cfg(6.0), seed=s).modulus for s in seeds]))
    assert m_pred_hi > m_pred_lo
    assert m_meas_hi > m_meas_lo
