"""Distributionally robust stability certificate (math-theory 1.4).

Fast, deterministic checks of the statistical core -- the ambiguity radius, the
certificate verdicts, the sample-complexity curve, and the log-log rate -- plus a
slow smoke test of the cross-seed empirical harness.
"""

from __future__ import annotations

import numpy as np
import pytest

from endo_market.analysis.robust_boundary import (
    empirical_radius,
    finite_sample_radius,
    loglog_rate,
    robust_certificate,
    sample_complexity,
)


def test_radius_is_z_times_std() -> None:
    est = [0.80, 0.85, 0.90, 0.82, 0.88]
    mean, std, radius = empirical_radius(est, confidence=0.95)
    assert mean == pytest.approx(float(np.mean(est)))
    assert std == pytest.approx(float(np.std(est, ddof=1)))
    assert radius == pytest.approx(1.6448536 * std, rel=1e-4)  # one-sided z_0.95


def test_certificate_three_verdicts() -> None:
    assert robust_certificate(0.80, 0.05, boundary=1.0).verdict == "stable"      # 0.85 < 1
    assert robust_certificate(1.20, 0.05, boundary=1.0).verdict == "unstable"    # 1.15 > 1
    assert robust_certificate(0.98, 0.05, boundary=1.0).verdict == "undecided"   # 0.93..1.03
    c = robust_certificate(0.80, 0.05, boundary=1.0)
    assert c.robust_boundary == pytest.approx(1.0 - 0.05)


def test_structural_floor_does_not_shrink() -> None:
    """eta_mod tightens the certificate and never disappears with more data (1.4 §6.2)."""
    # Zero statistical radius (infinite data) but 20% structural ambiguity.
    c = robust_certificate(0.90, 0.0, boundary=1.0, eta_mod=0.2)
    assert c.upper == pytest.approx(0.90 * 1.2)  # 1.08 > 1: worst case is unstable
    assert c.verdict != "stable"  # structural ambiguity blocks a stability claim
    # Without the structural floor the same point is clearly stable.
    assert robust_certificate(0.90, 0.0, boundary=1.0).verdict == "stable"


def test_sample_complexity_inverse_square() -> None:
    """n_req = (z*sigma/Delta)^2 ~ Delta^-2, diverging at the boundary (1.4 §5)."""
    n1 = sample_complexity(0.3, 0.10)
    n2 = sample_complexity(0.3, 0.05)  # half the distance
    assert n2 == pytest.approx(4.0 * n1, rel=1e-9)  # inverse-square
    assert sample_complexity(0.3, 0.0) == float("inf")  # exact crossing is unresolvable


def test_loglog_rate_recovers_minus_half() -> None:
    """A synthetic s ~ n^{-1/2} sequence has log-log slope -0.5 (the CRN rate, 1.4 §2.3)."""
    ns = [16, 32, 64, 128, 256]
    stds = [0.4 / np.sqrt(n) for n in ns]
    assert loglog_rate(ns, stds) == pytest.approx(-0.5, abs=1e-6)


def test_finite_sample_radius_shrinks_as_sqrt_n() -> None:
    r64 = finite_sample_radius(0.3, 64)
    r256 = finite_sample_radius(0.3, 256)
    assert r64 > 0.0
    assert r256 == pytest.approx(r64 / 2.0, rel=1e-9)  # 4x n -> half radius


@pytest.mark.slow
def test_empirical_robust_boundary_runs_and_is_deterministic() -> None:
    """The cross-seed harness runs, is reproducible, and returns a valid verdict."""
    from endo_market.analysis.robust_boundary import robust_boundary

    def cfg():
        from endo_market.config import load_config

        c = load_config("configs/default.yaml")
        c.bonds.n_bonds = 6
        c.clients.alpha = 0.5
        c.clients.toxicity_feedback = 3.0
        c.simulator.horizon = 20
        c.rrm.n_episodes = 8
        c.policy.inner_steps = 25
        c.policy.n_rollouts = 10
        c.operator.epochs = 20
        return c

    r1 = robust_boundary(cfg(), seeds=[0, 1, 2, 3])
    r2 = robust_boundary(cfg(), seeds=[0, 1, 2, 3])
    assert r1.modulus.verdict in {"stable", "unstable", "undecided"}
    assert np.isfinite(r1.modulus.radius)
    assert r1.modulus.mean == pytest.approx(r2.modulus.mean, abs=1e-9)
    # epsilon-space image is the modulus certificate scaled by gamma/beta
    assert r1.epsilon.mean == pytest.approx(r1.modulus.mean * r1.gamma_over_beta, rel=1e-9)
