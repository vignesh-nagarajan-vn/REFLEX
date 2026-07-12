"""Multi-dealer competition and systemic instability (math-theory 1.3).

Exact closed-form checks of the N-dealer boundary, the joint-Jacobian spectrum,
the critical dealer count, the mean-field limits, and the genuine N-dimensional
best-response cobweb -- plus one slow smoke test of the empirical common-mode probe
through the learned-operator pipeline.
"""

from __future__ import annotations

import numpy as np
import pytest

from reflex.theory.analytic_boundary import analytic_boundary
from reflex.theory.multi_dealer import (
    identify_kappa,
    joint_jacobian,
    mean_field_boundary,
    multi_dealer_boundary,
    n_eff,
    run_joint_rrm,
    strong_coupling_limit,
    sweep_dealer_count,
)
from reflex.config import load_config


def _cfg(**clients):
    cfg = load_config("configs/default.yaml")
    for k, v in clients.items():
        setattr(cfg.clients, k, v)
    return cfg


def test_n_eff_endpoints_and_monotone() -> None:
    assert n_eff(5, 0.0) == pytest.approx(1.0)  # decoupled
    assert n_eff(5, 1.0) == pytest.approx(5.0)  # full spillover
    vals = [n_eff(5, k) for k in (0.0, 0.25, 0.5, 0.75, 1.0)]
    assert all(np.diff(vals) > 0.0)


def test_joint_modulus_is_neff_times_single() -> None:
    """m_N = N_eff * m_1 exactly (1.3 §4)."""
    mb = multi_dealer_boundary(_cfg(toxicity_feedback=1.0), n_dealers=4, kappa=1.0)
    assert mb.m_N == pytest.approx(mb.n_eff * mb.m_1, rel=1e-12)
    assert mb.lambda_plus == pytest.approx(-mb.m_N)
    assert mb.lambda_minus == pytest.approx(-mb.m_1 * (1.0 - mb.kappa))


def test_kappa_zero_reproduces_single_dealer() -> None:
    """At kappa=0 the N-dealer boundary is the single-dealer boundary (regression)."""
    cfg = _cfg(toxicity_feedback=1.0, alpha=0.5)
    single = analytic_boundary(cfg)
    multi = multi_dealer_boundary(cfg, n_dealers=6, kappa=0.0)
    assert multi.n_eff == pytest.approx(1.0)
    assert multi.h_star == pytest.approx(single.h_star, abs=1e-6)
    assert multi.m_N == pytest.approx(single.modulus, rel=1e-9)


def test_joint_jacobian_spectrum() -> None:
    """J_BR has one common-mode eigenvalue -m_N and (N-1) differential -m_1(1-kappa)."""
    m1, N, kappa = 0.3, 4, 0.7
    J = joint_jacobian(m1, N, kappa)
    assert np.allclose(J, J.T)  # symmetric shared pool
    eig = np.sort(np.linalg.eigvalsh(J))
    ne = n_eff(N, kappa)
    lam_plus = -m1 * ne
    lam_minus = -m1 * (1.0 - kappa)
    # most-negative eigenvalue is the common mode; the rest are the degenerate diff mode
    assert eig[0] == pytest.approx(lam_plus, rel=1e-9)
    assert np.allclose(eig[1:], lam_minus, atol=1e-9)
    assert max(abs(e) for e in eig) == pytest.approx(m1 * ne, rel=1e-9)


def test_linear_in_n_law_and_integer_crossing() -> None:
    """Fixed-regime m_N = N_eff*m_1 is a straight line; stable iff N < N_c (1.3 §8.1-8.2)."""
    cfg = _cfg(toxicity_feedback=1.5, alpha=0.6)
    h0 = analytic_boundary(cfg).h_star  # hold the structural operating point
    sweep = sweep_dealer_count(cfg, list(range(1, 12)), kappa=1.0, reference_spread=h0)
    m1 = sweep[0].m_1
    # exactly linear through the origin with slope m_1 (m_1 constant across N here)
    for s in sweep:
        assert s.m_1 == pytest.approx(m1, rel=1e-12)
        assert s.m_N == pytest.approx(s.n_dealers * m1, rel=1e-9)  # kappa=1 => N_eff=N
    n_c = sweep[0].n_critical
    assert n_c == pytest.approx(1.0 / m1, rel=1e-9)
    for s in sweep:
        if s.n_dealers < n_c - 1e-9:
            assert s.stable
        if s.n_dealers > n_c + 1e-9:
            assert not s.stable


def test_defensive_widening_makes_self_consistent_sublinear() -> None:
    """Self-consistent PSNE widens with N, so m_N grows sub-linearly (1.1 §6.3, 1.3 §11)."""
    cfg = _cfg(toxicity_feedback=1.5, alpha=0.6)
    h0 = analytic_boundary(cfg).h_star
    for N in (4, 8, 12):
        self_consistent = multi_dealer_boundary(cfg, n_dealers=N, kappa=1.0)
        fixed = multi_dealer_boundary(cfg, n_dealers=N, kappa=1.0, reference_spread=h0)
        assert self_consistent.h_star > h0  # equilibrium spread widens with competition
        assert self_consistent.m_N < fixed.m_N  # widening damps the modulus


def test_gamma_joint_sign_matches_boundary() -> None:
    """gamma_joint > 0  <=>  stable (1.3 §6.2)."""
    for N in (1, 3, 6, 10):
        mb = multi_dealer_boundary(_cfg(toxicity_feedback=1.5, alpha=0.6), n_dealers=N, kappa=1.0)
        assert (mb.gamma_joint > 0.0) == mb.stable


def test_mean_field_limits() -> None:
    """kappa=c/N gives N_eff -> 1+c (finite); fixed kappa collapses the boundary (1.3 §7)."""
    mf = mean_field_boundary(_cfg(toxicity_feedback=1.0), c=2.0)
    assert mf.n_eff_limit == pytest.approx(3.0)
    assert mf.boundary_epsilon > 0.0
    strong = strong_coupling_limit()
    assert strong.n_eff_limit == float("inf")
    assert strong.boundary_epsilon == 0.0


def test_identify_kappa_round_trip() -> None:
    """Recover kappa from the (constructed) in-phase / anti-phase modulus ratio."""
    m1, N = 0.4, 5
    for kappa in (0.0, 0.3, 0.6, 1.0):
        ne = n_eff(N, kappa)
        m_common = ne * m1
        m_diff = (1.0 - kappa) * m1
        assert identify_kappa(m_common, m_diff, N) == pytest.approx(kappa, abs=1e-9)


def test_joint_cobweb_converges_to_symmetric_psne_when_stable() -> None:
    """A stable N-dealer market's joint cobweb settles to the symmetric PSNE."""
    cfg = _cfg(toxicity_feedback=1.0, alpha=0.5)
    mb = multi_dealer_boundary(cfg, n_dealers=3, kappa=1.0)
    assert mb.stable
    traj = run_joint_rrm(cfg, n_dealers=3, kappa=1.0, h0=1.0, n_steps=80)
    final = traj[-1]
    assert np.allclose(final, final[0], atol=1e-6)  # symmetric
    assert final[0] == pytest.approx(mb.h_star, abs=1e-3)


def test_joint_cobweb_fails_to_settle_when_unstable() -> None:
    """Past the critical dealer count the joint cobweb does not settle (1.3 §8.3)."""
    cfg = _cfg(toxicity_feedback=1.5, alpha=0.6)
    # pick N well above N_c so m_N > 1
    mb = multi_dealer_boundary(cfg, n_dealers=12, kappa=1.0)
    assert not mb.stable
    traj = run_joint_rrm(cfg, n_dealers=12, kappa=1.0, h0=mb.h_star, n_steps=60)
    # either it left the operating range early, or it never contracted to the PSNE
    settled = traj.shape[0] < 61 or not np.allclose(traj[-1], mb.h_star, atol=1e-2)
    assert settled


@pytest.mark.slow
def test_empirical_common_mode_probe_runs_and_is_deterministic() -> None:
    """The learned-operator in-phase/anti-phase probes run and are reproducible.

    We assert only determinism and finiteness: absolute agreement with the linear
    m_N = N_eff*m_1 law holds only in the responsive (non-saturated) regime -- the
    learned-operator measurement attenuates otherwise, a documented feature.
    """
    from reflex.theory.multi_dealer import common_mode_probe

    cfg = load_config("configs/default.yaml")
    cfg.bonds.n_bonds = 6
    cfg.clients.alpha = 0.5
    cfg.clients.toxicity_feedback = 3.0
    cfg.simulator.horizon = 20
    cfg.rrm.n_episodes = 6
    cfg.policy.inner_steps = 25
    cfg.policy.n_rollouts = 10
    cfg.operator.epochs = 20

    p1 = common_mode_probe(cfg, n_dealers=3, kappa=1.0, seed=0)
    p2 = common_mode_probe(cfg, n_dealers=3, kappa=1.0, seed=0)
    assert np.isfinite(p1.m_N_measured) and np.isfinite(p1.m_diff_measured)
    assert p1.m_N_measured == pytest.approx(p2.m_N_measured, abs=1e-9)
    assert p1.m_diff_measured == pytest.approx(p2.m_diff_measured, abs=1e-9)
