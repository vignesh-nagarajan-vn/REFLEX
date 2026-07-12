"""PerfGD-corrected loop (math-theory 1.2).

Deterministic closed-form checks of the un-blinding correction: the objective
curvature gamma_PO, the headline "stable beyond epsilon*" claim, the RRM-diverges
/ PerfGD-converges demonstration, and the echo-chamber gap.
"""

from __future__ import annotations

import pytest

from endo_market.analysis.analytic_boundary import epsilon, reference_state
from endo_market.config import load_config
from endo_market.equilibrium.perfgd_loop import (
    analyze_perfgd,
    perfgd_correction,
    solve_performative_optimum,
)


def _default():
    return load_config("configs/default.yaml")


def _unstable_regime():
    """A regime with an interior fixed point and RRM modulus m > 1 (1.2 §5)."""
    cfg = load_config("configs/default.yaml")
    cfg.clients.alpha = 1.0
    cfg.clients.toxicity_feedback = 6.0
    cfg.clients.info_intensity = 3.0
    cfg.clients.info_spread_decay = 0.8
    cfg.reward.quote_anchor_weight = 0.15
    return cfg


def test_correction_adds_curvature_in_operating_regime() -> None:
    """gamma_PO > gamma at the default optimum: the correction is stabilising."""
    r = analyze_perfgd(_default(), run_loops=False)
    assert r.gamma_po > r.gamma > 0.0


def test_perfgd_stable_beyond_rrm_boundary() -> None:
    """Headline (1.2 §5): where the RRM modulus exceeds 1, PerfGD stays concave."""
    r = analyze_perfgd(_unstable_regime(), run_loops=False)
    assert r.modulus_rrm > 1.0 and not r.rrm_stable
    assert r.gamma_po > 0.0 and r.perfgd_strongly_concave


def test_rrm_diverges_perfgd_converges() -> None:
    """In the unstable regime the blind cobweb fails to settle while PerfGD reaches h_PO."""
    r = analyze_perfgd(_unstable_regime(), n_steps=120)
    assert not r.rrm_converged
    assert r.perfgd_converged
    assert r.perfgd_trajectory[-1] == pytest.approx(r.h_po, abs=1e-3)


def test_correction_vanishes_at_psi_and_flips_sign() -> None:
    """Delta(psi) = 0 and it flips sign across h = psi (1.2 §7.1)."""
    cfg = _default()
    ref = reference_state(cfg)
    assert perfgd_correction(cfg, ref.psi, ref) == pytest.approx(0.0, abs=1e-12)
    below = perfgd_correction(cfg, ref.psi - 0.3, ref)
    above = perfgd_correction(cfg, ref.psi + 0.3, ref)
    assert below > 0.0 > above  # pulls wider below psi, tighter above


def test_echo_chamber_gap_positive_and_vanishes_without_feedback() -> None:
    """Stable point over-defends (h_SP > h_PO) and the gap -> 0 as epsilon -> 0."""
    cfg = _default()
    ref = reference_state(cfg)
    r = analyze_perfgd(cfg, run_loops=False)
    # In the default regime toxic flow is net profitable at the stable spread.
    assert r.h_sp > ref.psi
    assert r.gap.decision_gap > 0.0
    assert r.h_sp > r.h_po

    # Kill the performative feedback: SP and PO coincide.
    cfg0 = _default()
    cfg0.clients.toxicity_feedback = 0.0
    r0 = analyze_perfgd(cfg0, run_loops=False)
    assert r0.gap.decision_gap == pytest.approx(0.0, abs=1e-6)
    assert r0.h_sp == pytest.approx(r0.h_po, abs=1e-4)


def test_optimum_zeroes_the_corrected_gradient() -> None:
    """h_PO is a genuine root of Phi' = G + Delta."""
    from endo_market.equilibrium.perfgd_loop import perfgd_gradient

    cfg = _unstable_regime()
    ref = reference_state(cfg)
    h_po = solve_performative_optimum(cfg, ref)
    assert perfgd_gradient(cfg, h_po, ref) == pytest.approx(0.0, abs=1e-6)


def test_decision_gap_scales_with_adversariality() -> None:
    """The echo-chamber decision gap grows with the feedback gain (1.2 §7.2)."""
    gaps = [analyze_perfgd(_default_with(f), run_loops=False).gap.decision_gap for f in (0.1, 0.5, 1.5)]
    assert gaps[0] < gaps[1] < gaps[2]


def _default_with(f):
    cfg = load_config("configs/default.yaml")
    cfg.clients.toxicity_feedback = f
    return cfg
