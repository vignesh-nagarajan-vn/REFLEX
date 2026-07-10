"""Tests for the closed-form GLFT baseline policy (theory-anchored quoting)."""

from __future__ import annotations

import pytest
import torch

from reflex.config import load_config
from reflex.policy import GLFTBaselinePolicy, build_glft_baseline
from reflex.theory.analytic_boundary import reference_state, solve_fixed_point
from reflex.theory.perfgd import echo_chamber_gap, solve_performative_optimum

CONFIG = "configs/default.yaml"


@pytest.fixture()
def cfg():
    return load_config(CONFIG)


@pytest.fixture()
def ref_state(cfg):
    from reflex.env import StructuralSimulator

    torch.manual_seed(0)
    return StructuralSimulator(cfg).reset(seed=123)


def test_quotes_equal_analytic_stable_point(cfg, ref_state):
    pol = GLFTBaselinePolicy(cfg, mode="stable_point")
    ref = reference_state(cfg)
    h_star = solve_fixed_point(cfg, ref)
    q = pol.quote(ref_state)
    assert q.half_spread.shape[0] == ref_state.n_bonds
    assert torch.allclose(q.half_spread, torch.full_like(q.half_spread, h_star), atol=1e-6)
    assert torch.all(q.skew == 0.0)


def test_optimum_mode_matches_theory(cfg, ref_state):
    pol = build_glft_baseline(cfg, mode="optimum")
    ref = reference_state(cfg)
    h_po = solve_performative_optimum(cfg, ref)
    assert pol.half_spread_value == pytest.approx(h_po, abs=1e-6)


def test_stable_vs_optimum_gap_sign_consistent(cfg):
    """The SP-vs-PO ordering must match the echo-chamber gap of theory 1.2."""
    sp = GLFTBaselinePolicy(cfg, mode="stable_point").half_spread_value
    po = GLFTBaselinePolicy(cfg, mode="optimum").half_spread_value
    gap = echo_chamber_gap(cfg, reference_state(cfg))
    assert sp - po == pytest.approx(gap.decision_gap_exact, abs=1e-8)


def test_baseline_is_parameter_free_and_bounded(cfg, ref_state):
    pol = GLFTBaselinePolicy(cfg)
    assert pol.n_params == 0  # buffer, not a trainable parameter
    assert 0.0 <= pol.half_spread_value <= cfg.policy.max_half_spread
    # quote() must be repeatable / deterministic
    q1 = pol.quote(ref_state)
    q2 = pol.quote(ref_state)
    assert torch.equal(q1.half_spread, q2.half_spread)


def test_rejects_bad_inputs(cfg):
    with pytest.raises(ValueError):
        GLFTBaselinePolicy(cfg, mode="nonsense")
    with pytest.raises(TypeError):
        GLFTBaselinePolicy(cfg.policy)  # needs the full Config
