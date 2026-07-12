"""Core acceptance test: the performative-feedback gain controls stability.

The headline empirical claim is that the policy<->distribution retraining loop is
stable when the performative-feedback gain ε (``toxicity_feedback``) is small and
becomes unstable as ε grows, with the best-response contraction modulus crossing
the boundary ``m = 1``.  We assert this on the *median* modulus across seeds (the
per-seed estimate is noisy near the critical point, as expected for a finite
market), using the robust best-response-slope estimator.

ε (not α) is the control variable here: ε scales the feedback directly, whereas
α is confounded by best-response saturation (documented in the README).

These runs invoke the full deploy -> collect -> refit -> re-optimize pipeline, so
they are not instant; the configs are kept modest and the test is marked slow.
"""

from __future__ import annotations

import numpy as np
import pytest

from reflex.estimators import measure_response_modulus
from reflex.config import load_config


def _research_cfg(eps: float, n_bonds: int = 8):
    """The tuned regime in which the feedback-gain transition is measured."""
    cfg = load_config("configs/default.yaml")
    cfg.bonds.n_bonds = n_bonds
    cfg.clients.alpha = 0.5
    cfg.clients.toxicity_feedback = eps
    cfg.clients.info_base_intensity = 0.6
    cfg.clients.info_intensity = 1.4
    cfg.clients.info_cap = 8.0
    cfg.clients.info_spread_decay = 1.5
    cfg.simulator.fundamental_vol = 0.25
    cfg.simulator.init_mispricing_vol = 0.5
    cfg.simulator.horizon = 40
    cfg.reward.quote_anchor_weight = 0.25
    cfg.reward.quote_anchor_ref = 1.0
    cfg.rrm.n_episodes = 16
    cfg.policy.inner_steps = 60
    cfg.policy.n_rollouts = 16
    cfg.operator.epochs = 50
    return cfg


def _median_modulus(eps: float, seeds, n_bonds: int = 8) -> float:
    mods = []
    for s in seeds:
        cfg = _research_cfg(eps, n_bonds=n_bonds)
        mods.append(measure_response_modulus(cfg, seed=s).modulus)
    return float(np.median(mods))


def test_modulus_estimator_is_deterministic() -> None:
    """The BR-slope modulus probe is reproducible given a fixed seed."""
    cfg = _research_cfg(eps=5.0, n_bonds=6)
    cfg.simulator.horizon = 24
    cfg.rrm.n_episodes = 8
    cfg.policy.inner_steps = 30
    cfg.operator.epochs = 25
    a = measure_response_modulus(cfg, seed=0).modulus
    b = measure_response_modulus(cfg, seed=0).modulus
    assert a == pytest.approx(b, abs=1e-9)


def test_feedback_increases_modulus_fast() -> None:
    """Directional (fast): turning the performative feedback on raises the modulus.

    With ε = 0 there is no policy-dependent toxicity, so the loop is (near-)
    perfectly contractive (m ≈ 0); with ε large the modulus is materially larger.
    This is the robust core of the mechanism and uses a small, fast config.
    """
    def fast(eps):
        cfg = _research_cfg(eps, n_bonds=6)
        cfg.simulator.horizon = 24
        cfg.rrm.n_episodes = 8
        cfg.policy.inner_steps = 30
        cfg.policy.n_rollouts = 12
        cfg.operator.epochs = 25
        return cfg

    seeds = [0, 1, 2]
    m0 = float(np.median([measure_response_modulus(fast(0.0), seed=s).modulus for s in seeds]))
    m_hi = float(np.median([measure_response_modulus(fast(6.0), seed=s).modulus for s in seeds]))
    assert m0 < 0.3, f"ε=0 should be strongly contractive, got median m={m0:.3f}"
    assert m_hi > m0 + 0.15, f"feedback should raise the modulus: m(0)={m0:.3f}, m(6)={m_hi:.3f}"


@pytest.mark.slow
def test_stability_boundary_crossing() -> None:
    """The median modulus crosses 1 as ε grows: stable at ε=0, unstable at ε=6.

    Asserted on the median over seeds (per-seed estimates are noisy near the
    critical point). This is the headline acceptance criterion.
    """
    seeds = [0, 1, 2]
    m_low = _median_modulus(0.0, seeds)
    m_high = _median_modulus(6.0, seeds)
    assert m_low < 1.0, f"ε=0 must be stable (m<1), got median m={m_low:.3f}"
    assert m_high > 1.0, f"ε=6 must be unstable (m>1), got median m={m_high:.3f}"
    assert m_high > m_low, "modulus must increase from ε=0 to ε=6"
