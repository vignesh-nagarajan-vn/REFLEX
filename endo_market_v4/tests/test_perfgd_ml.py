"""Tests for the PerfGD-corrected ML loops (analytic surrogate + live summary)."""

from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

from reflex.config import load_config
from reflex.env import StructuralSimulator
from reflex.equilibrium import (
    collect,
    collect_initial_states,
    fit_operator,
    optimize_policy,
    rgd_step,
    run_loop,
)
from reflex.operator import MarketResponseOperator
from reflex.policy import build_policy
from reflex.types import policy_summary

CONFIG = "configs/default.yaml"


@pytest.fixture(scope="module")
def tiny_cfg():
    cfg = load_config(CONFIG)
    cfg.bonds.n_bonds = 4
    cfg.simulator.horizon = 16
    cfg.rrm.n_episodes = 4
    cfg.rrm.eval_episodes = 2
    cfg.rrm.max_iters = 2
    cfg.operator.epochs = 15
    cfg.operator.hidden = 32
    cfg.operator.context_window = 2
    cfg.policy.inner_steps = 10
    cfg.policy.n_rollouts = 4
    cfg.policy.rollout_horizon = 8
    return cfg


@pytest.fixture(scope="module")
def fitted(tiny_cfg):
    """A tiny fitted operator + simulator + init states, shared across tests."""
    torch.manual_seed(0)
    sim = StructuralSimulator(tiny_cfg)
    pol = build_policy(tiny_cfg.policy)
    trans = collect(sim, pol, tiny_cfg, base_seed=1, generator=torch.Generator().manual_seed(1))
    torch.manual_seed(0)
    op = MarketResponseOperator(tiny_cfg)
    fit_operator(op, trans, pol, tiny_cfg.operator, generator=torch.Generator().manual_seed(2))
    init_states = collect_initial_states(sim, 4, base_seed=3)
    return sim, op, init_states, pol


def _final_central_spread(policy, state) -> float:
    with torch.no_grad():
        return float(policy.quote(state).half_spread.mean())


def test_analytic_correction_steers_the_gradient(tiny_cfg, fitted):
    """A large positive surrogate Delta must push the spread up relative to a
    large negative one -- the correction hook works end to end."""
    _, op, init_states, base_pol = fitted
    results = {}
    for name, delta in (("up", +50.0), ("down", -50.0)):
        pol = copy.deepcopy(base_pol)
        rgd_step(
            pol, op, init_states, tiny_cfg,
            frozen_summary=policy_summary(init_states[0], pol).detach(),
            n_steps=6, lr=0.05,
            generator=torch.Generator().manual_seed(11),
            analytic_correction=lambda h, d=delta: d,
            correction_scale=1.0,
        )
        results[name] = _final_central_spread(pol, init_states[0])
    assert results["up"] > results["down"]


def test_live_summary_optimization_runs(tiny_cfg, fitted):
    """PerfGD-learned path: gradients flow through the live summary without error."""
    _, op, init_states, base_pol = fitted
    pol = copy.deepcopy(base_pol)
    res = optimize_policy(
        pol, op, init_states, tiny_cfg,
        generator=torch.Generator().manual_seed(21),
        live_summary=True,
    )
    assert res.objective_history, "no optimisation steps ran"
    assert np.isfinite(res.final_objective)


def test_stability_penalty_wiring(tiny_cfg, fitted):
    """Turning a stability weight on must change the optimisation trajectory."""
    _, op, init_states, base_pol = fitted
    finals = {}
    for name, w in (("off", 0.0), ("on", 5.0)):
        cfg = copy.deepcopy(tiny_cfg)
        cfg.stability.toxicity_w = w
        pol = copy.deepcopy(base_pol)
        optimize_policy(
            pol, op, init_states, cfg,
            generator=torch.Generator().manual_seed(31),
        )
        finals[name] = _final_central_spread(pol, init_states[0])
    assert finals["on"] != finals["off"]


def test_run_loop_rejects_unknown_mode(tiny_cfg):
    with pytest.raises(ValueError):
        run_loop(tiny_cfg, mode="cobweb_of_lies")


def test_run_loop_smoke_all_modes(tiny_cfg):
    """Two iterations of each mode: iterates + seam diagnostics recorded."""
    for mode in ("rrm", "perfgd_analytic", "perfgd_learned"):
        res = run_loop(tiny_cfg, mode=mode, seed=0)
        assert res.mode == mode
        assert len(res.trajectory.iterates) >= 1
        assert len(res.diagnostics) == len(res.trajectory.iterates)
        d = res.diagnostics[-1]
        assert d.window_len >= 1
        assert np.isfinite(d.analytic_toxic_slope)
        if mode == "perfgd_analytic":
            assert d.correction != 0.0
        else:
            assert d.correction == 0.0


@pytest.mark.slow
def test_perfgd_loop_level_beyond_boundary_integration():
    """Loop-level integration in the *genuinely* RRM-unstable regime.

    Honest scope (documented in the results analysis): the closed-form 1.2
    claims -- the cobweb diverges at the fixed point, the corrected 1-D ascent
    converges to h_PO -- are verified in ``test_perfgd_loop.py``.  The full ML
    loops do NOT currently reproduce that stabilisation at any tested scale:
    the blind operator's implied dJ/dh diverges from the structural one, so
    the analytically-corrected equilibrium lands far from h_PO and no mode
    settles.  What this test locks in is the verified loop-level behaviour:
    all three modes run to completion in the unstable regime, the blind loop
    does not converge (the instability is real), and the ML<->math seam
    diagnostics stay finite and are recorded every iteration.

    NOTE the regime: the *default* microstructure's fixed point saturates
    below m = 1 for every on-grid gain (theory 1.1 Section 6.3); "beyond the
    boundary" requires the slow-toxic-decay regime below (theory 1.2 Section
    5), the same one the closed-form tests use.  The old version of this test
    asserted stabilisation at the default constants, where the fixed point is
    closed-form *stable* (m ~ 0.5) -- the "epsilon* ~ 1.3" it cited is the v2
    probe-point crossing, not the fixed-point boundary.
    """
    from reflex.theory.perfgd import analyze_perfgd

    cfg = load_config(CONFIG)
    # the genuinely unstable regime (m(h*) > 1): slow toxic decay
    cfg.clients.alpha = 1.0
    cfg.clients.toxicity_feedback = 6.0
    cfg.clients.info_intensity = 3.0
    cfg.clients.info_spread_decay = 0.8
    cfg.reward.quote_anchor_weight = 0.15
    # tiny loop settings
    cfg.bonds.n_bonds = 4
    cfg.simulator.horizon = 24
    cfg.rrm.n_episodes = 8
    cfg.rrm.eval_episodes = 4
    cfg.rrm.max_iters = 6
    cfg.rrm.update_rule = "rrm"
    cfg.rrm.tol = 5e-3
    cfg.operator.epochs = 25
    cfg.operator.context_window = 3
    cfg.policy.inner_steps = 30
    cfg.policy.n_rollouts = 8
    cfg.policy.rollout_horizon = 10

    # Closed form at this regime: fixed point unstable, corrected 1-D converges.
    closed = analyze_perfgd(cfg, run_loops=True, n_steps=120)
    assert closed.modulus_rrm > 1.0 and not closed.rrm_stable
    assert not closed.rrm_converged and closed.perfgd_converged

    results = {}
    for mode in ("rrm", "perfgd_analytic", "perfgd_learned"):
        results[mode] = run_loop(cfg, mode=mode, seed=0)

    # The blind loop must exhibit the instability (no convergence).
    assert not results["rrm"].trajectory.converged

    for mode, res in results.items():
        assert res_ok(res), f"{mode} produced no iterates"
        spreads = np.asarray(res.trajectory.central_spreads, dtype=float)
        assert np.isfinite(spreads).all(), f"{mode} produced non-finite spreads"
        assert len(res.diagnostics) == len(res.trajectory.iterates)
        an = res.analytic_slopes
        assert np.isfinite(an).all() and (an < 0.0).all(), (
            f"{mode}: analytic toxic slopes must stay finite and negative"
        )
        if mode == "perfgd_analytic":
            assert any(d.correction != 0.0 for d in res.diagnostics)
        else:
            assert all(d.correction == 0.0 for d in res.diagnostics)


def res_ok(res) -> bool:
    return len(res.trajectory.iterates) >= 2
