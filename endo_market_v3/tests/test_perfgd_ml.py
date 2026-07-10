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
def test_perfgd_analytic_stabilizes_beyond_rrm_boundary():
    """The headline 1.2 demonstration at loop level: past epsilon* the blind
    best-response loop oscillates/diverges while the analytically-corrected
    loop settles.  Uses the full best-response update rule (the cobweb)."""
    cfg = load_config(CONFIG)
    cfg.bonds.n_bonds = 4
    cfg.simulator.horizon = 24
    cfg.rrm.n_episodes = 8
    cfg.rrm.eval_episodes = 4
    cfg.rrm.max_iters = 8
    cfg.rrm.update_rule = "rrm"
    cfg.rrm.tol = 5e-3
    cfg.operator.epochs = 25
    cfg.operator.context_window = 3
    cfg.policy.inner_steps = 30
    cfg.policy.n_rollouts = 8
    cfg.policy.rollout_horizon = 10
    cfg.clients.toxicity_feedback = 6.0  # far beyond epsilon* ~ 1.3

    blind = run_loop(cfg, mode="rrm", seed=0)
    corrected = run_loop(cfg, mode="perfgd_analytic", seed=0)

    # Compare late-stage step sizes: the corrected loop must be materially
    # calmer than the blind cobweb (which oscillates or blows up).
    def late_step(res):
        steps = res.trajectory.step_sizes
        return float(np.mean(steps[len(steps) // 2:])) if steps.size else float("inf")

    assert res_ok(blind), "blind run produced no iterates"
    assert res_ok(corrected), "corrected run produced no iterates"
    assert (
        corrected.trajectory.converged
        or late_step(corrected) < 0.5 * late_step(blind)
    ), (
        f"correction did not stabilise: corrected late-step {late_step(corrected):.4f} "
        f"vs blind {late_step(blind):.4f}"
    )


def res_ok(res) -> bool:
    return len(res.trajectory.iterates) >= 2
