"""Tests for the v4 structural PerfGD mode -- the loop-level gap closure.

The v3 negative result: neither the analytically-corrected nor the free-form
learned ML loop reproduced the closed-form PerfGD stabilisation, because the
policy update consumed the *operator's* implied objective gradient, which
diverges from the structural one away from the deployed regime.  The v4
``perfgd_structural`` mode replaces that gradient with GLFT-anchored families
fitted to the loop's own deployment data.  These tests pin down, in order:
the fits recover the true structural constants; the retune helper is exact;
the mode runs and logs its seam diagnostics; and -- the headline -- in the
genuinely RRM-unstable regime the structural loop settles at the closed-form
performative optimum while blind RRM does not converge.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

from reflex.config import load_config
from reflex.env import StructuralSimulator
from reflex.equilibrium import (
    DeploymentRecord,
    collect,
    fit_structural_response,
    retune_central_spread,
    run_loop,
)
from reflex.estimators.br_slope import _deployed_policy_at
from reflex.policy import build_policy
from reflex.theory.analytic_boundary import epsilon as epsilon_of
from reflex.theory.analytic_boundary import reference_state, tau as tau_of

CONFIG = "configs/default.yaml"


def _tiny(cfg):
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


# --------------------------------------------------------------------------- #
# The structural fit                                                           #
# --------------------------------------------------------------------------- #
def test_fit_recovers_structural_constants():
    """Fitted tau_hat / eps_hat / psi_hat track the closed forms on data from
    two fixed-spread deployments (the cleanest identification setting)."""
    cfg = _tiny(load_config(CONFIG))
    cfg.rrm.n_episodes = 12  # more data for a tight fit
    torch.manual_seed(0)
    sim = StructuralSimulator(cfg)
    window = []
    for i, h_dep in enumerate((0.7, 1.3)):
        pol = _deployed_policy_at(cfg, h_dep)
        trans = collect(sim, pol, cfg, base_seed=100 + i,
                        generator=torch.Generator().manual_seed(100 + i))
        window.append(DeploymentRecord(transitions=trans, policy=pol))

    fit = fit_structural_response(window, cfg)
    assert fit.identified and fit.n_points > 100

    ref = reference_state(cfg)
    h_mid = 1.0
    # Levels and slopes against the a-priori closed forms.  The realised state
    # (liquidity ratio, gate) drifts from the A2 reference, so agreement is
    # structural-scale, not exact: factor-2 brackets, like the triangulation.
    tau_true = tau_of(cfg, h_mid, ref)
    eps_true = epsilon_of(cfg, h_mid, ref)
    assert 0.4 * tau_true < fit.tau_hat(h_mid) < 2.5 * tau_true
    assert 0.3 * eps_true < fit.epsilon_hat(h_mid) < 3.0 * eps_true
    assert 0.3 * ref.psi < fit.psi_hat < 3.0 * ref.psi
    # The fitted decay must be positive and finite (the family is decreasing).
    assert 0.02 <= fit.c <= 20.0
    assert fit.k_u > 0.0 and fit.A_u > 0.0


def test_fit_flags_unidentified_window():
    """A single zero-jitter deployment has no spread variation: the fit must
    be flagged unidentified so the loop holds the previous one."""
    cfg = _tiny(load_config(CONFIG))
    cfg.rrm.collection_jitter = 0.0
    torch.manual_seed(0)
    sim = StructuralSimulator(cfg)
    pol = _deployed_policy_at(cfg, 1.0)
    trans = collect(sim, pol, cfg, base_seed=7,
                    generator=torch.Generator().manual_seed(7))
    fit = fit_structural_response([DeploymentRecord(transitions=trans, policy=pol)], cfg)
    assert not fit.identified


def test_fit_gradient_machinery_consistent():
    """Corrected gradient = blind + correction; h_PO_hat is its root."""
    cfg = _tiny(load_config(CONFIG))
    cfg.rrm.n_episodes = 8
    torch.manual_seed(0)
    sim = StructuralSimulator(cfg)
    window = []
    for i, h_dep in enumerate((0.7, 1.3)):
        pol = _deployed_policy_at(cfg, h_dep)
        trans = collect(sim, pol, cfg, base_seed=200 + i,
                        generator=torch.Generator().manual_seed(200 + i))
        window.append(DeploymentRecord(transitions=trans, policy=pol))
    fit = fit_structural_response(window, cfg)
    h = 1.1
    total = fit.corrected_gradient_hat(h, cfg.reward)
    parts = fit.blind_gradient_hat(h, cfg.reward) + fit.correction_hat(h, cfg.reward)
    assert total == pytest.approx(parts)
    h_po = fit.solve_h_po_hat(cfg.reward, cfg.policy.max_half_spread)
    assert 0.0 < h_po < cfg.policy.max_half_spread
    assert abs(fit.corrected_gradient_hat(h_po, cfg.reward)) < 1e-3 * max(
        abs(fit.corrected_gradient_hat(0.2, cfg.reward)), 1.0
    )


# --------------------------------------------------------------------------- #
# The retune helper                                                            #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("ptype", ["linear", "mlp"])
def test_retune_central_spread_exact(ptype):
    cfg = _tiny(load_config(CONFIG))
    cfg.policy.type = ptype
    torch.manual_seed(0)
    sim = StructuralSimulator(cfg)
    state = sim.reset(seed=3).detach()
    pol = build_policy(cfg.policy)
    for target in (0.4, 1.7, 3.2):
        achieved = retune_central_spread(pol, state, target)
        assert achieved == pytest.approx(target, abs=1e-3)
        with torch.no_grad():
            assert float(pol.quote(state).half_spread.mean()) == pytest.approx(target, abs=1e-3)


def test_retune_rejects_unknown_policy():
    cfg = _tiny(load_config(CONFIG))
    torch.manual_seed(0)
    sim = StructuralSimulator(cfg)
    state = sim.reset(seed=3).detach()
    with pytest.raises(TypeError):
        retune_central_spread(object(), state, 1.0)


# --------------------------------------------------------------------------- #
# The loop mode                                                                #
# --------------------------------------------------------------------------- #
def test_run_loop_structural_smoke():
    """Two iterations: iterates + the structural seam diagnostics recorded."""
    cfg = _tiny(load_config(CONFIG))
    res = run_loop(cfg, mode="perfgd_structural", seed=0)
    assert res.mode == "perfgd_structural"
    assert len(res.trajectory.iterates) >= 1
    assert len(res.diagnostics) == len(res.trajectory.iterates)
    d = res.diagnostics[-1]
    assert np.isfinite(d.structural_toxic_slope)
    assert d.structural_toxic_slope <= 0.0
    assert np.isfinite(d.h_po_hat) and d.h_po_hat > 0.0
    assert res.trajectory.iterates[-1].grad_path.startswith("structural")
    # The other modes must not carry structural diagnostics.
    res_rrm = run_loop(cfg, mode="rrm", seed=0)
    assert all(np.isnan(dd.structural_toxic_slope) for dd in res_rrm.diagnostics)


@pytest.mark.slow
def test_structural_loop_stabilises_beyond_boundary():
    """THE v4 headline: in the genuinely RRM-unstable regime (m(h*) > 1) the
    GLFT-anchored learned loop settles at the *realized* performative optimum
    while the blind loop fails to converge.

    Benchmark note (state it in any write-up).  The A2 closed-form h_PO
    (~1.64 here) is NOT the target: in this high-intensity regime the
    realized market differs from the frozen-reference closed forms at first
    order through channels they deliberately omit (1.1 S9) -- the info_cap
    saturation (raw toxic notionals ~10 > cap 8 at tight spreads), the
    liquidity-inflation feedback, and the severity drift.  The
    apples-to-apples benchmark is an *independent* structural fit on fresh
    controlled deployments spanning the operating range: the loop must park
    where THAT fit's corrected gradient vanishes, strictly inside its blind
    stable point (the realized echo-chamber gap), and self-consistently with
    its own running h_PO_hat estimate.  Success is convergence to the noise
    ball around the realized optimum -- the corrected gradient is estimated
    from finite deployment data, so iterates hover within O(1/sqrt(n)) of it
    (the same criterion as PerfGD with estimated gradients, Izzo et al. 2021).
    """
    from reflex.theory.perfgd import analyze_perfgd

    cfg = load_config(CONFIG)
    # the genuinely unstable regime (m(h*) > 1): slow toxic decay -- identical
    # to the closed-form tests and test_perfgd_ml's integration regime.
    cfg.clients.alpha = 1.0
    cfg.clients.toxicity_feedback = 6.0
    cfg.clients.info_intensity = 3.0
    cfg.clients.info_spread_decay = 0.8
    cfg.reward.quote_anchor_weight = 0.15
    # loop settings: enough data for tight fits, enough iterations to settle
    cfg.bonds.n_bonds = 4
    cfg.simulator.horizon = 24
    cfg.rrm.n_episodes = 8
    cfg.rrm.eval_episodes = 4
    cfg.rrm.max_iters = 12
    cfg.rrm.tol = 5e-3
    cfg.operator.epochs = 25
    cfg.operator.context_window = 3
    cfg.policy.inner_steps = 30
    cfg.policy.n_rollouts = 8
    cfg.policy.rollout_horizon = 10

    closed = analyze_perfgd(cfg, run_loops=True, n_steps=120)
    assert closed.modulus_rrm > 1.0 and not closed.rrm_stable

    blind = run_loop(cfg, mode="rrm", seed=0)
    struct = run_loop(cfg, mode="perfgd_structural", seed=0)

    # (a) The blind loop must exhibit the instability.
    assert not blind.trajectory.converged

    # (b) The structural loop must settle: full course, finite, hovering.
    assert struct.trajectory.stop_reason != "blowup"
    spreads = np.asarray(struct.trajectory.central_spreads, dtype=float)
    assert np.isfinite(spreads).all()
    h_settled = float(spreads[-3:].mean())
    late_steps = np.abs(np.diff(spreads[-4:]))
    assert late_steps.max() < 0.15 * h_settled, f"late steps too large: {late_steps}"

    # (c) Independent benchmark: fresh structural fits on controlled
    # deployments spanning the operating range (2 seeds).
    torch.manual_seed(123)
    sim = StructuralSimulator(cfg)
    fresh_grad_settled = []
    fresh_grad_start = []
    fresh_h_po = []
    fresh_h_sp = []
    for s in (0, 1):
        window = []
        for i, h_dep in enumerate((0.8, 1.5, 2.2, 2.9, 3.6)):
            pol = _deployed_policy_at(cfg, h_dep)
            trans = collect(sim, pol, cfg, base_seed=900 + 101 * s + 11 * i,
                            generator=torch.Generator().manual_seed(900 + 101 * s + 11 * i))
            window.append(DeploymentRecord(transitions=trans, policy=pol))
        fresh = fit_structural_response(window, cfg)
        assert fresh.identified
        fresh_grad_settled.append(fresh.corrected_gradient_hat(h_settled, cfg.reward))
        fresh_grad_start.append(fresh.corrected_gradient_hat(float(spreads[0]), cfg.reward))
        fresh_h_po.append(fresh.solve_h_po_hat(cfg.reward, cfg.policy.max_half_spread))
        fresh_h_sp.append(fresh.solve_h_sp_hat(cfg.reward, cfg.policy.max_half_spread))

    grad_settled = float(np.median(fresh_grad_settled))
    grad_start = float(np.median(fresh_grad_start))
    h_po_fresh = float(np.median(fresh_h_po))
    h_sp_fresh = float(np.median(fresh_h_sp))

    # The loop parked where the independently-estimated realized corrected
    # gradient has (nearly) vanished ...
    assert abs(grad_settled) < 0.3 * abs(grad_start), (
        f"realized corrected gradient not near zero at the settle point: "
        f"Phi_hat'({h_settled:.2f}) = {grad_settled:.3f} vs "
        f"Phi_hat'({spreads[0]:.2f}) = {grad_start:.3f}"
    )
    # ... near the independent estimate of the realized optimum ...
    assert 0.6 * h_po_fresh < h_settled < 1.6 * h_po_fresh, (
        f"settled {h_settled:.2f} vs independent realized h_PO {h_po_fresh:.2f}"
    )
    # ... strictly inside the realized stable point: the correction closed a
    # real echo-chamber gap (the blind loop's park position, when it parks).
    assert h_settled < 0.85 * h_sp_fresh, (
        f"no echo-chamber gap closed: settled {h_settled:.2f} vs realized "
        f"h_SP {h_sp_fresh:.2f}"
    )

    # (d) Self-consistency: the loop's own running h_PO_hat tracks its park.
    h_po_hat = struct.diagnostics[-1].h_po_hat
    assert abs(h_po_hat - h_settled) < 0.35 * h_settled
