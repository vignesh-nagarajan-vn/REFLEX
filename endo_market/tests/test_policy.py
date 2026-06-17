"""Tests for dealer policies and the reward objective."""

from __future__ import annotations

import torch

from endo_market import seed_everything
from endo_market.config import load_config
from endo_market.env import StructuralSimulator
from endo_market.objective import reward, reward_from_components
from endo_market.policy import LinearPolicy, MLPPolicy, build_policy

CONFIG = "configs/default.yaml"


def _sim_and_state(seed: int = 0):
    seed_everything(seed)
    cfg = load_config(CONFIG)
    sim = StructuralSimulator(cfg)
    return cfg, sim, sim.reset(seed=1)


def test_quotes_valid_both_policies() -> None:
    cfg, sim, state = _sim_and_state()
    n = sim.bonds.n_bonds
    for typ in ("linear", "mlp"):
        cfg.policy.type = typ
        pol = build_policy(cfg.policy)
        q = pol.quote(state)
        assert q.half_spread.shape == (n,)
        assert q.skew.shape == (n,)
        assert torch.isfinite(q.half_spread).all()
        assert (q.half_spread > 0).all(), "half-spread must be positive"
        assert (q.half_spread <= cfg.policy.max_half_spread + 1e-5).all()
        assert torch.isfinite(q.skew).all()


def test_initial_half_spread_matches_config() -> None:
    cfg, sim, state = _sim_and_state()
    pol = LinearPolicy(cfg.policy)
    # With ~zero feature weights the central half-spread should equal init.
    q = pol.quote(state)
    assert torch.allclose(
        q.half_spread.mean(), torch.tensor(float(cfg.policy.init_half_spread)), atol=0.05
    )


def test_flatten_load_flat_roundtrip() -> None:
    cfg, _, _ = _sim_and_state()
    for typ in ("linear", "mlp"):
        cfg.policy.type = typ
        pol = build_policy(cfg.policy)
        flat = pol.flatten()
        assert flat.numel() == pol.n_params
        perturbed = flat + 0.3
        pol.load_flat(perturbed)
        assert torch.equal(pol.flatten(), perturbed)
        pol.load_flat(flat)
        assert torch.equal(pol.flatten(), flat)


def test_policy_is_differentiable() -> None:
    cfg, sim, state = _sim_and_state()
    cfg.policy.type = "linear"
    pol = build_policy(cfg.policy)
    loss = pol.quote(state).half_spread.sum() + pol.quote(state).skew.pow(2).sum()
    loss.backward()
    grads = [p.grad for p in pol.parameters() if p.grad is not None]
    assert grads, "no gradients populated"
    total = sum(g.abs().sum() for g in grads)
    assert torch.isfinite(torch.as_tensor(total)) and total > 0


def test_reward_components_and_markings() -> None:
    cfg, sim, state = _sim_and_state()
    pol = build_policy(cfg.policy)
    gen = torch.Generator().manual_seed(2)
    tr = sim.step(state, pol.quote(state), generator=gen)

    rb_fund = reward(tr, cfg.reward, marking="fundamental")
    rb_obs = reward(tr, cfg.reward, marking="observable")
    for rb in (rb_fund, rb_obs):
        assert torch.isfinite(rb.objective)
        assert torch.isfinite(rb.spread_capture)

    # reward() must agree with the components primitive (fundamental marking).
    rb_prim = reward_from_components(
        tr.pnl_components, tr.next_state.inventory, cfg.reward
    )
    assert torch.allclose(rb_prim.objective, rb_fund.objective)


def test_inventory_risk_penalty_reduces_objective() -> None:
    """A larger inventory-risk weight must not increase the objective when holding inventory."""
    cfg, sim, state = _sim_and_state()
    # Force some inventory by constructing components with nonzero inventory_after.
    comps = {
        "spread_capture": torch.tensor([1.0, 1.0]),
        "inventory_pnl": torch.tensor([0.0, 0.0]),
        "adverse_selection_loss": torch.tensor([0.0, 0.0]),
    }
    inv = torch.tensor([2.0, -3.0])
    cfg.reward.inv_risk_weight = 0.0
    base = reward_from_components(comps, inv, cfg.reward).objective
    cfg.reward.inv_risk_weight = 0.5
    penalised = reward_from_components(comps, inv, cfg.reward).objective
    assert penalised < base, "inventory-risk penalty should lower the objective"


def test_reward_rollout_finite() -> None:
    cfg, sim, state = _sim_and_state()
    pol = build_policy(cfg.policy)
    gen = torch.Generator().manual_seed(9)
    total = 0.0
    with torch.no_grad():
        for _ in range(cfg.simulator.horizon):
            tr = sim.step(state, pol.quote(state), generator=gen)
            total += float(reward(tr, cfg.reward).objective)
            state = tr.next_state
    assert torch.isfinite(torch.tensor(total))
