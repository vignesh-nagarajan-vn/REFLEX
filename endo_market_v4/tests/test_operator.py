"""Tests for the Market Response Operator ``T_theta`` and its fitting.

The headline assertion (required by the build spec) is that fitting the operator
reduces the **held-out** negative log-likelihood relative to the untrained
baseline -- i.e. the operator actually learns the response distribution.
"""

from __future__ import annotations

import torch

from reflex import seed_everything
from reflex.config import load_config
from reflex.env import StructuralSimulator
from reflex.equilibrium import build_dataset, fit_operator
from reflex.operator import OUT_DIM, MarketResponseOperator
from reflex.operator.heads import GaussianHead, MixtureHead
from reflex.policy import build_policy
from reflex.types import Quotes, policy_summary

CONFIG = "configs/default.yaml"


def _collect(sim, pol, n_episodes, horizon, jitter=0.05, seed=99):
    gen = torch.Generator().manual_seed(seed)
    trans = []
    for ep in range(n_episodes):
        state = sim.reset(seed=1000 + ep)
        for _ in range(horizon):
            q = pol.quote(state)
            hj = (q.half_spread + jitter * torch.randn(sim.bonds.n_bonds, generator=gen)).clamp_min(0.01)
            q = Quotes(half_spread=hj.detach(), skew=q.skew.detach())
            tr = sim.step(state, q, generator=gen)
            trans.append(tr)
            state = tr.next_state.detach()
    return trans


def test_heads_shapes_and_logprob() -> None:
    torch.manual_seed(0)
    x = torch.randn(16, 12)
    for head in (
        GaussianHead(12, OUT_DIM),
        GaussianHead(12, OUT_DIM, full_cov=True),
        MixtureHead(12, OUT_DIM, n_components=3),
    ):
        d = head(x)
        target = torch.randn(16, OUT_DIM)
        lp = d.log_prob(target)
        assert lp.shape == (16,)
        assert torch.isfinite(lp).all()
        s = head.rsample(x)
        assert s.shape == (16, OUT_DIM)
        assert torch.isfinite(s).all()


def test_build_features_shape() -> None:
    cfg = load_config(CONFIG)
    sim = StructuralSimulator(cfg)
    pol = build_policy(cfg.policy)
    state = sim.reset(seed=1)
    q = pol.quote(state)
    summ = policy_summary(state, pol)
    feats = MarketResponseOperator.build_features(state, q, summ)
    assert feats.shape == (sim.bonds.n_bonds, 8)
    assert torch.isfinite(feats).all()


def test_fit_reduces_heldout_nll() -> None:
    """Core requirement: fitting must lower the held-out NLL vs the untrained baseline."""
    seed_everything(0)
    cfg = load_config(CONFIG)
    sim = StructuralSimulator(cfg)
    pol = build_policy(cfg.policy)
    trans = _collect(sim, pol, cfg.rrm.n_episodes, cfg.simulator.horizon)

    op = MarketResponseOperator(cfg)
    gen = torch.Generator().manual_seed(7)
    res = fit_operator(op, trans, pol, cfg.operator, generator=gen)
    assert res.n_rows > 0
    assert res.best_val_nll < res.baseline_val_nll, (
        f"held-out NLL did not improve: baseline={res.baseline_val_nll:.3f} "
        f"best={res.best_val_nll:.3f}"
    )


def test_dataset_target_channels() -> None:
    cfg = load_config(CONFIG)
    sim = StructuralSimulator(cfg)
    pol = build_policy(cfg.policy)
    trans = _collect(sim, pol, 4, cfg.simulator.horizon)
    X, Y = build_dataset(trans, pol)
    n = sim.bonds.n_bonds
    assert X.shape[0] == Y.shape[0] == 4 * cfg.simulator.horizon * n
    assert X.shape[1] == 8
    assert Y.shape[1] == OUT_DIM


def test_rollout_differentiable_and_finite() -> None:
    seed_everything(0)
    cfg = load_config(CONFIG)
    sim = StructuralSimulator(cfg)
    pol = build_policy(cfg.policy)
    op = MarketResponseOperator(cfg)

    init = sim.reset(seed=1)
    psum = policy_summary(init, pol).detach()
    gen = torch.Generator().manual_seed(3)
    roll = op.rollout(
        init, pol, horizon=cfg.policy.rollout_horizon, reward_cfg=cfg.reward,
        policy_summary_override=psum, generator=gen,
    )
    assert torch.isfinite(roll.objective)
    assert len(roll.states) == cfg.policy.rollout_horizon + 1

    (-roll.objective).backward()
    grads = [p.grad for p in pol.parameters() if p.grad is not None]
    total = sum(g.abs().sum() for g in grads)
    assert total > 0, "no pathwise gradient reached the policy"


def test_rollout_reproducible_with_generator() -> None:
    seed_everything(0)
    cfg = load_config(CONFIG)
    sim = StructuralSimulator(cfg)
    pol = build_policy(cfg.policy)
    op = MarketResponseOperator(cfg)
    init = sim.reset(seed=1)
    psum = policy_summary(init, pol).detach()

    def run():
        gen = torch.Generator().manual_seed(123)
        return op.rollout(
            init, pol, horizon=8, reward_cfg=cfg.reward,
            policy_summary_override=psum, generator=gen,
        ).objective

    a = run()
    b = run()
    assert torch.equal(a, b), "rollout not reproducible under a fixed generator"
