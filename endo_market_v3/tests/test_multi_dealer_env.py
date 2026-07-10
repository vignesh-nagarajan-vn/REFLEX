"""Tests for the genuine N-dealer environment and the simulated joint cobweb."""

from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

from reflex.config import load_config
from reflex.env import MultiDealerSimulator, StructuralSimulator
from reflex.equilibrium import measure_joint_modulus_sim, run_joint_cobweb_sim
from reflex.theory.multi_dealer import n_eff
from reflex.types import Quotes

CONFIG = "configs/default.yaml"


@pytest.fixture()
def cfg():
    cfg = load_config(CONFIG)
    cfg.bonds.n_bonds = 4
    cfg.simulator.horizon = 12
    return cfg


def _const_quotes(n_bonds: int, h: float) -> Quotes:
    return Quotes(half_spread=torch.full((n_bonds,), h), skew=torch.zeros(n_bonds))


# ------------------- exact single-dealer reduction (N = 1) ------------------ #
def test_n1_reduces_bitwise_to_single_dealer(cfg):
    """With N=1 the multi-dealer market must reproduce the single-dealer
    simulator bit for bit given the same seeds -- the correctness anchor."""
    cfg.clients.n_dealers = 1
    cfg.clients.toxic_spillover = 0.7  # must be irrelevant at N=1

    torch.manual_seed(0)
    single = StructuralSimulator(cfg)
    torch.manual_seed(0)
    multi = MultiDealerSimulator(cfg)

    s1 = single.reset(seed=42)
    sN = multi.reset(seed=42)
    assert torch.equal(s1.mid, sN.mid)
    assert torch.equal(s1.fundamental, sN.fundamental)

    g1 = torch.Generator().manual_seed(7)
    gN = torch.Generator().manual_seed(7)
    q = _const_quotes(cfg.bonds.n_bonds, 0.9)

    state1, stateN = s1, sN
    for _ in range(5):
        tr1 = single.step(state1, q, generator=g1)
        trN = multi.step(stateN, [q], generator=gN)
        d = trN.dealers[0]
        assert torch.allclose(tr1.fills.informed_volume, d.informed_volume, atol=1e-6)
        assert torch.allclose(tr1.fills.gross_volume, d.gross_volume, atol=1e-6)
        for key in ("spread_capture", "inventory_pnl", "adverse_selection_loss"):
            assert torch.allclose(
                tr1.pnl_components[key], d.pnl_components[key], atol=1e-6
            ), key
        assert torch.allclose(tr1.next_state.mid, trN.next_state.mid, atol=1e-6)
        assert torch.allclose(
            tr1.next_state.liquidity, trN.next_state.liquidity, atol=1e-6
        )
        state1, stateN = tr1.next_state, trN.next_state


# --------------------------- shared-pool mechanics -------------------------- #
def test_tighter_dealer_attracts_more_toxic_flow(cfg):
    """With two dealers and no spillover, the tighter dealer takes more of the
    informed pool on average.  (liq_flow_boost off so the combined-flow
    liquidity inflation does not push both dealers into the info_cap
    saturation -- the documented multi-dealer gotcha.)"""
    cfg.clients.n_dealers = 2
    cfg.clients.toxic_spillover = 0.0
    cfg.simulator.liq_flow_boost = 0.0
    torch.manual_seed(0)
    sim = MultiDealerSimulator(cfg)
    gen = torch.Generator().manual_seed(3)
    tox = np.zeros(2)
    state = sim.reset(seed=11)
    q_tight = _const_quotes(cfg.bonds.n_bonds, 0.5)
    q_wide = _const_quotes(cfg.bonds.n_bonds, 1.5)
    for _ in range(200):
        tr = sim.step(state, [q_tight, q_wide], generator=gen)
        tox += [float(d.informed_volume.sum()) for d in tr.dealers]
        state = tr.next_state
    assert tox[0] > 1.2 * tox[1]


def test_full_spillover_equalizes_toxicity(cfg):
    """At kappa=1 the toxic pool is common: both dealers face the same
    responsiveness regardless of their own spread."""
    cfg.clients.n_dealers = 2
    cfg.clients.toxic_spillover = 1.0
    cfg.simulator.liq_flow_boost = 0.0
    torch.manual_seed(0)
    sim = MultiDealerSimulator(cfg)
    gen = torch.Generator().manual_seed(3)
    tox = np.zeros(2)
    state = sim.reset(seed=11)
    q_tight = _const_quotes(cfg.bonds.n_bonds, 0.5)
    q_wide = _const_quotes(cfg.bonds.n_bonds, 1.5)
    for _ in range(300):
        tr = sim.step(state, [q_tight, q_wide], generator=gen)
        tox += [float(d.informed_volume.sum()) for d in tr.dealers]
        state = tr.next_state
    # same responsiveness -> toxic intake within noise of each other
    assert abs(tox[0] - tox[1]) < 0.15 * max(tox[0], tox[1])


def test_rejects_bad_inputs(cfg):
    cfg.clients.n_dealers = 2
    sim = MultiDealerSimulator(cfg)
    state = sim.reset(seed=0)
    with pytest.raises(ValueError):
        sim.step(state, [_const_quotes(cfg.bonds.n_bonds, 1.0)])  # one quote, two dealers
    cfg_bad = copy.deepcopy(cfg)
    cfg_bad.clients.toxic_spillover = 1.5
    with pytest.raises(ValueError):
        MultiDealerSimulator(cfg_bad)


# ----------------------- joint cobweb + modulus probes ---------------------- #
def test_joint_cobweb_runs_and_tracks_common_mode(cfg):
    cfg.clients.n_dealers = 3
    cfg.clients.toxic_spillover = 0.5
    res = run_joint_cobweb_sim(cfg, n_iters=4, seed=0, n_episodes=2)
    assert len(res.h_history) >= 2
    assert res.common_mode.shape[0] == len(res.h_history)
    assert np.isfinite(res.common_mode).all()


@pytest.mark.slow
def test_joint_modulus_amplifies_with_dealers(cfg):
    """The measured in-phase (common-mode) modulus must exceed the single-dealer
    modulus and move with N_eff, and dominate the differential modulus."""
    cfg.clients.toxic_spillover = 1.0
    cfg_1 = copy.deepcopy(cfg)
    cfg_1.clients.n_dealers = 1
    cfg_3 = copy.deepcopy(cfg)
    cfg_3.clients.n_dealers = 3

    m1 = measure_joint_modulus_sim(cfg_1, h_ref=1.0, seed=0, n_episodes=4)
    m3 = measure_joint_modulus_sim(cfg_3, h_ref=1.0, seed=0, n_episodes=4)

    assert m3.modulus_common > 1.5 * m1.modulus_common  # theory: x N_eff = 3
    ratio = m3.modulus_common / max(m1.modulus_common, 1e-9)
    expected = n_eff(3, 1.0)  # = 3
    assert 0.5 * expected < ratio < 2.0 * expected
    # common mode dominates the differential mode at full spillover
    assert m3.modulus_common > m3.modulus_differential
