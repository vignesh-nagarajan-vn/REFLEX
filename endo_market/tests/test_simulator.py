"""Tests for the structural ground-truth simulator ``T_true``.

These cover the structural invariants we rely on downstream: reproducibility,
finiteness of P&L, the exact P&L decomposition identity, and the *sign* of the
toxicity-vs-spread feedback that the whole convergence study is built on.
"""

from __future__ import annotations

import torch

from endo_market import seed_everything
from endo_market.config import load_config
from endo_market.env import StructuralSimulator
from endo_market.types import Quotes

CONFIG = "configs/default.yaml"


def _fixed_quotes(n: int, h: float) -> Quotes:
    return Quotes(half_spread=torch.full((n,), float(h)), skew=torch.zeros(n))


def _rollout(alpha: float, h: float, steps: int, seed: int = 0):
    """Roll a constant-spread policy and return summed P&L components + flow."""
    seed_everything(seed)
    cfg = load_config(CONFIG)
    cfg.clients.alpha = alpha
    sim = StructuralSimulator(cfg)
    gen = torch.Generator().manual_seed(123)
    state = sim.reset(seed=1)
    n = sim.bonds.n_bonds
    totals = {"spread_capture": 0.0, "inventory_pnl": 0.0, "adverse_selection_loss": 0.0}
    informed = 0.0
    for _ in range(steps):
        tr = sim.step(state, _fixed_quotes(n, h), generator=gen)
        for k in totals:
            totals[k] += tr.pnl_components[k].sum().item()
        informed += tr.fills.informed_volume.sum().item()
        state = tr.next_state
    return totals, informed


def test_reset_shapes_and_finite() -> None:
    cfg = load_config(CONFIG)
    sim = StructuralSimulator(cfg)
    n = sim.bonds.n_bonds
    state = sim.reset(seed=3)
    for name in ("inventory", "mid", "fundamental", "liquidity", "flow_recent", "vol_recent"):
        t = getattr(state, name)
        assert t.shape == (n,), f"{name} has wrong shape {t.shape}"
        assert torch.isfinite(t).all(), f"{name} not finite"
    assert torch.equal(state.inventory, torch.zeros(n))
    assert (state.liquidity > 0).all()


def test_step_pnl_finite() -> None:
    cfg = load_config(CONFIG)
    sim = StructuralSimulator(cfg)
    n = sim.bonds.n_bonds
    gen = torch.Generator().manual_seed(0)
    state = sim.reset(seed=0)
    for _ in range(64):
        tr = sim.step(state, _fixed_quotes(n, 0.5), generator=gen)
        for v in tr.pnl_components.values():
            assert torch.isfinite(v).all()
        assert torch.isfinite(tr.next_state.inventory).all()
        assert torch.isfinite(tr.next_state.mid).all()
        assert torch.isfinite(tr.next_state.liquidity).all()
        state = tr.next_state


def test_determinism() -> None:
    """Same seed -> identical transition; the simulator must be reproducible."""
    def one():
        seed_everything(11)
        cfg = load_config(CONFIG)
        sim = StructuralSimulator(cfg)
        n = sim.bonds.n_bonds
        gen = torch.Generator().manual_seed(5)
        state = sim.reset(seed=2)
        tr = sim.step(state, _fixed_quotes(n, 0.4), generator=gen)
        return tr.next_state.mid, tr.pnl_components["spread_capture"]

    mid_a, sc_a = one()
    mid_b, sc_b = one()
    assert torch.equal(mid_a, mid_b)
    assert torch.equal(sc_a, sc_b)


def test_pnl_decomposition_identity() -> None:
    """The three components must sum to total economic P&L exactly.

    total = delta_cash + (q_after * v_next - q_before * v)
          = spread_capture + inventory_pnl - adverse_selection_loss.
    """
    cfg = load_config(CONFIG)
    sim = StructuralSimulator(cfg)
    n = sim.bonds.n_bonds
    gen = torch.Generator().manual_seed(7)
    state = sim.reset(seed=4)
    for _ in range(20):
        q = Quotes(half_spread=torch.full((n,), 0.5), skew=0.1 * torch.randn(n))
        tr = sim.step(state, q, generator=gen)
        S = tr.fills  # noqa: F841 (use components directly below)

        h = q.half_spread
        skew = q.skew
        v = state.fundamental
        m = state.mid
        q0 = state.inventory
        d_inv = tr.fills.qty
        q_after = q0 + d_inv
        v_next = tr.next_state.fundamental
        bid = m + skew - h
        ask = m + skew + h
        # Recover S, B from net and gross is not unique; instead reconstruct the
        # cash flow from the named components and check the accounting identity.
        comp_total = (
            tr.pnl_components["spread_capture"]
            + tr.pnl_components["inventory_pnl"]
            - tr.pnl_components["adverse_selection_loss"]
        )
        # Independent ground truth: inventory value change marked to fundamental
        # plus realised cash flow. delta_cash = -d_inv*(m) - spread paid? Build it
        # from the canonical identity total == (S+B)h + (S-B)skew + q_after*(v'-v)
        #   + (B-S)(v-m).  We can get (S-B) = -d_inv and (S+B) from gross volume.
        gross = tr.fills.gross_volume
        S_minus_B = -d_inv
        S_plus_B = gross
        independent_total = (
            S_plus_B * h
            + S_minus_B * skew
            + q_after * (v_next - v)
            + (-S_minus_B) * (v - m)
        )
        assert torch.allclose(comp_total, independent_total, atol=1e-4), (
            "P&L decomposition identity violated"
        )
        state = tr.next_state


def test_toxicity_feedback_sign() -> None:
    """Core mechanism: tighter spreads and higher alpha both raise adverse selection."""
    # Tighter spread -> more adverse selection (fixed alpha).
    tight, _ = _rollout(alpha=0.5, h=0.30, steps=300)
    wide, _ = _rollout(alpha=0.5, h=0.70, steps=300)
    assert tight["adverse_selection_loss"] > wide["adverse_selection_loss"], (
        "tighter spread should admit more toxic flow"
    )

    # Higher alpha -> more adverse selection (fixed spread).
    low, low_inf = _rollout(alpha=0.1, h=0.45, steps=300)
    high, high_inf = _rollout(alpha=0.9, h=0.45, steps=300)
    assert high["adverse_selection_loss"] > low["adverse_selection_loss"], (
        "higher alpha should produce more adverse selection"
    )
    assert high_inf > low_inf, "higher alpha should produce more informed volume"
