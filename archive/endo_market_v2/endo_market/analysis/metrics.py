"""Market-quality and toxicity metrics computed from ``T_true`` rollouts.

These summarise a deployed policy's realised behaviour on the structural market:
realised half-spread, fill rate, a VPIN-like toxicity share (informed volume as a
fraction of gross volume), inventory variance, and a Herfindahl (HHI)
concentration of volume across bonds.  They are diagnostics for interpreting the
stability results, not part of the optimisation.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..config import Config
from ..env.simulator import StructuralSimulator
from ..objective.reward import reward as reward_fn


@dataclass
class MarketMetrics:
    """Aggregate metrics over a rollout (means per step unless noted)."""

    realized_half_spread: float
    fill_rate: float  # gross volume per step per bond
    toxicity_share: float  # informed volume / gross volume (VPIN-like)
    adverse_loss_per_step: float
    spread_capture_per_step: float
    objective_per_step: float
    inventory_std: float  # std of inventory across the rollout
    hhi: float  # mean Herfindahl concentration of gross volume across bonds


def compute_metrics(
    cfg: Config,
    policy,
    seed: int = 0,
    n_episodes: int = 8,
    simulator: StructuralSimulator | None = None,
) -> MarketMetrics:
    """Roll ``policy`` on ``T_true`` for ``n_episodes`` and aggregate metrics."""
    if simulator is None:
        simulator = StructuralSimulator(cfg)
    gen = torch.Generator().manual_seed(seed + 13)
    horizon = cfg.simulator.horizon

    hs_sum = gross_sum = inf_sum = adv_sum = sc_sum = obj_sum = 0.0
    hhi_sum = 0.0
    hhi_n = 0
    n_steps = 0
    inv_vals = []

    with torch.no_grad():
        for ep in range(int(n_episodes)):
            state = simulator.reset(seed=seed + ep)
            for _ in range(horizon):
                q = policy.quote(state)
                tr = simulator.step(state, q, generator=gen)
                rb = reward_fn(tr, cfg.reward)
                hs_sum += float(q.half_spread.mean())
                gv = tr.fills.gross_volume
                gross_sum += float(gv.mean())
                inf_sum += float(tr.fills.informed_volume.sum())
                gross_total = float(gv.sum())
                adv_sum += float(tr.pnl_components["adverse_selection_loss"].sum())
                sc_sum += float(tr.pnl_components["spread_capture"].sum())
                obj_sum += float(rb.objective)
                if gross_total > 1e-8:
                    shares = gv / gv.sum()
                    hhi_sum += float((shares ** 2).sum())
                    hhi_n += 1
                inv_vals.append(float(tr.next_state.inventory.abs().mean()))
                n_steps += 1
                state = tr.next_state

    # toxicity share = total informed volume / total gross volume.
    # gross_sum accumulated mean-over-bonds per step; inf_sum accumulated
    # sum-over-bonds per step, so scale gross by n_bonds to get totals.
    n_bonds = cfg.bonds.n_bonds
    gross_total_volume = gross_sum * n_bonds
    toxicity_share = inf_sum / max(gross_total_volume, 1e-8)

    return MarketMetrics(
        realized_half_spread=hs_sum / max(n_steps, 1),
        fill_rate=gross_sum / max(n_steps, 1),
        toxicity_share=toxicity_share,
        adverse_loss_per_step=adv_sum / max(n_steps, 1),
        spread_capture_per_step=sc_sum / max(n_steps, 1),
        objective_per_step=obj_sum / max(n_steps, 1),
        inventory_std=float(torch.tensor(inv_vals).std()) if inv_vals else 0.0,
        hhi=hhi_sum / max(hhi_n, 1),
    )
