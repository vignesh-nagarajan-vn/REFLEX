"""Collect data from the structural simulator ``T_true`` under a fixed policy.

Each RRM iteration deploys the current policy, plays a number of episodes through
the *real* market, and records the transitions used to refit the operator.  A
small amount of **exploration jitter** is added to the deployed half-spread so
the dataset contains local variation in the action -- without it the operator
could not learn how the P&L channels (in particular the adverse-selection cost)
respond to the quoted spread, which is exactly the local sensitivity the dealer
exploits when optimising.

Collection runs under ``torch.no_grad`` and detaches everything: these
transitions are *data*, not part of any computation graph.
"""

from __future__ import annotations

from typing import List

import torch

from ..config import Config
from ..env.simulator import StructuralSimulator
from ..types import Quotes, Transition


def collect(
    simulator: StructuralSimulator,
    policy,
    cfg: Config,
    base_seed: int = 0,
    generator: torch.Generator | None = None,
) -> List[Transition]:
    """Play ``cfg.rrm.n_episodes`` episodes under ``policy`` and return transitions.

    Parameters
    ----------
    simulator:
        The structural ground-truth market.
    policy:
        The deployed dealer policy (used in eval / no-grad mode here).
    cfg:
        Full config (uses ``rrm.n_episodes``, ``rrm.collection_jitter`` and
        ``simulator.horizon``).
    base_seed:
        Episodes are reset with ``base_seed + episode_index`` so different RRM
        iterations can use disjoint, reproducible episode seeds.
    generator:
        RNG for the exploration jitter and the simulator steps.

    Returns
    -------
    list of Transition
        Detached transitions ready for operator fitting.
    """
    if generator is None:
        generator = torch.Generator().manual_seed(base_seed)
    n_bonds = simulator.bonds.n_bonds
    horizon = cfg.simulator.horizon
    jitter = float(cfg.rrm.collection_jitter)

    transitions: List[Transition] = []
    with torch.no_grad():
        for ep in range(int(cfg.rrm.n_episodes)):
            state = simulator.reset(seed=base_seed + ep)
            for _ in range(horizon):
                q = policy.quote(state)
                if jitter > 0:
                    noise = jitter * torch.randn(n_bonds, generator=generator)
                    hs = (q.half_spread + noise).clamp_min(0.01)
                    q = Quotes(half_spread=hs, skew=q.skew)
                tr = simulator.step(state, q, generator=generator)
                # Detach the whole transition -- it is fixed data.
                transitions.append(
                    Transition(
                        state=tr.state.detach(),
                        quotes=Quotes(
                            half_spread=tr.quotes.half_spread.detach(),
                            skew=tr.quotes.skew.detach(),
                        ),
                        fills=tr.fills,
                        next_state=tr.next_state.detach(),
                        pnl_components={
                            k: v.detach() for k, v in tr.pnl_components.items()
                        },
                    )
                )
                state = tr.next_state
    return transitions


def collect_initial_states(
    simulator: StructuralSimulator,
    n_states: int,
    base_seed: int = 0,
):
    """Return a list of ``n_states`` fresh initial states (for rollout starts)."""
    return [simulator.reset(seed=base_seed + i).detach() for i in range(n_states)]
