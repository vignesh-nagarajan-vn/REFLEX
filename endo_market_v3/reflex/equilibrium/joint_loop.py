"""Joint ``N``-dealer retraining dynamics on the genuine multi-dealer market.

The 1.3 theory predicts the joint best-response cobweb's spectrum: a common
mode with modulus ``m_N = N_eff * m_1`` (``N_eff = 1 + kappa*(N-1)``) and
``N-1`` differential modes at ``(1-kappa) * m_1``.  This module measures those
predictions with *simulation in the loop*: the toxic environment each dealer
best-responds to is realised by :class:`~reflex.env.multi_dealer.MultiDealerSimulator`
-- the genuine shared-pool market -- not by the closed-form ``tau``.

Two instruments:

* :func:`run_joint_cobweb_sim` -- iterate the joint frozen-environment best
  response ``h^{k+1}_i = BR(tau_hat_i(h^k))`` where ``tau_hat_i`` is dealer
  ``i``'s *measured* mean toxic notional at the deployed spread vector.  The
  common-mode iterates converge iff ``m_N < 1``.
* :func:`measure_joint_modulus_sim` -- common-random-number probes of the joint
  BR map: an **in-phase** perturbation (all dealers ``h +/- delta``) measures
  the common-mode modulus (theory: ``N_eff * m_1``); an **anti-phase**
  perturbation (dealer 0 at ``h+delta``, dealer 1 at ``h-delta``) measures the
  differential modulus (theory: ``(1-kappa) * m_1``).

Both are cheap (no operator fits): the frozen-``T`` best response given a
measured toxic level is the 1-D closed-form FOC root of theory 1.1, while the
toxic level itself comes from the real environment.  The fully-learned
``N``-dealer loop (per-dealer operators + policies) is deliberately out of
scope here and flagged as future work.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch

from ..config import Config
from ..env.multi_dealer import MultiDealerSimulator
from ..theory.analytic_boundary import ReferenceState, best_response, reference_state
from ..types import Quotes


def _measure_toxic_levels(
    cfg: Config,
    simulator: MultiDealerSimulator,
    h_vec: np.ndarray,
    seed: int,
    n_episodes: int,
) -> np.ndarray:
    """Mean per-step per-bond toxic notional per dealer at a deployed spread vector.

    Seeding depends only on ``seed`` (not on ``h_vec``), so successive calls
    share their randomness -- common random numbers across cobweb iterations
    and probe arms.
    """
    n = simulator.bonds.n_bonds
    gen = torch.Generator().manual_seed(seed)
    horizon = int(cfg.simulator.horizon)
    totals = np.zeros(len(h_vec))
    n_steps = 0
    with torch.no_grad():
        for ep in range(int(n_episodes)):
            state = simulator.reset(seed=seed + 71 * ep)
            for _ in range(horizon):
                quotes = [
                    Quotes(
                        half_spread=torch.full((n,), float(h)),
                        skew=torch.zeros(n),
                    )
                    for h in h_vec
                ]
                tr = simulator.step(state, quotes, generator=gen)
                for i, d in enumerate(tr.dealers):
                    totals[i] += float(d.informed_volume.sum()) / n
                n_steps += 1
                state = tr.next_state
    return totals / max(n_steps, 1)


@dataclass
class JointCobwebResult:
    """Trajectory of the simulated joint best-response iteration."""

    h_history: List[np.ndarray] = field(default_factory=list)  # [k][N]
    converged: bool = False
    common_mode: np.ndarray = field(default_factory=lambda: np.array([]))  # mean_i h_i per k
    spread_across_dealers: np.ndarray = field(default_factory=lambda: np.array([]))

    @property
    def final_h(self) -> np.ndarray:
        return self.h_history[-1] if self.h_history else np.array([])


def run_joint_cobweb_sim(
    cfg: Config,
    h0: Optional[np.ndarray] = None,
    n_iters: int = 12,
    seed: int = 0,
    n_episodes: int = 4,
    tol: float = 1e-4,
    ref: Optional[ReferenceState] = None,
    simulator: Optional[MultiDealerSimulator] = None,
) -> JointCobwebResult:
    """Iterate the joint frozen-environment best response on the genuine market."""
    if simulator is None:
        torch.manual_seed(seed)
        simulator = MultiDealerSimulator(cfg)
    N = simulator.n_dealers
    if ref is None:
        ref = reference_state(cfg)
    if h0 is None:
        h0 = np.full(N, float(cfg.policy.init_half_spread))
    h = np.asarray(h0, dtype=float).copy()
    if h.shape != (N,):
        raise ValueError(f"h0 must have shape ({N},), got {h.shape}")

    result = JointCobwebResult()
    result.h_history.append(h.copy())
    hi = float(cfg.policy.max_half_spread)
    for k in range(int(n_iters)):
        tau_hat = _measure_toxic_levels(cfg, simulator, h, seed=seed, n_episodes=n_episodes)
        h_next = np.array(
            [best_response(cfg, float(t), ref) for t in tau_hat], dtype=float
        )
        h_next = np.clip(h_next, 0.0, hi)
        result.h_history.append(h_next.copy())
        if np.max(np.abs(h_next - h)) < tol:
            result.converged = True
            h = h_next
            break
        h = h_next

    traj = np.stack(result.h_history, axis=0)
    result.common_mode = traj.mean(axis=1)
    result.spread_across_dealers = traj.max(axis=1) - traj.min(axis=1)
    return result


@dataclass
class JointModulusResult:
    """CRN-probed joint moduli next to their 1.3 closed-form predictions."""

    modulus_common: float  # in-phase probe (theory: N_eff * m_1)
    modulus_differential: float  # anti-phase probe (theory: (1-kappa) * m_1)
    h_ref: float
    delta: float
    n_dealers: int
    kappa: float


def measure_joint_modulus_sim(
    cfg: Config,
    h_ref: float,
    delta: Optional[float] = None,
    seed: int = 0,
    n_episodes: int = 4,
    simulator: Optional[MultiDealerSimulator] = None,
) -> JointModulusResult:
    """Probe the joint BR map's common and differential modes on the real env.

    ``delta`` defaults to ``0.25 * h_ref`` (scale-relative).  Both probe arms
    share all randomness (common random numbers) so the finite difference
    isolates the deterministic spread response.
    """
    if simulator is None:
        torch.manual_seed(seed)
        simulator = MultiDealerSimulator(cfg)
    N = simulator.n_dealers
    ref = reference_state(cfg)
    delta = 0.25 * h_ref if delta is None else float(delta)

    def br_vector(h_vec: np.ndarray) -> np.ndarray:
        tau_hat = _measure_toxic_levels(cfg, simulator, h_vec, seed=seed, n_episodes=n_episodes)
        return np.array([best_response(cfg, float(t), ref) for t in tau_hat])

    # In-phase (common-mode) probe: all dealers together.
    br_plus = br_vector(np.full(N, h_ref + delta))
    br_minus = br_vector(np.full(N, h_ref - delta))
    modulus_common = float(np.abs(br_plus.mean() - br_minus.mean()) / (2.0 * delta))

    # Anti-phase (differential) probe: needs at least two dealers.
    if N >= 2:
        up = np.full(N, h_ref)
        dn = np.full(N, h_ref)
        up[0], up[1] = h_ref + delta, h_ref - delta
        dn[0], dn[1] = h_ref - delta, h_ref + delta
        br_up = br_vector(up)
        br_dn = br_vector(dn)
        # Project onto the anti-symmetric direction (dealer0 - dealer1)/2.
        diff_up = 0.5 * (br_up[0] - br_up[1])
        diff_dn = 0.5 * (br_dn[0] - br_dn[1])
        modulus_diff = float(np.abs(diff_up - diff_dn) / (2.0 * delta))
    else:
        modulus_diff = float("nan")

    return JointModulusResult(
        modulus_common=modulus_common,
        modulus_differential=modulus_diff,
        h_ref=float(h_ref),
        delta=delta,
        n_dealers=N,
        kappa=simulator.kappa,
    )
