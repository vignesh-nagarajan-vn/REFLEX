"""The Repeated-Risk-Minimization (RRM) outer loop.

This is the central experiment.  Starting from an initial policy, each iteration:

1. **deploys** the current policy on the structural market ``T_true`` and collects
   data (:func:`collect`);
2. **refits** the operator ``T_theta`` from scratch on that data (plain repeated
   retraining -- one deployment's worth of data, so it cannot see performativity);
3. **evaluates** the true performative risk of the deployed policy on fresh
   ``T_true`` rollouts;
4. **re-optimises** the policy against the frozen operator to get the next iterate.

The sequence of iterates ``phi_k`` and the successive step sizes
``||phi_{k+1} - phi_k||`` are the raw material for the convergence analysis.  We
also track a scalar cobweb coordinate (the central half-spread) because it is the
dominant direction of the iteration and the cleanest thing to plot.

Whether the loop converges or diverges depends on the adversariality ``alpha``:
the per-iteration step is driven by how much toxicity the dealer's re-tightening
summons, which scales with ``alpha`` (and the explicit ``toxicity_feedback``
lever).  Locating the ``alpha`` at which the modulus crosses 1 is the headline
result.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch
from torch import Tensor

from ..config import Config
from ..env.simulator import StructuralSimulator
from ..objective.reward import reward as reward_fn
from ..policy import build_policy
from ..types import MarketState, policy_summary
from .data_collection import collect, collect_initial_states
from .fit_operator import fit_operator
from .optimize_policy import optimize_policy, rgd_step

# Divergence guards: a runaway high-alpha run drives performative risk and
# toxicity to extreme magnitudes. Beyond these (very loose) thresholds we declare
# the run divergent and stop, rather than letting NaNs reach the next operator
# fit. They are intentionally far above any healthy-regime value.
_BLOWUP_RISK = 1.0e6
_BLOWUP_TOX = 1.0e4


@dataclass
class RRMIterate:
    """Per-iteration record."""

    k: int
    flat_params: np.ndarray  # policy parameter vector after this iteration
    central_half_spread: float  # scalar cobweb coordinate (mean half-spread)
    step_size: float  # ||phi_{k} - phi_{k-1}|| (0 for k=0)
    performative_risk: float  # PR of the policy deployed at this iteration (on T_true)
    operator_val_nll: float
    operator_baseline_nll: float
    opt_objective_start: float
    opt_objective_final: float
    grad_path: str
    mean_informed_volume: float  # toxicity proxy (per step, summed over bonds)
    mean_adverse_loss: float
    hhi: float  # concentration of gross volume across bonds


@dataclass
class RRMTrajectory:
    """Full record of an RRM run."""

    alpha: float
    seed: int
    iterates: List[RRMIterate] = field(default_factory=list)
    converged: bool = False
    stop_reason: str = "max_iters"

    # --- convenience views ------------------------------------------------ #
    @property
    def step_sizes(self) -> np.ndarray:
        return np.array([it.step_size for it in self.iterates[1:]], dtype=float)

    @property
    def central_spreads(self) -> np.ndarray:
        return np.array([it.central_half_spread for it in self.iterates], dtype=float)

    @property
    def flat_iterates(self) -> np.ndarray:
        return np.stack([it.flat_params for it in self.iterates], axis=0)

    @property
    def performative_risks(self) -> np.ndarray:
        return np.array([it.performative_risk for it in self.iterates], dtype=float)


def _central_half_spread(policy, ref_state: MarketState) -> float:
    """Scalar summary of the policy's spread level at a reference state."""
    with torch.no_grad():
        return float(policy.quote(ref_state).half_spread.mean())


def _evaluate_true_risk(
    simulator: StructuralSimulator,
    policy,
    cfg: Config,
    base_seed: int,
    generator: torch.Generator,
):
    """Mean per-step performative risk + toxicity/HHI diagnostics on fresh T_true."""
    horizon = cfg.simulator.horizon
    obj_total = 0.0
    inf_total = 0.0
    adv_total = 0.0
    n_steps = 0
    hhi_acc = 0.0
    hhi_count = 0
    with torch.no_grad():
        for ep in range(int(cfg.rrm.eval_episodes)):
            state = simulator.reset(seed=base_seed + ep)
            for _ in range(horizon):
                q = policy.quote(state)
                tr = simulator.step(state, q, generator=generator)
                rb = reward_fn(tr, cfg.reward)
                obj_total += float(rb.objective)
                inf_total += float(tr.fills.informed_volume.sum())
                adv_total += float(tr.pnl_components["adverse_selection_loss"].sum())
                gv = tr.fills.gross_volume
                tot = float(gv.sum())
                if tot > 1e-8:
                    shares = (gv / gv.sum())
                    hhi_acc += float((shares ** 2).sum())
                    hhi_count += 1
                n_steps += 1
                state = tr.next_state
    pr = -(obj_total / max(n_steps, 1))
    return {
        "performative_risk": pr,
        "mean_informed_volume": inf_total / max(n_steps, 1),
        "mean_adverse_loss": adv_total / max(n_steps, 1),
        "hhi": hhi_acc / max(hhi_count, 1),
    }


def run_rrm(
    cfg: Config,
    simulator: Optional[StructuralSimulator] = None,
    seed: Optional[int] = None,
    logger=None,
    verbose: bool = False,
) -> RRMTrajectory:
    """Run the RRM loop and return the full iterate trajectory.

    Parameters
    ----------
    cfg:
        Full configuration (``alpha`` lives in ``cfg.clients.alpha``).
    simulator:
        Optional pre-built simulator; if ``None`` one is constructed from ``cfg``.
    seed:
        Master seed for this run (defaults to ``cfg.seed``).  Controls the policy
        init, data collection, fitting and evaluation RNGs.
    logger:
        Optional logger passed through to the policy optimiser.
    verbose:
        If ``True`` print a one-line summary per iteration.

    Returns
    -------
    RRMTrajectory
    """
    seed = int(cfg.seed if seed is None else seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    if simulator is None:
        simulator = StructuralSimulator(cfg)

    policy = build_policy(cfg.policy)
    traj = RRMTrajectory(alpha=float(cfg.clients.alpha), seed=seed)

    # A fixed reference state for the scalar coordinate (zero inventory, neutral).
    ref_state = simulator.reset(seed=seed + 9991).detach()

    prev_flat = policy.flatten().clone()
    rng_master = torch.Generator().manual_seed(seed)

    for k in range(int(cfg.rrm.max_iters)):
        # Distinct, reproducible sub-seeds per iteration.
        collect_seed = seed + 1000 * (k + 1)
        eval_seed = seed + 7000 + 113 * k
        gen_collect = torch.Generator().manual_seed(collect_seed)
        gen_fit = torch.Generator().manual_seed(seed + 3000 + 17 * k)
        gen_opt = torch.Generator().manual_seed(seed + 5000 + 29 * k)
        gen_eval = torch.Generator().manual_seed(eval_seed)

        # 1. deploy phi_k and collect data on T_true.
        transitions = collect(simulator, policy, cfg, base_seed=collect_seed, generator=gen_collect)

        # 2. refit operator from scratch on this deployment's data.
        from ..operator.response_operator import MarketResponseOperator

        operator = MarketResponseOperator(cfg)
        fit_res = fit_operator(operator, transitions, policy, cfg.operator, generator=gen_fit)

        # 3. evaluate the true performative risk of the deployed policy.
        diag = _evaluate_true_risk(simulator, policy, cfg, eval_seed, gen_eval)
        central = _central_half_spread(policy, ref_state)

        # Blowup guard: a divergent high-alpha run drives toxicity / risk to
        # extreme values. Record the iterate, mark the run divergent, and stop
        # cleanly rather than letting NaNs propagate into the next fit.
        blowup = (
            not np.isfinite(diag["performative_risk"])
            or not np.isfinite(central)
            or abs(diag["performative_risk"]) > _BLOWUP_RISK
            or diag["mean_informed_volume"] > _BLOWUP_TOX
        )

        # 4. re-optimise the policy against the frozen operator -> phi_{k+1}.
        if blowup:
            # Record a final iterate marking the divergence and stop.
            new_flat = policy.flatten().clone()
            step = float(torch.norm(new_flat - prev_flat))
            traj.iterates.append(
                RRMIterate(
                    k=k,
                    flat_params=new_flat.numpy(),
                    central_half_spread=central if np.isfinite(central) else float("inf"),
                    step_size=(0.0 if k == 0 else step),
                    performative_risk=diag["performative_risk"],
                    operator_val_nll=fit_res.best_val_nll,
                    operator_baseline_nll=fit_res.baseline_val_nll,
                    opt_objective_start=float("nan"),
                    opt_objective_final=float("nan"),
                    grad_path="n/a",
                    mean_informed_volume=diag["mean_informed_volume"],
                    mean_adverse_loss=diag["mean_adverse_loss"],
                    hhi=diag["hhi"],
                )
            )
            traj.converged = False
            traj.stop_reason = "blowup"
            if verbose:
                print(
                    f"[a={cfg.clients.alpha:.2f} s={seed}] iter {k:2d} | BLOWUP "
                    f"(PR={diag['performative_risk']:.3e}, tox={diag['mean_informed_volume']:.3e}) -> divergent"
                )
            break

        if not cfg.rrm.warm_start_policy:
            policy = build_policy(cfg.policy)
        frozen = policy_summary(ref_state, policy).detach()
        init_states = collect_initial_states(simulator, cfg.policy.n_rollouts, base_seed=collect_seed + 777)

        if cfg.rrm.update_rule == "rrm":
            opt_res = optimize_policy(
                policy, operator, init_states, cfg,
                frozen_summary=frozen, generator=gen_opt, logger=logger,
            )
            opt_start = opt_res.objective_history[0] if opt_res.objective_history else float("nan")
            opt_final = opt_res.final_objective
            grad_path = opt_res.grad_path
        else:  # repeated gradient descent (default)
            opt_final = rgd_step(
                policy, operator, init_states, cfg,
                frozen_summary=frozen, n_steps=cfg.rrm.rgd_steps, lr=cfg.rrm.rgd_lr,
                generator=gen_opt,
            )
            opt_start = float("nan")
            grad_path = "rgd"

        new_flat = policy.flatten().clone()
        step = float(torch.norm(new_flat - prev_flat))

        it = RRMIterate(
            k=k,
            flat_params=new_flat.numpy(),
            central_half_spread=central,
            step_size=(0.0 if k == 0 else step),
            performative_risk=diag["performative_risk"],
            operator_val_nll=fit_res.best_val_nll,
            operator_baseline_nll=fit_res.baseline_val_nll,
            opt_objective_start=opt_start,
            opt_objective_final=opt_final,
            grad_path=grad_path,
            mean_informed_volume=diag["mean_informed_volume"],
            mean_adverse_loss=diag["mean_adverse_loss"],
            hhi=diag["hhi"],
        )
        traj.iterates.append(it)

        if verbose:
            print(
                f"[a={cfg.clients.alpha:.2f} s={seed}] iter {k:2d} | "
                f"h={central:.3f} step={it.step_size:.4f} "
                f"PR={diag['performative_risk']:.3f} "
                f"tox={diag['mean_informed_volume']:.3f} "
                f"NLL {fit_res.baseline_val_nll:.2f}->{fit_res.best_val_nll:.2f} "
                f"[{grad_path}]"
            )

        # Convergence check (skip the cold-start first step).
        if k >= 1 and step < cfg.rrm.tol:
            traj.converged = True
            traj.stop_reason = "tol"
            break

        prev_flat = new_flat

    return traj
