"""The v3 outer retraining loops: blind RRM vs. PerfGD-corrected (analytic / learned).

One driver, :func:`run_loop`, subsumes the v2 RRM experiment and adds the two
un-blinded variants derived in theory 1.2:

* ``mode="rrm"`` -- blind repeated retraining.  The operator is fit and the
  policy re-optimised under the frozen-summary convention; the loop's stability
  is governed by the cobweb modulus ``m = epsilon*beta/gamma`` and it diverges
  past ``epsilon* = gamma/beta``.
* ``mode="perfgd_analytic"`` -- same frozen-summary optimisation *plus* the
  closed-form correction ``Delta(h) = -beta*(h - psi)*epsilon(h)`` from
  :mod:`reflex.theory.perfgd`, injected as a surrogate gradient term.  Stability
  is governed by the objective curvature ``gamma_PO``, so the loop converges
  for ``epsilon`` beyond ``epsilon*`` (theory 1.2 §5).
* ``mode="perfgd_learned"`` -- the fully-ML counterpart: the operator is fit on
  a **window** of recent deployments (``operator.context_window``), so its
  summary-dependence identifies the distribution response, and the policy is
  optimised with a **live** summary so that learned response enters the
  gradient.  No closed form is consumed; the market model is learned end to
  end.  Requires ``context_window >= 2`` to be meaningful.

Every iteration also records the ML<->math seam diagnostics: the operator's
*learned* toxic slope ``d E_hat[adverse]/dh`` (via
:meth:`~reflex.operator.response_operator.MarketResponseOperator.distribution_response`)
next to the theory's closed form ``-psi*epsilon(h)``, so the two can be plotted
against each other along the run.

The per-iterate record type is shared with the v2-compatible
:mod:`reflex.equilibrium.rrm_loop` (kept as the frozen baseline entry point),
so all downstream analysis (convergence classification, sweeps, plots) works on
either loop's output unchanged.
"""

from __future__ import annotations

import copy
import math
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch

from ..config import Config
from ..env.simulator import StructuralSimulator
from ..operator.response_operator import MarketResponseOperator
from ..policy import build_policy
from ..theory.analytic_boundary import epsilon as epsilon_of
from ..theory.analytic_boundary import reference_state
from ..theory.perfgd import perfgd_correction
from ..types import policy_summary
from .data_collection import collect, collect_initial_states
from .fit_operator import DeploymentRecord, fit_operator_windowed
from .optimize_policy import optimize_policy, rgd_step
from .rrm_loop import (
    _BLOWUP_RISK,
    _BLOWUP_TOX,
    RRMIterate,
    RRMTrajectory,
    _central_half_spread,
    _evaluate_true_risk,
)

LOOP_MODES = ("rrm", "perfgd_analytic", "perfgd_learned")


@dataclass
class LoopDiagnostics:
    """Per-iteration ML<->math seam diagnostics."""

    k: int
    learned_toxic_slope: float  # operator's d E_hat[adverse]/dh (nan if unavailable)
    analytic_toxic_slope: float  # theory: -psi * epsilon(h_central)
    correction: float  # analytic Delta applied this iteration (0 unless perfgd_analytic)
    window_len: int  # deployments in the operator's fitting window


@dataclass
class LoopResult:
    """Full record of a :func:`run_loop` run."""

    mode: str
    trajectory: RRMTrajectory
    diagnostics: List[LoopDiagnostics] = field(default_factory=list)

    # --- convenience views ------------------------------------------------ #
    @property
    def learned_slopes(self) -> np.ndarray:
        return np.array([d.learned_toxic_slope for d in self.diagnostics], dtype=float)

    @property
    def analytic_slopes(self) -> np.ndarray:
        return np.array([d.analytic_toxic_slope for d in self.diagnostics], dtype=float)


def run_loop(
    cfg: Config,
    mode: Optional[str] = None,
    simulator: Optional[StructuralSimulator] = None,
    seed: Optional[int] = None,
    logger=None,
    verbose: bool = False,
) -> LoopResult:
    """Run the outer retraining loop in the requested mode.

    Parameters mirror :func:`reflex.equilibrium.rrm_loop.run_rrm`; ``mode``
    defaults to ``cfg.rrm.loop_mode``.  Seeding is identical across modes, so a
    three-way comparison from a common start isolates the effect of the
    correction.
    """
    mode = str(cfg.rrm.loop_mode if mode is None else mode)
    if mode not in LOOP_MODES:
        raise ValueError(f"unknown loop mode {mode!r} (expected one of {LOOP_MODES})")
    # The fitting window applies in every mode: under "rrm" the frozen-summary
    # optimisation keeps the loop blind even when the windowed operator has
    # learned the slope (the diagnostic then shows exactly that gap), while
    # "perfgd_learned" needs the window for its correction to be identified.
    window_size = int(cfg.operator.context_window)
    if mode == "perfgd_learned" and window_size < 2:
        print(
            "[loops] WARNING: perfgd_learned with context_window < 2 -- the "
            "operator cannot identify dD/dphi from a constant summary; the "
            "learned correction will be noise."
        )

    seed = int(cfg.seed if seed is None else seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    if simulator is None:
        simulator = StructuralSimulator(cfg)

    policy = build_policy(cfg.policy)
    traj = RRMTrajectory(alpha=float(cfg.clients.alpha), seed=seed)
    result = LoopResult(mode=mode, trajectory=traj)

    ref_state = simulator.reset(seed=seed + 9991).detach()

    # Closed-form pieces (used by the analytic mode and by the diagnostics).
    ref_theory = reference_state(cfg)
    correction_fn = None
    if mode == "perfgd_analytic":
        correction_fn = lambda h: perfgd_correction(cfg, h, ref_theory)  # noqa: E731
    # Units: theory's Delta is per step per bond; the rollout objective sums
    # over horizon steps and n_bonds bonds (per rollout).
    correction_scale = float(cfg.policy.rollout_horizon * cfg.bonds.n_bonds)
    live = mode == "perfgd_learned"

    window: "deque[DeploymentRecord]" = deque(maxlen=max(window_size, 1))
    prev_flat = policy.flatten().clone()

    for k in range(int(cfg.rrm.max_iters)):
        collect_seed = seed + 1000 * (k + 1)
        eval_seed = seed + 7000 + 113 * k
        gen_collect = torch.Generator().manual_seed(collect_seed)
        gen_fit = torch.Generator().manual_seed(seed + 3000 + 17 * k)
        gen_opt = torch.Generator().manual_seed(seed + 5000 + 29 * k)
        gen_eval = torch.Generator().manual_seed(eval_seed)

        # 1. deploy phi_k, collect data, append the deployment to the window.
        transitions = collect(simulator, policy, cfg, base_seed=collect_seed, generator=gen_collect)
        window.append(DeploymentRecord(transitions=transitions, policy=copy.deepcopy(policy)))

        # 2. refit the operator from scratch on the window.
        operator = MarketResponseOperator(cfg)
        fit_res = fit_operator_windowed(operator, list(window), cfg.operator, generator=gen_fit)

        # 3. evaluate the true performative risk of the deployed policy.
        diag = _evaluate_true_risk(simulator, policy, cfg, eval_seed, gen_eval)
        central = _central_half_spread(policy, ref_state)

        # ML<->math seam diagnostics at the current operating point.
        frozen = policy_summary(ref_state, policy).detach()
        try:
            with torch.enable_grad():
                learned_slope = operator.toxic_slope(
                    ref_state, policy.quote(ref_state), frozen
                )
        except Exception:  # pragma: no cover - diagnostic must never kill a run
            learned_slope = float("nan")
        analytic_slope = -ref_theory.psi * epsilon_of(cfg, central, ref_theory)
        applied_correction = (
            float(correction_fn(central)) if correction_fn is not None else 0.0
        )
        result.diagnostics.append(
            LoopDiagnostics(
                k=k,
                learned_toxic_slope=learned_slope,
                analytic_toxic_slope=analytic_slope,
                correction=applied_correction,
                window_len=len(window),
            )
        )

        # Blowup guard (identical to the v2 baseline loop).
        blowup = (
            not np.isfinite(diag["performative_risk"])
            or not np.isfinite(central)
            or abs(diag["performative_risk"]) > _BLOWUP_RISK
            or diag["mean_informed_volume"] > _BLOWUP_TOX
        )
        if blowup:
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
                print(f"[{mode} s={seed}] iter {k:2d} | BLOWUP -> divergent")
            break

        # 4. re-optimise the policy -> phi_{k+1}.
        if not cfg.rrm.warm_start_policy:
            policy = build_policy(cfg.policy)
        init_states = collect_initial_states(
            simulator, cfg.policy.n_rollouts, base_seed=collect_seed + 777
        )

        if cfg.rrm.update_rule == "rrm":
            opt_res = optimize_policy(
                policy, operator, init_states, cfg,
                frozen_summary=frozen, generator=gen_opt, logger=logger,
                analytic_correction=correction_fn,
                correction_scale=correction_scale,
                live_summary=live,
            )
            opt_start = opt_res.objective_history[0] if opt_res.objective_history else float("nan")
            opt_final = opt_res.final_objective
            grad_path = opt_res.grad_path
        else:
            opt_final = rgd_step(
                policy, operator, init_states, cfg,
                frozen_summary=frozen, n_steps=cfg.rrm.rgd_steps, lr=cfg.rrm.rgd_lr,
                generator=gen_opt,
                analytic_correction=correction_fn,
                correction_scale=correction_scale,
                live_summary=live,
            )
            opt_start = float("nan")
            grad_path = "rgd" + ("+live" if live else "") + (
                "+delta" if correction_fn is not None else ""
            )

        new_flat = policy.flatten().clone()
        step = float(torch.norm(new_flat - prev_flat))

        traj.iterates.append(
            RRMIterate(
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
        )

        if verbose:
            ls = result.diagnostics[-1].learned_toxic_slope
            print(
                f"[{mode} s={seed}] iter {k:2d} | h={central:.4f} step={step:.4f} "
                f"PR={diag['performative_risk']:.3f} "
                f"slope(learned={ls:.3f}, analytic={analytic_slope:.3f}) "
                f"win={len(window)} [{grad_path}]"
            )

        if k >= 1 and step < cfg.rrm.tol:
            traj.converged = True
            traj.stop_reason = "tol"
            break

        prev_flat = new_flat

    return result
