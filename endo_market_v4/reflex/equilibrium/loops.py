"""The v4 outer retraining loops: blind RRM vs. PerfGD-corrected (analytic /
learned / structural).

One driver, :func:`run_loop`, subsumes the v2 RRM experiment and adds the
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
* ``mode="perfgd_learned"`` -- the free-form ML counterpart: the operator is
  fit on a **window** of recent deployments (``operator.context_window``), so
  its summary-dependence identifies the distribution response, and the policy
  is optimised with a **live** summary so that learned response enters the
  gradient.  No closed form is consumed.  Requires ``context_window >= 2`` to
  be meaningful.  Kept as the documented v3 negative result: the free-form
  operator's implied ``dJ/dh`` diverges from the structural one away from the
  deployed regime, so this mode does not reproduce the stabilisation.
* ``mode="perfgd_structural"`` -- the v4 mode that closes that gap.  Every 1.2
  ingredient is **fitted from the window's own transitions** by
  :func:`reflex.equilibrium.structural_response.fit_structural_response`
  (the GLFT-anchored families ``tau_hat = C0 + C1*exp(-c*h)`` and
  ``u_hat = A_u*exp(-k_u*h)``, plus the realized severity ``psi_hat``), and
  the policy's central spread takes one ascent step per deployment on the
  estimated corrected gradient ``Phi_hat'(h) = G_hat(h) + Delta_hat(h)``
  (step ``1/gamma_PO_hat`` unless ``rrm.structural_eta`` overrides;
  ``rrm.structural_eta_decay`` shrinks it per iteration).  No closed-form
  market constant is consumed.  When the window's realised spread range falls
  below ``rrm.structural_min_rel_range`` the previous identifiable fit is
  held (the anti-echo-chamber freeze).

Every iteration also records the ML<->math seam diagnostics: the operator's
*learned* toxic slope ``d E_hat[adverse]/dh`` (via
:meth:`~reflex.operator.response_operator.MarketResponseOperator.distribution_response`)
next to the theory's closed form ``-psi*epsilon(h)`` -- and, in the structural
mode, the *fitted* slope ``-psi_hat*eps_hat(h)`` -- so every reading of the
distribution response can be plotted against the others along the run.

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
from .structural_response import (
    StructuralResponse,
    fit_structural_response,
    retune_central_spread,
)

LOOP_MODES = ("rrm", "perfgd_analytic", "perfgd_learned", "perfgd_structural")


@dataclass
class LoopDiagnostics:
    """Per-iteration ML<->math seam diagnostics."""

    k: int
    learned_toxic_slope: float  # operator's d E_hat[adverse]/dh (nan if unavailable)
    analytic_toxic_slope: float  # theory: -psi * epsilon(h_central)
    correction: float  # correction applied this iteration (analytic or fitted; 0 in blind modes)
    window_len: int  # deployments in the operator's fitting window
    structural_toxic_slope: float = float("nan")  # fitted -psi_hat*eps_hat(h) (structural mode)
    h_po_hat: float = float("nan")  # estimated performative optimum (structural mode)
    structural_frozen: bool = False  # True when the anti-echo freeze held the last fit


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

    @property
    def structural_slopes(self) -> np.ndarray:
        return np.array([d.structural_toxic_slope for d in self.diagnostics], dtype=float)


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
    if mode == "perfgd_structural" and float(cfg.rrm.collection_jitter) <= 0.0:
        print(
            "[loops] WARNING: perfgd_structural with collection_jitter = 0 -- "
            "once the loop settles there is no within-deployment spread "
            "variation left to identify the structural response; the "
            "anti-echo freeze will hold a stale fit."
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

    # perfgd_structural state: the current fitted response plus its own long
    # deployment memory (the operator's short context window is not enough to
    # identify the exponential families globally; the loop's traversed
    # history is).
    structural: Optional[StructuralResponse] = None
    structural_frozen = False
    struct_history: "deque[DeploymentRecord]" = deque(
        maxlen=max(int(cfg.rrm.structural_window), 1)
    )

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

        # 2b. (structural mode) refit the GLFT-anchored response on the loop's
        # own long history; hold the previous fit when the history is
        # unidentified (anti-echo freeze).
        if mode == "perfgd_structural":
            struct_history.append(window[-1])
            fit_new = fit_structural_response(list(struct_history), cfg)
            if fit_new.identified or structural is None:
                structural = fit_new
                structural_frozen = not fit_new.identified
            else:
                structural_frozen = True

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
        if mode == "perfgd_structural" and structural is not None:
            structural_slope = structural.toxic_slope_hat(central)
            h_po_hat = structural.solve_h_po_hat(cfg.reward, cfg.policy.max_half_spread)
            applied_correction = structural.correction_hat(central, cfg.reward)
        else:
            structural_slope = float("nan")
            h_po_hat = float("nan")
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
                structural_toxic_slope=structural_slope,
                h_po_hat=h_po_hat,
                structural_frozen=structural_frozen,
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

        if mode == "perfgd_structural":
            # One 1-D ascent step on the *fitted* corrected gradient (1.2 §3
            # with every ingredient estimated from the window's data).  The
            # central spread is the dominant coordinate of the iterate map
            # (1.1 §1); the update is realised in the policy by retuning its
            # half-spread bias.  Step: 1/gamma_PO_hat at the estimated optimum
            # (the near-optimal step of 1.2 §4.2) unless overridden, decayed
            # by structural_eta_decay^k for optional Robbins-Monro damping.
            g_hat = structural.corrected_gradient_hat(central, cfg.reward)
            if float(cfg.rrm.structural_eta) > 0.0:
                eta0 = float(cfg.rrm.structural_eta)
            else:
                gpo = structural.gamma_po_hat(h_po_hat, cfg.reward)
                eta0 = 1.0 / gpo if gpo > 1e-9 else 1e-2
            eta_k = eta0 * (float(cfg.rrm.structural_eta_decay) ** k)
            h_next = central + eta_k * g_hat
            # Trust region: never move beyond the fraction of the current
            # spread the fits can be trusted to extrapolate over.
            cap = float(cfg.rrm.structural_max_rel_step) * max(central, 1e-3)
            h_next = min(max(h_next, central - cap), central + cap)
            h_next = min(max(h_next, 1e-3), float(cfg.policy.max_half_spread))
            retune_central_spread(policy, ref_state, h_next)
            opt_start = float("nan")
            opt_final = float("nan")
            grad_path = "structural" + ("(frozen-fit)" if structural_frozen else "")
        elif cfg.rrm.update_rule == "rrm":
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
            d_last = result.diagnostics[-1]
            extra = (
                f" h_PO_hat={d_last.h_po_hat:.3f} struct={d_last.structural_toxic_slope:.3f}"
                if mode == "perfgd_structural" else ""
            )
            print(
                f"[{mode} s={seed}] iter {k:2d} | h={central:.4f} step={step:.4f} "
                f"PR={diag['performative_risk']:.3f} "
                f"slope(learned={d_last.learned_toxic_slope:.3f}, "
                f"analytic={analytic_slope:.3f}){extra} "
                f"win={len(window)} [{grad_path}]"
            )

        if k >= 1 and step < cfg.rrm.tol:
            traj.converged = True
            traj.stop_reason = "tol"
            break

        prev_flat = new_flat

    return result
