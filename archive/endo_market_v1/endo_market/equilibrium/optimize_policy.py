"""Re-optimise the dealer policy against the frozen operator ``T_theta``.

This is the inner loop of RRM: holding the learned dynamics fixed, find the
policy that maximises expected dealer objective under ``T_theta``.  Two crucial
conventions:

* **Frozen regime.** The policy summary fed to the operator is frozen at the
  *deployed* policy's value (Repeated-Risk-Minimization).  The candidate policy's
  quotes still move the operator's mechanical fill/spread channels, but the
  toxicity regime is anchored, so the dealer does not anticipate that
  re-tightening will summon more toxic flow.
* **Pathwise by default, with an explicit REINFORCE fallback.**  Reparameterised
  (pathwise) gradients through the operator's rollout are the default and are far
  lower variance.  A score-function (REINFORCE) estimator is available and is
  used if forced by config or if pathwise gradients become non-finite -- and the
  chosen path is logged, never switched silently.

The Monte-Carlo rollouts are vectorised by stacking the ``R`` rollout init states
along the bond axis: because ``T_theta`` treats bonds independently, ``R``
rollouts of ``N`` bonds are computed as a single rollout of ``R*N`` bonds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import torch
from torch import Tensor

from ..config import Config
from ..types import MarketState, policy_summary


@dataclass
class OptimizeResult:
    """Diagnostics from a single policy optimisation."""

    objective_history: List[float] = field(default_factory=list)
    grad_path: str = "pathwise"  # "pathwise" or "reinforce"
    frozen_summary: Optional[Tensor] = None
    final_objective: float = float("nan")
    switched_to_reinforce: bool = False


def _stack_states(states: Sequence[MarketState]) -> MarketState:
    """Concatenate several states along the bond axis into one combined state."""
    return MarketState(
        inventory=torch.cat([s.inventory for s in states]),
        mid=torch.cat([s.mid for s in states]),
        fundamental=torch.cat([s.fundamental for s in states]),
        liquidity=torch.cat([s.liquidity for s in states]),
        flow_recent=torch.cat([s.flow_recent for s in states]),
        vol_recent=torch.cat([s.vol_recent for s in states]),
        t=0,
    )


def optimize_policy(
    policy,
    operator,
    init_states: Sequence[MarketState],
    cfg: Config,
    frozen_summary: Optional[Tensor] = None,
    generator: Optional[torch.Generator] = None,
    logger=None,
) -> OptimizeResult:
    """Optimise ``policy`` against the fixed ``operator``.

    Parameters
    ----------
    policy:
        The dealer policy to optimise (updated in place).
    operator:
        The fixed learned operator ``T_theta`` (its weights are not modified).
    init_states:
        Pool of initial states; ``cfg.policy.n_rollouts`` are stacked per step.
    cfg:
        Full config (uses the ``policy`` and ``reward`` sections).
    frozen_summary:
        The frozen policy summary (regime).  If ``None`` it is computed once from
        the current policy at ``init_states[0]`` and detached.
    generator:
        RNG for reproducible rollouts / minibatch selection.
    logger:
        Optional object with an ``info``/``log`` method; falls back to ``print``.

    Returns
    -------
    OptimizeResult
    """
    pcfg = cfg.policy
    log = _make_log(logger)

    if frozen_summary is None:
        with torch.no_grad():
            frozen_summary = policy_summary(init_states[0], policy).detach()

    # Freeze operator weights so only the policy is updated (gradients still flow
    # through the operator's forward to reach the policy).
    op_req = [p.requires_grad for p in operator.parameters()]
    for p in operator.parameters():
        p.requires_grad_(False)
    operator.eval()

    opt = torch.optim.Adam(policy.parameters(), lr=pcfg.lr)
    result = OptimizeResult(grad_path="reinforce" if pcfg.reinforce else "pathwise")
    result.frozen_summary = frozen_summary

    n_roll = int(pcfg.n_rollouts)
    pool = list(init_states)
    baseline = 0.0  # running objective baseline for REINFORCE variance reduction
    use_reinforce = bool(pcfg.reinforce)
    log(f"policy optimisation: starting on '{result.grad_path}' path "
        f"({pcfg.inner_steps} steps, {n_roll} rollouts x H={pcfg.rollout_horizon})")

    try:
        for step in range(int(pcfg.inner_steps)):
            # Select a batch of init states (cycling through the pool).
            idx = [
                int(torch.randint(len(pool), (1,), generator=generator).item())
                for _ in range(n_roll)
            ]
            batch = _stack_states([pool[i] for i in idx])

            roll = operator.rollout(
                batch,
                policy,
                horizon=pcfg.rollout_horizon,
                reward_cfg=cfg.reward,
                policy_summary_override=frozen_summary,
                generator=generator,
                score_function=use_reinforce,
            )
            # Objective is summed over the R*N stacked bonds; normalise per rollout.
            mean_obj = roll.objective / n_roll

            if use_reinforce:
                adv = (mean_obj.detach() - baseline)
                loss = -(adv * roll.logprob_sum / n_roll)
                baseline = 0.9 * baseline + 0.1 * float(mean_obj.detach())
            else:
                loss = -mean_obj

            opt.zero_grad()
            loss.backward()

            # Guard against non-finite pathwise gradients -- switch loudly, never silently.
            if not use_reinforce and not _grads_finite(policy):
                log("WARNING: pathwise gradients are non-finite; switching to REINFORCE")
                use_reinforce = True
                result.grad_path = "reinforce"
                result.switched_to_reinforce = True
                opt.zero_grad()
                continue

            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=10.0)
            opt.step()
            result.objective_history.append(float(mean_obj.detach()))
    finally:
        # Restore operator parameter grad flags.
        for p, r in zip(operator.parameters(), op_req):
            p.requires_grad_(r)

    if result.objective_history:
        result.final_objective = result.objective_history[-1]
    log(f"policy optimisation done on '{result.grad_path}' path: "
        f"objective {result.objective_history[0]:.3f} -> {result.final_objective:.3f}"
        if result.objective_history else "policy optimisation produced no steps")
    return result


def rgd_step(
    policy,
    operator,
    init_states: Sequence[MarketState],
    cfg: Config,
    frozen_summary: Tensor,
    n_steps: int,
    lr: float,
    generator: Optional[torch.Generator] = None,
) -> float:
    """Take ``n_steps`` plain-gradient steps on the risk under the fixed operator.

    This is the repeated-gradient-descent (RGD) update used by the RRM loop.
    Unlike :func:`optimize_policy` (which fully re-optimizes and can overshoot by
    extrapolating the operator far from the deployed action), RGD takes a small,
    controlled step from the deployed policy, so the outer-loop map is well behaved
    and its contraction modulus is governed by the performative feedback (and hence
    by ``alpha``).

    The operator weights are frozen (gradients still flow through to the policy via
    the pathwise rollout).  Returns the mean per-rollout objective at the final
    step (for logging).
    """
    pcfg = cfg.policy
    n_roll = int(pcfg.n_rollouts)
    pool = list(init_states)

    op_req = [p.requires_grad for p in operator.parameters()]
    for p in operator.parameters():
        p.requires_grad_(False)
    operator.eval()

    opt = torch.optim.SGD(policy.parameters(), lr=lr)
    last_obj = float("nan")
    try:
        for _ in range(int(n_steps)):
            idx = [
                int(torch.randint(len(pool), (1,), generator=generator).item())
                for _ in range(n_roll)
            ]
            batch = _stack_states([pool[i] for i in idx])
            roll = operator.rollout(
                batch, policy,
                horizon=pcfg.rollout_horizon, reward_cfg=cfg.reward,
                policy_summary_override=frozen_summary, generator=generator,
                score_function=False,
            )
            mean_obj = roll.objective / n_roll
            loss = -mean_obj
            opt.zero_grad()
            loss.backward()
            if not _grads_finite(policy):
                opt.zero_grad()
                continue
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=10.0)
            opt.step()
            last_obj = float(mean_obj.detach())
    finally:
        for p, r in zip(operator.parameters(), op_req):
            p.requires_grad_(r)
    return last_obj


def _grads_finite(module) -> bool:
    for p in module.parameters():
        if p.grad is not None and not torch.isfinite(p.grad).all():
            return False
    return True


def _make_log(logger):
    if logger is None:
        return lambda msg: None  # quiet by default; RRM loop logs the summary
    if hasattr(logger, "info"):
        return logger.info
    if hasattr(logger, "log"):
        return logger.log
    return print
