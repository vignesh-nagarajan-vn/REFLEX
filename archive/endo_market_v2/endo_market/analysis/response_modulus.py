"""Estimate the RRM contraction modulus by probing the best-response map.

The cleanest, lowest-variance estimator of whether the policy<->distribution loop
contracts is the **local slope of the best-response map** at a reference spread.
The Repeated-Risk-Minimization iteration is

    h_{k+1} = BR(h_k)  :=  argmax_h  E_{T_theta fit to D(h_k)}[ objective ],

(with the policy summary frozen at the deployed regime).  Linearising at a fixed
point ``h*`` gives ``h_{k+1} - h* ~ BR'(h*) (h_k - h*)``, so the iteration
contracts iff the **modulus** ``m = |BR'(h*)| < 1``.  We estimate ``BR'`` by a
symmetric finite difference: deploy a policy quoting ``h_ref + delta`` and one
quoting ``h_ref - delta``, run the full deploy -> collect -> refit-operator ->
re-optimize pipeline for each, and take

    m = | BR(h_ref + delta) - BR(h_ref - delta) | / (2 * delta).

Crucially we drive **both** probes with the *same* random-number generators
(common random numbers): the operator initialisation, the data-collection noise,
and the policy-optimization noise are shared, so the difference isolates the
deterministic dependence of the best response on the deployed regime and the
operator-fit sampling noise largely cancels.  Estimating ``m`` directly this way
avoids the well-known problem that a *contracting* noisy trajectory shrinks to its
sampling-noise floor (where successive-step ratios are ~1), which makes
contraction invisible to trajectory-based Lipschitz estimates.

This modulus is the object whose dependence on the performative-feedback gain
``epsilon`` (the ``toxicity_feedback`` knob) we study: as ``epsilon`` grows the
modulus crosses 1, reproducing the performative-prediction stability boundary
``epsilon < gamma / beta``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from ..config import Config
from ..env.simulator import StructuralSimulator
from ..operator.response_operator import MarketResponseOperator
from ..policy import build_policy
from ..policy.dealer_policy import LinearPolicy, _inverse_softplus
from ..types import policy_summary
from ..equilibrium.data_collection import collect, collect_initial_states
from ..equilibrium.fit_operator import fit_operator
from ..equilibrium.optimize_policy import optimize_policy


@dataclass
class ResponseModulusResult:
    """Result of a best-response-slope modulus probe."""

    modulus: float  # |BR'(h_ref)| -- contracts iff < 1
    br_plus: float  # central half-spread of BR to the (h_ref + delta) deployment
    br_minus: float  # central half-spread of BR to the (h_ref - delta) deployment
    h_ref: float
    delta: float
    stable: bool  # modulus < 1


def _deployed_policy_at(cfg: Config, h0: float) -> LinearPolicy:
    """A LinearPolicy that quotes a constant half-spread ``h0`` (zero skew)."""
    pol = LinearPolicy(cfg.policy)
    with torch.no_grad():
        pol.w_h.zero_()
        pol.b_h.fill_(_inverse_softplus(h0))
        pol.w_skew.zero_()
        pol.b_skew.zero_()
    return pol


def _best_response_central_spread(
    cfg: Config,
    simulator: StructuralSimulator,
    h_dep: float,
    seed: int,
    ref_state,
) -> float:
    """Deploy a fixed-``h_dep`` policy, refit the operator, fully re-optimize, and
    return the central (mean) half-spread of the resulting best-response policy.

    Uses generators seeded deterministically from ``seed`` so that two calls with
    the same ``seed`` but different ``h_dep`` share their randomness (common random
    numbers)."""
    gen_collect = torch.Generator().manual_seed(seed)
    gen_fit = torch.Generator().manual_seed(seed + 1)
    gen_opt = torch.Generator().manual_seed(seed + 2)

    deployed = _deployed_policy_at(cfg, h_dep)
    transitions = collect(simulator, deployed, cfg, base_seed=seed, generator=gen_collect)

    torch.manual_seed(seed)  # common operator initialisation across probes
    operator = MarketResponseOperator(cfg)
    fit_operator(operator, transitions, deployed, cfg.operator, generator=gen_fit)

    candidate = build_policy(cfg.policy)
    frozen = policy_summary(ref_state, deployed).detach()
    init_states = collect_initial_states(simulator, cfg.policy.n_rollouts, base_seed=seed + 999)
    optimize_policy(
        candidate, operator, init_states, cfg,
        frozen_summary=frozen, generator=gen_opt,
    )
    with torch.no_grad():
        return float(candidate.quote(ref_state).half_spread.mean())


def measure_response_modulus(
    cfg: Config,
    seed: int = 0,
    h_ref: float = 1.0,
    delta: float = 0.25,
    simulator: Optional[StructuralSimulator] = None,
) -> ResponseModulusResult:
    """Estimate the RRM contraction modulus at ``h_ref`` via the BR-slope probe.

    Parameters
    ----------
    cfg:
        Full configuration (the performative-feedback gain lives in
        ``cfg.clients.toxicity_feedback``).
    seed:
        Master seed; the two probe deployments share generators derived from it.
    h_ref:
        Reference half-spread at which the best-response slope is measured.
    delta:
        Half-width of the symmetric finite difference.
    simulator:
        Optional pre-built simulator; constructed from ``cfg`` if ``None``.

    Returns
    -------
    ResponseModulusResult
    """
    if simulator is None:
        torch.manual_seed(seed)
        simulator = StructuralSimulator(cfg)
    ref_state = simulator.reset(seed=seed + 9991).detach()

    br_plus = _best_response_central_spread(cfg, simulator, h_ref + delta, seed, ref_state)
    br_minus = _best_response_central_spread(cfg, simulator, h_ref - delta, seed, ref_state)
    modulus = abs(br_plus - br_minus) / (2.0 * delta)
    return ResponseModulusResult(
        modulus=modulus,
        br_plus=br_plus,
        br_minus=br_minus,
        h_ref=h_ref,
        delta=delta,
        stable=modulus < 1.0,
    )
