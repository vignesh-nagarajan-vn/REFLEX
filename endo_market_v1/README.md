# endo_market_v1

> **Renamed.** This folder was previously `endo_market/`; it is the first Python
> iteration of the REFLEX program, superseded by `endo_market_v2/` and now
> `endo_market_v3/`. Kept frozen for lineage; don't extend it. (The *inner*
> Python package it ships is still named `endo_market` — code unchanged.)

**Performative prediction in an endogenous OTC corporate-bond market-making
environment.** The dealer's quoting policy `phi` induces the data distribution
`D(phi)` (tighter quotes summon more informed/toxic flow). We study when the
policy<->distribution loop, under repeated retraining, **converges** versus
**diverges** as the adversariality / toxic-flow knob `alpha in [0, 1]` varies,
and we locate the critical `alpha*` where the empirical contraction modulus
`L_hat` crosses 1.

This is a research codebase: CPU-friendly, reproducible from `(config, seed)`,
no GPU required.

> STATUS (honest): the environment, learned operator, and policy optimization
> (Phases 1-5) are complete and tested (18 passing tests). The outer
> Repeated-Risk-Minimization loop (Phase 6) **runs end-to-end** but the headline
> empirical claim -- `L_hat` crossing 1 at a tunable `alpha* ~ 0.5` -- is **not
> yet reproduced/validated**. The mechanism is in place and analytically
> motivated (below), but the two gain knobs still need tuning and the
> phase-diagram experiments (Phase 7) are not built. See "What works / what's
> left".

---

## The idea

### Performative setup
A dealer quotes a half-spread `h` (and inventory skew) across a small universe of
bonds. Two client streams arrive each step:

* **Uninformed** flow: benign, decays with the quoted spread, earns the dealer
  the spread with little adverse selection.
* **Informed (toxic)** flow: picks the dealer off (buys when the mid is too low,
  sells when too high). Its intensity is gated by `alpha` and, crucially,
  **responds to the spread**: tighter quotes admit proportionally more toxic flow.

The toxic flow is split into

    toxic ~ gate(edge) * [ info_base_intensity                                  # alpha-independent LEVEL
                           + alpha * toxicity_feedback * info_intensity * exp(-elasticity*h) ]  # alpha-scaled SLOPE

The **level** is alpha-independent on purpose (it fixes the dealer's response
regime, i.e. the gain `kappa = dh*/dtau`), while the **slope** `dtau/dh ~ alpha`
is the feedback that drives the instability.

### Why repeated retraining is blind to it
The learned **Market Response Operator** `T_theta` (a differentiable surrogate
for the market) is refit from scratch each iteration on one deployment's data,
and conditions on a `policy_summary` that is ~constant within a deployment. So a
freshly-refit operator structurally **cannot** learn `dD/dpolicy` (the
performative derivative) -- exactly the performative-prediction insight that
repeated retraining ignores performativity. During policy optimization the
summary is **frozen** at the deployed `phi_k` (Repeated Risk Minimization), so
the dealer re-tightens without anticipating that it will summon more toxic flow
at the next deployment. That gap, scaled by `alpha`, is the cobweb.

### The cobweb / phase transition
With an explicit, tunable **quoting-cost convexity** in the dealer's objective (a
quadratic anchor on the spread), the best-response map linearises near its fixed
point to

    h_{k+1} - h*  ~  -K*alpha * (h_k - h*),   K = (toxic slope gain) / (objective curvature)

i.e. the classic cobweb `e_{k+1} = -m*e_k` with **modulus `m = K*alpha`** (linear
in alpha, through the origin). It contracts iff `m < 1`, so

    alpha*  =  1 / K        (converges for alpha < alpha*, diverges for alpha > alpha*)

and `K` is set by the `toxicity_feedback` and quoting-cost-weight knobs -- this is
the performative-prediction stability condition `epsilon < gamma/beta` made
concrete. The empirical estimator `L_hat` is the median ratio of successive
iterate step sizes, which equals `m` for this map.

---

## Locked P&L accounting (verified as an exact identity)
Per bond, with `S` = dealer sell volume, `B` = dealer buy volume,
`q_after = q0 + B - S`, mid `m`, fundamental `v`, next fundamental `v'`:

    spread_capture         = S*(h + skew) + B*(h - skew)
    inventory_pnl          = q_after*(v' - v)            # carry, mean-zero
    adverse_selection_loss = (B - S)*(m - v)             # positive in expectation, ~ alpha
    spread_capture + inventory_pnl - adverse_selection_loss  ==  total economic P&L   (exact)

Dealer objective (maximised) =
`spread_capture + inventory_pnl - adverse_selection_loss - inv_risk_weight*q_after^2`
(times `pnl_scale`), minus the quoting-cost anchor. Performative risk
`PR = -E[objective]`. The identity is asserted in `tests/test_simulator.py`.

---

## Repository layout

    endo_market/
    |- config.py                 # nested dataclass config + YAML loader (no Hydra)
    |- types.py                  # MarketState, Quotes, Fills, Transition, policy_summary
    |- env/
    |  |- bonds.py               # static bond universe + correlation (Cholesky)
    |  |- liquidity_field.py     # latent mean-reverting, spread-degradable liquidity
    |  |- clients.py             # uninformed + informed (toxic) flow -- the alpha channel
    |  \- simulator.py           # T_true: structural ground-truth dynamics + P&L
    |- policy/
    |  \- dealer_policy.py       # Linear / MLP differentiable quoting policies
    |- objective/
    |  \- reward.py              # locked dealer objective; fundamental/observable marking
    |- operator/
    |  |- heads.py               # Gaussian / mixture output heads (reparam. sampling)
    |  \- response_operator.py   # T_theta: differentiable surrogate dynamics + rollout
    |- equilibrium/
    |  |- data_collection.py     # deploy phi on T_true, collect transitions (w/ jitter)
    |  |- fit_operator.py        # MLE fit of T_theta with held-out early stopping
    |  |- optimize_policy.py     # policy opt vs frozen T_theta (pathwise / REINFORCE / RGD)
    |  \- rrm_loop.py            # the Repeated-Risk-Minimization outer loop
    |- analysis/
    |  \- convergence.py         # empirical Lipschitz, fixed-point residual, classifier
    \- utils/                    # seeding, logging

    configs/
    |- default.yaml              # single-run config
    \- sweep_adversariality.yaml # alpha-grid sweep config (phase diagram)
    tests/                       # test_simulator, test_policy, test_operator (18 tests)

---

## Install & test

    pip install -e .            # editable install (CPU torch is fine)
    pytest -q                   # 18 tests: env, P&L identity, policy, operator fit

## Run (single alpha)

    from endo_market.config import load_config
    from endo_market.equilibrium import run_rrm
    from endo_market.analysis import empirical_lipschitz, classify_run

    cfg = load_config("configs/default.yaml")
    cfg.clients.alpha = 0.3
    traj = run_rrm(cfg, seed=0, verbose=True)

    h = traj.central_spreads                      # scalar cobweb coordinate per iterate
    print("L_hat =", empirical_lipschitz(h, burn_in=1))
    print("class:", classify_run(h))

---

## What works / what's left

**Done and tested (Phases 1-5, 18 passing tests):**

* `env/` -- bond universe, liquidity field, client flow, and `T_true` simulator,
  with the **P&L decomposition asserted as an exact identity** and the
  toxicity-vs-spread / toxicity-vs-alpha signs verified.
* `policy/` -- differentiable Linear and MLP quoting policies (flatten/load,
  gradients verified).
* `operator/` + `equilibrium/fit_operator.py` -- `T_theta` with Gaussian/mixture
  heads; fitting **reduces held-out NLL** (asserted, ~10.0 -> ~8.5).
* `equilibrium/{data_collection, optimize_policy}` -- collection with exploration
  jitter; policy optimization against frozen `T_theta` via **pathwise** gradients
  (default), with **REINFORCE** and **RGD** paths (path is logged, never switched
  silently).
* `analysis/convergence.py` -- `empirical_lipschitz`, `fixed_point_residual`,
  `is_oscillating`, `classify_run`.
* `equilibrium/rrm_loop.py` -- the full RRM loop **runs end-to-end** (collect ->
  refit -> evaluate true risk -> re-optimize), with a blow-up guard that marks a
  run divergent instead of crashing.

**Not done / not yet validated:**

1. **Tuning the phase transition (the headline result).** The loop runs, but
   `L_hat` does **not yet** cleanly satisfy `L_hat(low alpha) < 1 < L_hat(high
   alpha)` with `alpha* ~ 0.5`. The mechanism (alpha-independent toxic level +
   alpha-scaled slope + quoting-cost convexity) is in place and analytically
   gives `m = K*alpha`, but the two knobs (`clients.toxicity_feedback` and the
   quoting-cost weight) still need to be fit so `K ~ 2`. The quoting-cost anchor
   term itself was **in the middle of being added** to `objective/reward.py` when
   work paused -- this is the immediate next step.
2. **`tests/test_rrm_convergence.py`** -- the core acceptance test (seeded:
   low-alpha contracts, high-alpha diverges) is **not written** (blocked on #1).
3. **`objective/stability.py`** -- entropy/HHI/toxicity/Lipschitz diagnostics
   (config weights default to 0) -- **not written**.
4. **Phase 7 -- experiments & analysis:** `analysis/sweep.py`,
   `analysis/plots.py` (phase diagram), `analysis/metrics.py`, and the
   `experiments/run_single.py` / `run_sweep.py` entry points -- **not written**.
   The full sweep (8 alpha x 3 seeds x ~15 iters) is ~30 min on 1 CPU, so the
   sweep config is deliberately smaller and may need further trimming.

In short: the scientific scaffolding is built and the cheap-to-verify pieces are
tested, but **the central convergence/divergence dichotomy is not yet
empirically demonstrated.** Don't cite a result from this repo until #1 and #2
are done.
