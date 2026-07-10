# endo_market_v2 (6-17-26)

> **Superseded by [`endo_market_v3/`](../endo_market_v3/) (the `reflex`
> package).** v3 absorbs this library (simulator, operator, loops, the five
> analytic modules) and extends it: un-blinded operator (learned `dD/dphi`),
> PerfGD-corrected loops, a genuine N-dealer market, the epsilon-estimator
> triangulation, and real-data calibration. This folder is kept frozen as the
> generation that first reproduced the `epsilon* ~ 1.3` boundary crossing --
> don't extend it.

**Performative prediction in an endogenous OTC corporate-bond market-making
environment.** A dealer's quoting policy `phi` induces the data distribution
`D(phi)`: tighter quotes summon more informed ("toxic") flow that picks the
dealer off. We study, under *repeated retraining*, when the policy<->distribution
loop **converges** versus **diverges**, and we locate the critical
performative-feedback gain where the loop's contraction modulus crosses 1 --
i.e. the performative-prediction stability boundary `epsilon < gamma/beta`.

CPU-friendly, reproducible from `(config, seed)`, no GPU.

## Headline result

Holding adversariality fixed and sweeping the **performative-feedback gain**
`epsilon` (the `clients.toxicity_feedback` knob), the best-response contraction
modulus `m` rises from a stable regime and crosses the boundary `m = 1`:

| epsilon | 0.0 | 2.0 | 3.0 | 4.0 | 6.0 | 8.0 |
|--------:|----:|----:|----:|----:|----:|----:|
| median modulus `m` | 0.51 | 1.25 | 1.27 | 1.25 | 1.26 | 1.60 |
| fraction of seeds unstable | 0% | 100% | 100% | 100% | 100% | 67% |

(3 seeds; `outputs/phase_diagram_toxicity_feedback.png`, `outputs/sweep_toxicity_feedback_results.csv`.)
The loop is **stable at low feedback and unstable once epsilon exceeds
epsilon\* ~ 1.3**, reproducing the performative-prediction stability boundary in a
structural market model. Beyond the crossing the modulus **saturates near ~1.25**
rather than growing without bound -- the dealer's best response defensively
widens into a region where spread-capture curvature vanishes -- which is a real,
reported feature of the mechanism, not a clean linear blow-up.

### Why epsilon, not alpha, is the clean control
The naive expectation is that *adversariality* `alpha` drives the transition.
It does enter the feedback (toxic slope ~ `alpha * toxicity_feedback`), but
sweeping `alpha` is **confounded**: high `alpha` pushes the dealer to flee to wide
spreads where the toxic spread-response `exp(-decay*h)` has decayed to ~0, so the
local feedback `dtau/dh` -- and hence the modulus -- flattens or even *reverses*.
Empirically the sign of `dm/dalpha` is not robust (it flips with the bond-universe
size). The `toxicity_feedback` gain scales the feedback directly without moving
the operating regime, so it is the clean control variable and the one the headline
sweep uses. The `alpha` sweep is provided (`configs/sweep_adversariality.yaml`) and
its non-monotonicity is an honest secondary finding.

## Mechanism

**Toxic flow** = an `alpha`-independent baseline *level* (which fixes the dealer's
response regime, i.e. the response gain `kappa`) plus an `alpha`- and
`epsilon`-scaled *spread-responsive* term whose slope `dtau/dh` is the feedback:

    toxic ~ gate(edge) * [ info_base_intensity
                           + alpha * toxicity_feedback * info_intensity * exp(-info_spread_decay * h) ]

**Why repeated retraining is blind to performativity.** The learned **Market
Response Operator** `T_theta` (a differentiable surrogate for the market) is refit
from scratch each iteration on one deployment's data and conditions on a
`policy_summary` that is ~constant within a deployment, so it *cannot* learn
`dD/dphi`. When the dealer re-optimizes, the summary is **frozen** at the deployed
regime (Repeated Risk Minimization), so the dealer re-tightens without
anticipating the extra toxic flow it will summon -- that gap, scaled by the
feedback gain, is the cobweb.

**Convexity / well-posedness.** A quadratic quoting-cost anchor
`-quote_anchor_weight*(h - ref)^2` gives the dealer's objective tunable convexity,
which pins the best response (no runaway) and makes the response gain `kappa`
finite and controllable.

**Measuring the modulus (methodological core).** Trajectory-based Lipschitz
estimates fail on the *stable* side: a contracting noisy iteration shrinks to its
sampling-noise floor, where successive-step ratios are ~1, so contraction is
invisible. Instead we estimate the modulus as the **local slope of the
best-response map**, by a symmetric finite difference with **common random
numbers** (the two probes share operator init and all RNG streams, so fit noise
cancels in the difference):

    m = | BR(h_ref + delta) - BR(h_ref - delta) | / (2*delta),  contracts iff m < 1.

See `endo_market/analysis/response_modulus.py`.

## Locked P&L identity (asserted in tests)
Per bond, `S`/`B` = dealer sell/buy volume, `q_after = q0 + B - S`, mid `m`,
fundamental `v`, next `v'`:

    spread_capture         = S*(h + skew) + B*(h - skew)
    inventory_pnl          = q_after*(v' - v)
    adverse_selection_loss = (B - S)*(m - v)
    spread_capture + inventory_pnl - adverse_selection_loss == total economic P&L  (exact)

Objective (maximised) = `spread_capture + inventory_pnl - adverse_selection_loss
- inv_risk_weight*q_after^2 - quote_anchor_weight*(h-ref)^2`, times `pnl_scale`.

## Layout

    endo_market/
    |- config.py, types.py    (config adds multi-dealer n_dealers / toxic_spillover, 1.3)
    |- env/         bonds, liquidity_field, clients (the toxic-flow channel), simulator (T_true)
    |- policy/      Linear / MLP differentiable quoting policies
    |- objective/   reward (locked P&L + quoting-cost anchor), stability diagnostics
    |- operator/    heads + response_operator (T_theta, differentiable rollout)
    |- equilibrium/ data_collection, fit_operator, optimize_policy (pathwise/REINFORCE/RGD),
    |               rrm_loop, perfgd_loop (analytic PerfGD-corrected loop, math-theory 1.2)
    |- analysis/    response_modulus (model-free BR-slope estimator), + the closed-form
    |               analytic-theory modules: analytic_boundary (1.1), multi_dealer_modulus (1.3),
    |               robust_boundary (1.4), factor_reduction (1.5); convergence, sweep, metrics, plots
    \- utils/
    configs/  default.yaml | sweep_feedback.yaml (PRIMARY, epsilon) | sweep_adversariality.yaml (alpha)
    experiments/  run_single.py | run_sweep.py
    tests/    test_simulator, test_policy, test_operator, test_rrm_convergence, + one per analytic
              module (test_analytic_boundary, test_perfgd_loop, test_multi_dealer,
              test_robust_boundary, test_factor_reduction)  -- 63 tests (59 fast + 4 slow)

## Install, test, run

    pip install -e .
    pytest -q -m "not slow"        # 59 fast tests
    pytest -q -m slow              # +4 slow end-to-end / boundary tests (~min each)

    # single run: one RRM trajectory + BR-slope modulus + market metrics + plot
    python -m experiments.run_single --config configs/default.yaml --seed 0

    # phase diagram: sweep epsilon, locate epsilon*, save figure + CSV
    python -m experiments.run_sweep --config configs/sweep_feedback.yaml

## Status

**Complete and tested (63 tests):** environment + exact P&L identity; differentiable
policies; the learned operator `T_theta` (held-out NLL drops ~10 -> ~3); data
collection; policy optimization (pathwise default, REINFORCE and RGD paths, path
logged never switched silently); the RRM loop with a blow-up guard; the BR-slope
modulus estimator; sweep / metrics / plotting; both entry points. The headline
stability-boundary crossing is asserted in `test_rrm_convergence.py` (median over
seeds: stable at epsilon=0, unstable at epsilon=6).

**Analytic theory now implemented (math-theory 1.1--1.5).** The closed-form
counterparts to the swept modulus live alongside the simulator and are each
tested: `analysis/analytic_boundary.py` (1.1: `gamma`, `beta`, `epsilon`, `h*`,
`m = epsilon*beta/gamma`), `equilibrium/perfgd_loop.py` (1.2: the analytic PerfGD
correction, `h_PO`, `gamma_PO`, echo-chamber gap, RRM-diverges-vs-PerfGD-converges),
`analysis/multi_dealer_modulus.py` (1.3: `epsilon < gamma/(N_eff*beta)`, joint
spectrum, mean-field limit), `analysis/robust_boundary.py` (1.4: ambiguity radius,
robust certificate, `O(1/sqrt(n))` rate), `analysis/factor_reduction.py` (1.5:
`d x d` modulus matrix, Woodbury reduction, truncation bound). Derivations and the
module-by-module map are in `../research/math-theory/`.

**Calibration dataset.** `../research/{data_collection,preprocessing}/` hold
a real (public, verified) macro + bond-factor dataset used to calibrate the
microstructure regime; it is *not* trade-level TRACE (that needs WRDS access),
so `h`, per-dealer `q`, and per-bond `A`/`k` are proxied -- stated plainly, not
hidden. The next phase runs the loops/sweeps calibrated from this dataset to
produce the publication-grade phase diagrams.

**Honest caveats / limitations:**
- The per-seed modulus is **noisy near the critical point** (finite 8-bond market);
  results are reported as medians/IQR over seeds. The clean transition is in the
  *median*, not every seed.
- The modulus **saturates** above the crossing instead of growing linearly (best-
  response widening into the low-curvature regime). The transition is stable->
  unstable, not an unbounded blow-up.
- The `alpha` dependence is **confounded / non-monotone** (best-response
  saturation). `epsilon` is the defensible control variable; the `alpha` sweep is
  secondary and its non-monotonicity is reported, not hidden.
- For a publication-grade phase diagram, scale up: larger bond universe (100+),
  more seeds (10-20), report median + IQR bands. The infrastructure supports this
  by editing the config; only compute (single CPU) limits the defaults here.

Nothing here fabricates a result: the demonstrated claim is the epsilon-driven
stability transition with the caveats above.
