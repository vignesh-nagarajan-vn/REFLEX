# endo_market_v3 - the `reflex` package

**Performative prediction in an endogenous OTC corporate-bond market, third
generation.** One self-contained package that unifies the three previously
separate strands of the REFLEX program:

1. the **structural simulator + learned Market Response Operator `T_theta`**
   (lineage: `endo_market_v2`), with the operator **un-blinded** - it can now
   *learn* the distribution response `dD/dphi` it was structurally blind to;
2. the **five closed-form math-theory results** (1.1–1.5) as first-class
   modules (`reflex.theory`), woven into training and every experiment
   (predict-then-verify everywhere); and
3. the **real-data calibration pipeline** (`reflex.calibration`) - regime-fitted
   microstructure from 36 years of verified market data, shipped in `data/`.

CPU-only, reproducible from `(config, seed)`, no GPU. Python >= 3.9.

## The five pillars (theory ↔ code ↔ experiment)

| # | Closed form (`theory/`, `reflex/theory/`) | ML / measurement counterpart | Experiment |
|---|---|---|---|
| **1.1** | `m = eps*beta/gamma`, boundary `eps < gamma/beta` (`analytic_boundary.py`) | three independent `eps` estimators: BR-slope, Sinkhorn/W1, CKS flow-curve (`reflex/estimators/`) | `run_sweep` (predict-then-verify), `run_triangulation` |
| **1.2** | PerfGD correction `Delta = -beta*(h-psi)*eps(h)`, optimum `h_PO`, `gamma_PO` (`perfgd.py`) | windowed operator + live-summary optimisation: the **learned** `dD/dphi` enters the gradient (`equilibrium/loops.py`, modes `rrm \| perfgd_analytic \| perfgd_learned`) | `run_perfgd` (incl. `--ml`) |
| **1.3** | `eps < gamma/(N_eff*beta)`, `N_eff = 1 + kappa*(N-1)` (`multi_dealer.py`) | genuine N-dealer market with one shared informed pool (`env/multi_dealer.py`), simulated joint cobweb + CRN joint-modulus probes (`equilibrium/joint_loop.py`) | `run_dealers` |
| **1.4** | robust certificate `eps_hat + delta_n < gamma/beta`, `O(1/sqrt(n))` radius (`robust.py`) | cross-seed ambiguity bands on every sweep point | `run_sweep` (bands) |
| **1.5** | modulus matrix `M = beta*Gamma^-1*E`, `rho(M) < 1`, Woodbury `O(d k^2)` (`factor_scaling.py`) | per-bond sigmas calibrated from real cross-sectional dispersion | `run_universe` |

Plus the **real-data headline**: `run_fragility` evaluates the 1.1 boundary on
every trading day 1990–2026 - the *stability headroom* `eps*(t) = gamma(t)/beta`
collapses ~4x (IG) / ~13x (HY) from calm to crisis, peaking at Lehman
(2008-10-06) and the March-2020 freeze, while the modulus at observed spreads
*falls* (defensive widening, live on real data). `run_calibrated` tables the
a-priori boundary per `(rating x regime)`.

## The ML upgrade in one paragraph

v2's operator conditioned on a policy summary that was ~constant within a
deployment, so it could not learn how the induced distribution moves when the
policy moves - the RRM loop was *blind* and diverged past `eps*`. v3 fits the
operator on a **window of recent deployments** (`operator.context_window`),
each row carrying its own deployment's summary, so the summary-dependence - the
learned `dD/dphi`, read out by `operator.distribution_response()` - is
identified. Optimisation then either stays frozen (blind baseline), adds the
**closed-form** correction (`perfgd_analytic`), or lets gradients flow through
the **live** summary (`perfgd_learned` - the fully-ML path). Every loop logs
the learned toxic slope next to the analytic `-psi*eps(h)`: the ML↔math seam.

## Layout

    endo_market_v3/
    |- README.md                 <- this file
    |- pyproject.toml            <- package `reflex` (install: pip install -e .)
    |- theory/                   <- the five derivations (shipped copies) + code map
    |- data/                     <- calibration CSVs + daily master panel (provenance in data/README.md)
    |- configs/                  <- default | smoke | sweep_feedback(_smoke) | sweep_adversariality
    |- reflex/
    |  |- config.py, types.py
    |  |- calibration/           <- loader, (rating, regime) -> Config mapping, VIX regimes
    |  |- env/                   <- bonds, clients, liquidity_field, simulator, multi_dealer
    |  |- policy/                <- linear / MLP + closed-form GLFT baseline
    |  |- operator/              <- T_theta: heads, response_operator (+ distribution_response)
    |  |- theory/                <- 1.1 analytic_boundary | 1.2 perfgd | 1.3 multi_dealer
    |  |                            | 1.4 robust | 1.5 factor_scaling   (numpy-only)
    |  |- equilibrium/           <- data_collection, fit_operator (windowed), optimize_policy
    |  |                            (correction hooks), loops (3 modes), joint_loop, rrm_loop (v2 baseline)
    |  |- estimators/            <- br_slope | sinkhorn | cks | triangulate
    |  |- objective/             <- reward (locked P&L identity), stability penalties
    |  |- analysis/              <- convergence, fragility, metrics, phase, sweep, plots
    |  \- utils/                 <- seeding, logging
    |- experiments/              <- run_single | run_sweep | run_perfgd | run_dealers | run_universe
    |                               | run_triangulation | run_fragility | run_calibrated | run_all
    |- outputs/                  <- CSVs + PNGs land here
    \- tests/                    <- 15 files, 103 fast + 7 slow tests (110 total)

## Install, test, run

    pip install -e .
    pytest -q -m "not slow"        # fast suite
    pytest -q -m slow              # + end-to-end boundary/loop tests (slow)

    # everything, small settings (~20 min CPU; artifacts in outputs/):
    python -m experiments.run_all --profile smoke
    # paper-grade (hours):
    python -m experiments.run_all --profile full

    # individual experiments:
    python -m experiments.run_fragility                     # real-data fragility index (seconds)
    python -m experiments.run_calibrated [--measure]        # per-regime a-priori boundaries
    python -m experiments.run_sweep --config configs/sweep_feedback.yaml
    python -m experiments.run_perfgd --ml                   # blind vs corrected loops
    python -m experiments.run_dealers --probe               # (N, f) systemic surface
    python -m experiments.run_universe                      # 128-bond factor scaling
    python -m experiments.run_triangulation                 # three-way eps agreement

## Units & calibration

Calibrated configs work in **per-$100-par** units, one step ~ one trading day
(`reflex/calibration/mapping.py` is the single unit-conversion point). Only
`(A, k, sigma, h)` are data-identified; the toxic/informed channel is
*structurally scaled* by documented ratios so cross-regime comparisons isolate
the data-identified variation. `eps` (`clients.toxicity_feedback`) remains the
swept control variable whose critical value the theory predicts per regime.
All probe widths/tolerances are scale-relative so real-unit configs work
unchanged.

## Honest caveats

- The dataset is **not trade-level TRACE** (WRDS access pending): spreads are
  VIX-implied, the intensity fits proxy-level. Regime *ordering* of the
  boundary is data-driven; the absolute critical gain is not. See
  `data/README.md`.
- The crisis-regime intensity fit is **degenerate** (`k = 0`, n = 74 days);
  crisis boundaries sit on the anchor floor and are flagged as such.
- **The self-consistent fixed point saturates below `m = 1`** at default-like
  constants (defensive widening, theory 1.1 §6.3): the empirical boundary
  crossing is a statement about the *local* retraining map at the probe
  spread (the loop's operating region), not about the fixed point.  The sweep
  overlays the closed form at both spreads and labels them.
- **The measured modulus is protocol-dependent**: the BR-slope probe measures
  the finite-budget retraining map, whose slope depends on the per-deployment
  optimisation budget (the lazy-deploy knob, 1.1 §6.2).  Probes use the
  loop's own budget; levels are comparable within a protocol, not across.
- **Loop-level PerfGD does not yet stabilise the learned loop.**  The 1.2
  claims hold in closed form (verified: the cobweb diverges in the unstable
  regime, the corrected 1-D ascent converges to `h_PO`), but in the full ML
  loop the blind operator's implied `dJ/dh` diverges from the structural one,
  the corrected equilibrium lands far from `h_PO`, and on-trajectory learned
  `dD/dphi` flips sign in the unstable regime.  `run_perfgd --ml` therefore
  reports the ML<->math *seam diagnostics*, not a stabilisation proof - the
  gap is a documented finding, not a hidden one.
- The learned `dD/dphi` needs `operator.context_window >= 2` and enough spread
  variation across deployments; with a single deployment the learned slope is
  noise (the loop warns).
- Multi-dealer runs inflate the shared liquidity field (combined-flow boost),
  which can push informed volume into the `info_cap` saturation - scale
  `liq_flow_boost` down or `info_cap` up for flow-allocation studies, and
  probe the joint modulus only in the interior regime
  (`interior_probe_config`; deep past the boundary the closed-form BR rails
  at the spread cap and the probe reads 0, flagged by `br_clipped`).
- Smoke-profile outputs prove the pipeline, not the science; paper-grade
  figures come from `--profile full`.
