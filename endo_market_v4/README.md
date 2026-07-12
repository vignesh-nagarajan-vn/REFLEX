# endo_market_v4 - the `reflex` package

**Performative prediction in an endogenous OTC corporate-bond market, fourth
and final generation.** A faithful superset of `endo_market_v3` (now frozen in
[`../archive/`](../archive/)) that completes the research program's remaining
to-do items: the lazy-deployment theory + sweep (1.6), the estimator tuning
(Sinkhorn blur, robust ambiguity radius), the **structurally-anchored learned
loop** that closes the v3 loop-level gap, and the verification layer
(numerical proof certificates + Lean 4 formal skeletons). One self-contained
package unifying:

1. the **structural simulator + learned Market Response Operator `T_theta`**
   (lineage: `endo_market_v2`), with the operator **un-blinded** - it can
   *learn* the distribution response `dD/dphi` it was structurally blind to;
2. the **six closed-form math-theory results** (1.1-1.6) as first-class
   modules (`reflex.theory`), woven into training and every experiment
   (predict-then-verify everywhere);
3. the **real-data calibration pipeline** (`reflex.calibration`) - regime-fitted
   microstructure from 36 years of verified market data, shipped in `data/`; and
4. the **verification layer** (`reflex.verification` + `lean/`) - every
   load-bearing identity of the theory machine-checked against the code.

CPU-only, reproducible from `(config, seed)`, no GPU. Python >= 3.9.

## The six pillars (theory ↔ code ↔ experiment)

| # | Closed form (`theory/`, `reflex/theory/`) | ML / measurement counterpart | Experiment |
|---|---|---|---|
| **1.1** | `m = eps*beta/gamma`, boundary `eps < gamma/beta` (`analytic_boundary.py`) | three independent `eps` estimators: BR-slope, Sinkhorn/W1 (tuned blur), CKS flow-curve (`reflex/estimators/`) | `run_sweep` (predict-then-verify), `run_triangulation` |
| **1.2** | PerfGD correction `Delta = -beta*(h-psi)*eps(h)`, optimum `h_PO`, `gamma_PO` (`perfgd.py`) | four loop modes (`equilibrium/loops.py`): blind `rrm` \| `perfgd_analytic` (closed-form surrogate) \| `perfgd_learned` (free-form live summary; the documented negative result) \| **`perfgd_structural`** (v4: GLFT-anchored fits of the loop's own data - the mode that stabilises beyond the boundary) | `run_perfgd` (incl. `--ml`) |
| **1.3** | `eps < gamma/(N_eff*beta)`, `N_eff = 1 + kappa*(N-1)` (`multi_dealer.py`) | genuine N-dealer market with one shared informed pool (`env/multi_dealer.py`), simulated joint cobweb + CRN joint-modulus probes (`equilibrium/joint_loop.py`) | `run_dealers` |
| **1.4** | robust certificate `eps_hat + delta_n < gamma/beta`, `O(1/sqrt(n))` radius (`robust.py`) + the v4-calibrated quantile radius | cross-seed ambiguity bands on every sweep point; coverage-calibrated radius (`calibrate_radius`) | `run_sweep` (bands), `run_tuning` |
| **1.5** | modulus matrix `M = beta*Gamma^-1*E`, `rho(M) < 1`, Woodbury `O(d k^2)` (`factor_scaling.py`) | per-bond sigmas calibrated from real cross-sectional dispersion | `run_universe` |
| **1.6** | K-step outer map `mu(K) = -m + c^K(1+m)`, `gamma_eff(K)`, deadbeat / max-stable K (`lazy_deploy.py`) | signed CRN K-step probe (`estimators/br_slope.py: measure_rgd_response`) + the one-parameter `c`-fit | `run_lazy_deploy` |

Plus the **real-data headline**: `run_fragility` evaluates the 1.1 boundary on
every trading day 1990-2026 - the *stability headroom* `eps*(t) = gamma(t)/beta`
collapses ~4.4x (IG) / ~4.3x (HY) from calm to crisis (and HY sits >10x below
IG in every regime), saturating at its crisis plateau through the GFC and the
March-2020 freeze, while the modulus at observed spreads *falls* (defensive
widening, live on real data). `run_calibrated` tables the a-priori boundary
per `(rating x regime)`. And the **verification headline**: every identity,
inequality and dynamical claim of 1.1-1.6 is re-derived numerically by
`run_certificates` (66 checks, raw + calibrated real-unit configs), with the
logical skeletons formalised in [`lean/`](lean/).

## The ML upgrade in one paragraph (v4 edition)

v2's operator conditioned on a policy summary that was ~constant within a
deployment, so it could not learn how the induced distribution moves when the
policy moves - the RRM loop was *blind* and diverged past `eps*`. v3 fit the
operator on a **window of deployments** so the learned `dD/dphi` was
identified, but the *free-form* corrected loop still failed to stabilise: an
MLP's implied `dJ/dh` diverges from the structural one away from the deployed
regime (the honestly-documented v3 gap). v4 closes it by **anchoring the
learned response to the GLFT structural form**: `perfgd_structural` fits the
theory's own families `tau_hat = C0 + C1*exp(-c*h)` and `u_hat =
A_u*exp(-k_u*h)` plus the realized severity `psi_hat` to the loop's own
deployment history (per-bond, per-step samples - the collection jitter is the
identification the echo chamber cannot destroy), and ascends the estimated
corrected gradient with a trust region and an anti-echo freeze. In the
genuinely RRM-unstable regime the blind loop fails while the structural loop
settles at the *realized* performative optimum - verified against independent
fresh fits (see Honest caveats for why the realized optimum, not the A2
closed form, is the right benchmark). Every loop logs the free-form learned
slope, the structural fitted slope and the analytic `-psi*eps(h)` side by
side: the three-way ML↔math seam.

## Layout

    endo_market_v4/
    |- README.md                 <- this file
    |- pyproject.toml            <- package `reflex` v4.0.0 (install: pip install -e .)
    |- theory/                   <- the six derivations (shipped copies) + code map
    |- lean/                     <- Lean 4 formal skeletons of 1.1-1.6 + honest status
    |- data/                     <- calibration CSVs + daily master panel (provenance in data/README.md)
    |- configs/                  <- default | smoke | sweep_feedback(_smoke) | sweep_adversariality
    |- reflex/
    |  |- config.py, types.py
    |  |- calibration/           <- loader, (rating, regime) -> Config mapping, VIX regimes
    |  |- env/                   <- bonds, clients, liquidity_field, simulator, multi_dealer
    |  |- policy/                <- linear / MLP + closed-form GLFT baseline
    |  |- operator/              <- T_theta: heads, response_operator (+ distribution_response)
    |  |- theory/                <- 1.1 analytic_boundary | 1.2 perfgd | 1.3 multi_dealer
    |  |                            | 1.4 robust (+ calibrate_radius) | 1.5 factor_scaling
    |  |                            | 1.6 lazy_deploy   (numpy-only)
    |  |- equilibrium/           <- data_collection, fit_operator (windowed), optimize_policy
    |  |                            (correction hooks), loops (4 modes), structural_response
    |  |                            (the GLFT-anchored fits), joint_loop, rrm_loop (v2 baseline)
    |  |- estimators/            <- br_slope (+ K-step probe) | sinkhorn (+ blur tuning) | cks | triangulate
    |  |- objective/             <- reward (locked P&L identity), stability penalties
    |  |- analysis/              <- convergence, fragility, metrics, phase, sweep, plots
    |  |- verification/          <- numerical proof certificates for 1.1-1.6
    |  \- utils/                 <- seeding, logging
    |- experiments/              <- run_single | run_sweep | run_perfgd | run_dealers | run_universe
    |                               | run_triangulation | run_fragility | run_calibrated
    |                               | run_lazy_deploy | run_tuning | run_certificates | run_all
    |- outputs/                  <- CSVs + PNGs land here
    \- tests/                    <- 20 files, 142 fast + 10 slow tests (152 total)

## Install, test, run

    pip install -e .
    pytest -q -m "not slow"        # fast suite (142 tests)
    pytest -q -m slow              # + end-to-end boundary/loop tests (10, slow)

    # everything, small settings (~minutes; artifacts in outputs/):
    python -m experiments.run_all --profile smoke
    # paper-grade (~15-25 min CPU measured):
    python -m experiments.run_all --profile full

    # individual experiments:
    python -m experiments.run_certificates                  # 66 numerical proof checks (seconds)
    python -m experiments.run_fragility                     # real-data fragility index (seconds)
    python -m experiments.run_calibrated [--measure]        # per-regime a-priori boundaries
    python -m experiments.run_sweep --config configs/sweep_feedback.yaml
    python -m experiments.run_perfgd --ml                   # blind vs corrected loops (4 modes)
    python -m experiments.run_dealers --probe               # (N, f) systemic surface
    python -m experiments.run_universe                      # 128-bond factor scaling
    python -m experiments.run_triangulation                 # three-way eps agreement
    python -m experiments.run_lazy_deploy                   # K-step map vs theory 1.6
    python -m experiments.run_tuning                        # Sinkhorn blur + robust radius

## Units & calibration

Calibrated configs work in **per-$100-par** units, one step ~ one trading day
(`reflex/calibration/mapping.py` is the single unit-conversion point). Only
`(A, k, sigma, h)` are data-identified; the toxic/informed channel is
*structurally scaled* by documented ratios so cross-regime comparisons isolate
the data-identified variation. `eps` (`clients.toxicity_feedback`) remains the
swept control variable whose critical value the theory predicts per regime.
All probe widths/tolerances are scale-relative so real-unit configs work
unchanged - including the v4 Sinkhorn blur (`reg="auto"` = tuned
`TUNED_REL_REG * sample std`; an absolute blur is the same unit bug as an
absolute probe width).

## Honest caveats

- The dataset is **not trade-level TRACE** (WRDS access pending): spreads are
  VIX-implied, the intensity fits proxy-level. Regime *ordering* of the
  boundary is data-driven; the absolute critical gain is not. See
  `data/README.md`.
- The crisis-regime intensity fit is **degenerate** (`k = 0`, n = 74 days);
  crisis boundaries sit on the anchor floor and are flagged as such.
- **The self-consistent fixed point saturates below `m = 1`** at default-like
  constants (defensive widening, theory 1.1 §6.3): the empirical boundary
  crossing is measured at the probe spread, and beyond it the probe readings
  scatter (seed-level bifurcation) - finite-difference diagnostics there, not
  local slopes.
- **The structural loop optimises the *realized* market, not the A2 closed
  form.** In high-intensity regimes the realized response differs from the
  frozen-reference closed forms at first order through channels they
  deliberately omit (1.1 §9): the `info_cap` saturation (raw toxic notionals
  ~10 exceed the cap 8 at tight spreads in the beyond-boundary demo regime),
  the liquidity-inflation feedback, and severity drift. `perfgd_structural`
  is therefore benchmarked against *independent structural fits* of the same
  market, not against the A2 `h_PO`.
- **The free-form learned mode (`perfgd_learned`) remains a negative
  result**, kept and documented: anchoring to the structural form is what
  closes the gap, not more MLP capacity.
- **The Lean files are reviewed formal statements, not yet compiled**: no
  Lean toolchain exists in the development environment. The numerical
  certificate suite (`run_certificates`, in the test suite) is the
  verification of record; see `lean/README.md` for the full honest status.
- The 1-D frozen-gradient helpers (`best_response`, `Phi'`) omit the
  inventory-risk curvature, so their slope identities hold at
  `lambda_q = 0` while the full-pipeline modulus uses the
  `lambda_q`-inclusive `gamma` - a certificate-discovered convention note
  (cf. 1.1 §7), stated where it matters.
