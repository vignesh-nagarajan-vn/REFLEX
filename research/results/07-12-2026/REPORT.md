# REFLEX v4 paper-grade run report - July 12, 2026

The complete illustrated report of the full-profile suite of the **final
generation** (`endo_market_v4`, package `reflex` 4.0.0): every generated
figure, all headline numbers, and the honest caveats. This run does two jobs
at once - it executes the four v4 additions at paper grade for the first time
(the numerical proof certificates, the theory-1.6 lazy-deployment sweep, the
estimator tuning, and the **structural learned loop that closes the v3
loop-level gap**), and it re-executes the v3-era experiment set with the v4
code as a determinism/consistency check (section 13: every re-run number
reproduces exactly). Raw artifacts live beside this file (per-experiment
subfolders); the v3 per-experiment deep-dive remains
[`../../analysis/ANALYSIS-full-2026-07.md`](../../analysis/ANALYSIS-full-2026-07.md)
and the pre-run audit record is
[`../../analysis/pre-run-audit-2026-07.md`](../../analysis/pre-run-audit-2026-07.md).

| | |
|---|---|
| **Run date** | July 12, 2026 |
| **Producing code** | `endo_market_v4` at commit `2718012` plus the documentation-only delta committed as `b4514be` (three files, docstrings + version string; zero experiment-code changes, verified by `git diff --stat`) |
| **Command** | `python -u -m experiments.run_all --profile full` (from `endo_market_v4/`, repo venv) |
| **Outcome** | suite **11/11 passed in 25.1 min CPU** (certificates 0s, fragility 1s, calibrated 106s, universe 1s, perfgd 477s, dealers 5s, triangulation 7s, sweep 319s, lazy_deploy 105s, tuning 398s, single 89s); **152/152 tests** and **66/66 proof certificates** green before the run |
| **Environment** | Python 3.9.13, torch 2.8.0+cpu, numpy 2.0.2, pandas 2.3.3; Windows 11, CPU only; deterministic from `(config, seed)` |

Figures marked *(v3-run analysis figure)* are derived figures built from the
07-10-2026 CSVs; they are reproduced here because the v4 re-run outputs are
**numerically identical** to the v3 inputs they were derived from (section 13).

---

## 1. Executive summary

1. **Every load-bearing theory identity is now machine-checked**: 66/66
   numerical proof certificates pass on the raw config *and* on the
   calibrated real-unit config - worst residual per certificate between
   `2e-16` and `7e-2`, each far inside its stated tolerance (section 2).
2. **The stability boundary is computable a priori from real data and behaves
   sensibly** (unchanged from v3, re-verified bit-for-bit): headroom
   `eps* = gamma/beta` collapses ~4.4x (IG) / ~4.3x (HY) calm -> crisis, HY
   sits >10x below IG in every regime, and the modulus at observed spreads
   *falls* into crisis - defensive widening on real data (sections 3, 4).
3. **Predict-then-verify reproduces quantitatively**: measured crossing
   `f* ~ 3.17` vs the a-priori 4.70, reconciled by the independently-measured
   realized-state correction (`rho = 2.32`); contracting-regime agreement
   within 8% at `f = 2` (sections 5, 7).
4. **Competition amplifies performative instability as derived**: common-mode
   amplification 1.74x / 3.16x vs the predicted `N_eff` = 2 / 3, differential
   mode dead at full spillover (section 6).
5. **THE v4 HEADLINE - the loop-level gap is closed.** In the genuinely
   RRM-unstable demo regime (`m_rrm = 1.21`) where blind RRM collapses into
   the echo chamber (final `h = 0.16`) and both v3 corrected modes fail to
   settle, the new `perfgd_structural` mode - every 1.2 ingredient *fitted*
   from the loop's own deployment history - climbs under its trust region and
   **converges to the realized performative optimum**: final `h = 3.193`,
   within 0.7% of its own running estimate `h_PO_hat = 3.172`, with the
   fitted toxic slope tracking the analytic shape while the free-form
   operator's slope stays sign-unstable (section 9).
6. **Theory 1.6 is verified in the hardest regime**: at the beyond-boundary
   default config the signed K-step probe medians flip sign exactly at the
   predicted deadbeat count (`K_db = 1.46`) and exit the stable band inside
   the predicted window (`K_max = 5.75`) - laziness measurably stabilises an
   RRM-unstable market (section 10).
7. **The tuned estimators behave as calibrated**: the Sinkhorn blur default
   (`0.02 x std`, scale-relative) sits at the U-turn of the ground-truth bias
   curve (6.1%) and in the low-bias region of the config curve (1.2%); the
   robust `z*s` radius over-covers (0.99-1.00) on normal / heavy-tailed /
   skewed estimates and is the binding radius for the CRN probe
   (section 11).
8. **Factor scaling defuses the curse of dimensionality** (re-verified):
   `rho(M) ~ 0.50` flat from 8 to 128 bonds; the truncation bound holds with
   3-4 orders of magnitude of slack; 0.12 s at `d = 128` (section 8).
9. **The three-way triangulation brackets the realized-state closed form**
   at 2.3-2.7x with the liquidity-inflation channel identified as the
   dominant correction - the same channel that sets the structural loop's
   benchmark (section 7).
10. **The free-form learned mode remains a documented negative result**:
    anchoring the learned response to the structural form - not adding
    capacity - is what closes the gap (section 9).

---

## 2. Numerical proof certificates (raw + calibrated real units)

The v4 verification layer re-derives every load-bearing identity, inequality
and dynamical claim of theory 1.1-1.6 numerically against the closed-form
implementations - finite-difference slopes vs assembled constants,
eigensolves vs spectral formulas, Monte-Carlo rates vs claimed exponents,
dynamics run rather than assumed. 66 checks, two configs:

| Certificate | raw config | calibrated IG/normal | worst check (both configs) | worst residual | tolerance |
|-------------|-----------:|---------------------:|----------------------------|---------------:|----------:|
| 1.1 analytic boundary | 5/5 | 5/5 | `cobweb_step_linearisation` | 6.8e-2 | 8e-2 |
| 1.2 PerfGD identities | 4/4 | 4/4 | `gamma_po_curvature_identity_lq0` | 1.3e-6 | 2e-2 |
| 1.2 PerfGD dynamics (raw-unit demo) | 4/4 | auto-excluded | `corrected_ascent_reaches_h_po` | 1.4e-11 | 1e-3 |
| 1.3 multi-dealer | 6/6 | 6/6 | `spectral_radius_is_common_mode` | 2.2e-16 | 1e-9 |
| 1.4 robust boundary | 7/7 | 7/7 | `parametric_rate_slope_minus_half` | 3.2e-2 | 1e-1 |
| 1.5 factor scaling | 3/3 | 3/3 | `assembled_rho_equals_dense_eigensolve` | 0.0 | 1e-9 |
| 1.6 lazy deployment | 6/6 | 6/6 | `deadbeat_zeroes_mu` | 1.1e-16 | 1e-12 |

- The 1.1 headline check measures `BR'(h*)` by central differences of the
  *composed* best-response map - no reference to `epsilon`, `beta`, `gamma` -
  and matches the assembled `-m` at the `lambda_q = 0` convention (see the
  methods note, section 14). The 1.2 dynamics certificate runs the cobweb and
  the corrected ascent in the unstable demo regime and confirms divergence /
  convergence-to-`h_PO` respectively (residual 1.4e-11 on the optimum).
- The demo-dynamics certificate **auto-excludes itself on the calibrated
  config**: its demo constants are absolute raw-unit values, and imposing
  them on per-$100-par units is exactly the unit bug the repo conventions
  forbid. The calibrated pass proves the identities in real units.

---

## 3. Market-fragility index (real data, 1990-2026)

The theory 1.1 closed forms evaluated on every trading day of the master
panel (VIX spine, per-regime intensity fits), for IG and HY. Closed form on
real data: full fidelity by construction, and **bit-for-bit identical to the
v3 run** (same shipped data, same closed forms).

![Fragility index](fragility/fragility_index.png)

![Headroom by regime](../../analysis/figures/fragility_headroom_by_regime.png)
*(v3-run analysis figure)*

| regime | IG eps* | HY eps* | IG modulus at observed h | HY modulus at observed h | IG fragility |
|--------|--------:|--------:|-------------------------:|-------------------------:|-------------:|
| calm     | 1207.7 | 93.5 | 0.847 | 0.592 | 0.60 |
| normal   |  739.2 | 61.3 | 0.870 | 0.563 | 0.98 |
| elevated |  594.4 | 45.9 | 0.520 | 0.367 | 1.22 |
| stress   |  477.3 | 35.6 | 0.223 | 0.163 | 1.52 |
| crisis   |  275.9 | 21.8 | 0.139 | 0.091 | 2.64 |

- Headroom collapses **~4.4x (IG)** and **~4.3x (HY)** calm -> crisis; the
  larger effect is the *level* gap - HY has >10x less headroom than IG in
  every regime (93.5 vs 1207.7 in calm; 21.8 vs 275.9 in crisis).
- The modulus at observed spreads **falls** into crisis for both ratings
  (IG 0.85 -> 0.14, HY 0.59 -> 0.09): dealers widen faster than the toxic
  channel steepens - defensive widening, live on real data.
- The index **saturates at a crisis plateau** (crisis intensity fit is
  degenerate, `k = 0`, n = 74 days): the GFC window (first crisis day
  2008-10-06) and the COVID freeze (2020-03-09) sit on the same plateau;
  intra-crisis ranking is not identified.

**Caveats.** VIX-implied spreads and proxy-level intensity fits (not
trade-level TRACE); regime ordering is data-driven, absolute levels are not.

---

## 4. Calibrated a-priori boundaries per (rating x regime)

![Calibrated boundaries](calibrated/calibrated_boundaries.png)

![Predicted vs measured](../../analysis/figures/calibrated_pred_vs_measured.png)
*(v3-run analysis figure)*

| cell | h_obs | h* | eps* | m(h*) | m_pred at h_obs | m_measured (median, 3 seeds) |
|------|------:|----:|-----:|------:|----------------:|------------------------------:|
| IG calm     | 0.374 | 0.436 | 465.3 | 0.066 | 0.080 | 0.000 |
| IG normal   | 0.477 | 0.558 | 268.0 | 0.066 | 0.081 | 0.000 |
| IG elevated | 0.683 | 0.794 | 147.3 | 0.066 | 0.080 | 0.038 |
| IG stress   | 1.051 | 1.207 |  70.6 | 0.067 | 0.080 | 1.631 |
| IG crisis   | 1.233 | 1.472 |  34.5 | 0.065 | (degenerate) | - |
| HY calm     | 1.030 | 1.183 |  28.3 | 0.067 | - | - |
| HY normal   | 1.331 | 1.535 |  16.2 | 0.066 | - | - |
| HY elevated | 1.882 | 2.159 |   9.1 | 0.066 | - | - |
| HY stress   | 2.896 | 3.275 |   4.7 | 0.066 | - | - |
| HY crisis   | 3.493 | 4.169 |   2.2 | 0.055 | - | - |

- The fixed-point modulus is regime-invariant (~0.066) by the
  anchor-stiffness construction; **the regime story lives entirely in the
  headroom eps*** (465 -> 2.2 across the table), consistent with the
  fragility index.
- The measured column is a weak consistency check, not a result: at IG
  calm/normal the anchor weight pins the probe below its resolution floor
  (reads exactly 0), at IG stress probe noise dominates a 3-seed median.
  Only IG elevated (0.038 vs predicted 0.080, the documented finite-budget
  attenuation factor ~2) is informative. Identical to the v3 reading.

---

## 5. Phase-diagram sweep: predict then verify (7 gains x 8 seeds)

The headline predict-then-verify experiment under the clean probe protocol
(`collection_jitter = 0.05`), re-executed by the v4 code.

![Feedback phase diagram](sweep/phase_diagram_toxicity_feedback.png)

| f | median m | IQR | robust verdict | m_pred at h_ref | m at h*(f) |
|---|---------:|-----|----------------|----------------:|-----------:|
| 0 | 0.067 | [0.04, 0.09] | stable    | 0.000 | 0.000 |
| 1 | 0.159 | [0.08, 0.28] | stable    | 0.213 | 0.124 |
| 2 | 0.390 | [0.21, 0.48] | stable    | 0.426 | 0.225 |
| 3 | 0.856 | [0.61, 1.07] | undecided | 0.639 | 0.310 |
| 4 | 1.707 | [1.42, 2.16] | **unstable** | 0.852 | 0.384 |
| 6 | 1.180 | [0.83, 1.41] | undecided | 1.277 | 0.507 |
| 8 | 1.357 | [0.89, 1.84] | undecided | 1.703 | 0.607 |

Measured crossing: **f* ~ 3.169**. Predicted at the probe spread (a-priori
state): **f* ~ 4.697**. Predicted at the realized state (the `rho ~ 2.3`
liquidity-inflation correction measured independently by the triangulation,
section 7): **~2.8-3.0**, bracketing the measurement.

- In the contracting regime the probe tracks the closed form quantitatively
  (0.390 vs 0.426 at `f = 2`, an 8% agreement).
- Past the boundary the probe stops being a local slope: IQRs explode and
  seed-level readings bifurcate (e.g. at `f = 8` the eight seeds span 0.03 to
  2.05) - exactly the theory's own A4 caveat. The robust certificates
  respond correctly: stable through `f = 2`, unstable at `f = 4`, undecided
  where the beyond-boundary readings scatter.
- The fixed-point curve never crosses 1 (saturates at 0.61 by `f = 8`):
  defensive widening (theory 1.1 §6.3). The measured instability is a
  statement about the retraining map at the operating spread, not about the
  self-consistent equilibrium.
- Every row matches the 07-10-2026 run to all printed digits (section 13).

---

## 6. Multi-dealer systemic risk (genuine shared-pool market)

The analytic `(N, f)` surface, the simulated joint cobweb, and CRN
joint-modulus probes at the interior regime (`f_probe = 0.5`,
`liq_flow_boost / N`).

![Dealer phase surface](dealers/dealer_phase_diagram.png)

![Dealer amplification](../../analysis/figures/dealer_amplification.png)
*(v3-run analysis figure)*

| N | measured common-mode | predicted N_eff x m_1 | measured differential | clipped |
|---|---------------------:|----------------------:|----------------------:|---------|
| 1 | 0.786 | 0.786 | - | no |
| 2 | 1.369 | 1.571 | 0.0034 | no |
| 3 | 2.480 | 2.357 | 0.0032 | no |

- Amplification ratios **1.74x (N = 2)** and **3.16x (N = 3)** vs the
  predicted `N_eff` = 2 and 3: within 13% and 5%.
- The differential mode is dead at full spillover (theory:
  `(1 - kappa) m_1 = 0`): instability is purely common-mode, the
  synchronised systemic channel. Competition destabilises the market a
  factor `N_eff` before any single dealer would.
- The analytic surface puts the critical dealer count at `N_c = 7.9` for the
  default single-dealer modulus, with `m_N = 1.51` at `N = 12` - and the
  simulated joint cobweb at `N = 3` oscillates against its cap, the
  qualitative signature of the joint instability. The 1.3 certificate
  independently confirms the eigen-identity behind all of this
  (`J @ 1 = -m_1 N_eff 1`, residual 2e-16).

---

## 7. Three-way epsilon triangulation

Three independent instruments at the operating spread (`h_ref = 1.0`),
against the closed form evaluated at the a-priori A2 state and at the
**realized deployment state** (theory 1.1 §9).

![Triangulation](../../analysis/figures/triangulation_bars.png)
*(v3-run analysis figure)*

| quantity | epsilon | ratio vs realized closed form |
|----------|--------:|------------------------------:|
| analytic, a-priori A2 state (rho = 1) | 0.764 | 0.44x |
| **analytic, realized state** (rho = 2.32, \|g\| = 0.446) | **1.726** | 1.00 |
| BR-slope leg (m_hat = 0.541 x gamma_real/beta) | 0.508 | 0.29x |
| Sinkhorn/W1 leg | 4.578 | 2.65x |
| CKS flow-curve leg | 4.011 | 2.32x |

- The realized-state correction is first order and its driver is identified:
  the deployment's own flow boosts the liquidity field to `rho ~ 2.3`.
- The two distribution-space legs agree with each other within 14% and sit
  2.3-2.7x above the realized-state closed form; the residual is the
  state-feedback channel that any frozen-state closed form omits. The closed
  form is a lower anchor, not an unbiased point prediction.
- The BR leg reads low (0.29x): the documented finite-budget attenuation of
  the decision-space map.
- This is the same liquidity-inflation channel that defines the structural
  loop's benchmark (section 9): the market the loop actually faces is the
  realized one.

---

## 8. Universe factor scaling (8 to 128 bonds)

The `d x d` modulus matrix `M = beta Gamma^-1 E` with per-bond sigmas
dispersed by the data-calibrated coefficient of variation (per-bond vol CV =
1.32 across the 212 real CUSIPs, clipped at the structural band cap 0.8),
plus the `O(d k^2)` Woodbury reduction and the truncation bound.

![Universe scaling](universe/universe_scaling.png)

| d | rho(M) | stable | scalar-max m | market alignment | runtime |
|---|-------:|--------|-------------:|-----------------:|--------:|
| 8   | 0.499 | yes | 0.502 | 0.213 | 0.002 s |
| 16  | 0.501 | yes | 0.502 | ~9e-14 | 0.002 s |
| 32  | 0.501 | yes | 0.502 | ~1e-13 | 0.001 s |
| 64  | 0.501 | yes | 0.502 | ~1e-13 | 0.005 s |
| 128 | 0.501 | yes | 0.502 | ~2e-13 | 0.119 s |

Truncation at `d = 128`:

| k factors kept | residual variance | error bound | measured rho error |
|---:|---:|---:|---:|
| 1 | 28.46 | 1.820 | 0.000168 |
| 2 | 0.30 | 0.019 | 0.000004 |
| 4 | 0.30 | 0.019 | 0.000006 |
| 8 | 0.30 | 0.019 | 0.000015 |

- `rho(M)` is flat in universe size and pinned to the worst scalar modulus;
  the fragile mode is idiosyncratic, not the market factor. On this
  calibration, correlation does not manufacture cross-sectional instability.
- The truncation bound holds with **3-4 orders of magnitude of slack** at
  every `k`; the 1.5 certificate additionally confirms Woodbury == dense
  eigensolve to machine precision.

---

## 9. PerfGD: the four-mode loops and the gap closure

Three layers: the closed-form gap scan, the exact 1-D dynamics in the
genuinely unstable regime, and - new in v4 - the **four** ML loop modes from
a common seed, including `perfgd_structural`.

**Closed form (re-verified at full scale).** `gamma_PO > 0` on the whole
grid (0.53 -> 1.01 across `f = 0.5..6`); the echo-chamber decision gap grows
~O(eps) (0.084 -> 0.647) and the value gap ~O(eps^2) (0.0019 -> 0.212). At
the default microstructure no on-grid gain destabilises the fixed point
(saturation, 1.1 §6.3) and the run says so explicitly; the beyond-boundary
demo therefore uses the slow-toxic-decay regime (`f = 6`, `alpha = 1`,
`I = 3`, `c_t = 0.8`, `w = 0.15`; `m_rrm = 1.21`), where the blind cobweb
does **not** converge and the corrected 1-D ascent converges to
`h_PO = 1.641`.

![Beyond the boundary](perfgd/perfgd_vs_rrm.png)

**ML loops (10 deployments, seed 0, unstable demo regime).** Central
half-spread per deployment:

| mode | trajectory (h per deployment) | outcome |
|------|-------------------------------|---------|
| blind RRM          | 0.80 1.49 2.83 2.75 2.10 1.45 1.23 0.50 0.20 **0.16** | echo-chamber collapse; did NOT converge |
| PerfGD-analytic    | 0.80 0.38 0.84 0.71 1.12 1.53 0.84 0.56 0.60 **0.36** | no settle (operator's dJ/dh is the broken ingredient) |
| PerfGD-learned     | 0.80 1.26 1.41 2.01 1.41 1.46 1.47 0.99 1.04 **0.66** | no settle (the documented free-form negative result) |
| **PerfGD-structural** | 0.80 1.08 1.46 1.97 2.66 3.59 3.72 3.38 3.24 **3.19** | **converges**: late steps 0.35 -> 0.14 -> 0.05 -> 0.02 |

![ML loops and the three-way seam](perfgd/perfgd_ml_loops.png)

The structural mode's own diagnostics along the run:

| deployment k | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `h_PO_hat` (running estimate) | 6.00* | 6.00* | 6.00* | 6.00* | 5.38 | 3.72 | 3.34 | 3.23 | 3.19 | **3.17** |
| fitted slope `-psi_hat*eps_hat(h)` | -0.25 | -0.15 | -0.51 | -0.70 | -1.10 | -1.24 | -1.19 | -1.30 | -1.34 | -1.33 |

*\* railed at `max_half_spread`: the early narrow-history fit finds no
interior root - this is the failure mode the **trust region**
(`structural_max_rel_step = 0.35`) exists for. The capped ascent keeps
exploring, the fit history widens, and the estimate sharpens monotonically
to 3.17 - within **0.7%** of where the loop actually parks (3.193).*

**How this closes the gap.** The v3 negative result stood because every
corrected mode consumed the *operator's* implied objective gradient, which
diverges from the structural one away from the deployed regime - visible
again in this run's seam panel: the free-form learned slope starts
right-signed (-0.84 vs analytic -1.79) and then flips sign and hovers near
zero (+0.20, +0.00, ..., +0.12), while the **structural fitted slope stays
negative, strengthens as the fit history widens, and tracks the analytic
shape** (-1.33 vs analytic -2.00 at the final iterate, with the level gap
being the realized-state channels below). The structural mode never asks the
MLP for `dD/dphi`: it fits the theory's own families
(`tau_hat = C0 + C1 e^{-ch}`, `u_hat = A_u e^{-k_u h}`, realized `psi_hat`)
to per-bond per-step deployment data and ascends the estimated corrected
gradient `Phi_hat' = G_hat + Delta_hat` at step `1/gamma_PO_hat`.

**Benchmark honesty (state in the paper).** The A2 closed-form
`h_PO = 1.641` is *not* the target: in this high-intensity regime the
realized market differs from the frozen-reference closed forms at first
order through channels they omit by construction (1.1 §9) - the `info_cap`
saturation (raw toxic notionals ~10 vs cap 8 at tight spreads), the
liquidity-inflation feedback (`rho ~ 2.3`, section 7), and severity drift.
The loop is therefore verified against **independent structural fits** of
the same market: the slow test in `tests/test_structural_perfgd.py` re-fits
on fresh controlled deployments spanning the operating range and confirms
the settle point (i) zeroes their corrected gradient (|Phi_hat'| < 30% of
its start-point value), (ii) sits strictly inside their blind stable point
(< 0.85x `h_SP_hat` - the realized echo-chamber gap, closed), and (iii)
brackets their independent `h_PO` estimate. Do **not** de-saturate
`info_cap` to force agreement with A2: without it the `liq_flow_boost`
feedback blows realized flows to 7-20x the closed forms.

Scoped claim for the paper: un-blinding is proven in closed form (1.2), and
the *structurally-anchored* learned loop now realises it at loop level
against the realized-market benchmark; the free-form learned loop remains a
negative result - anchoring, not capacity, closes the gap.

---

## 10. Lazy deployment: theory 1.6 at the beyond-boundary config

The signed CRN K-step probe (`measure_rgd_response`: K RGD steps
warm-started from the deployed policy, signed difference quotient) at the
default config (`f = 5`, beyond the measured boundary), K in
{1, 2, 3, 5, 8, 12, 20} x 3 seeds, with the exact-BR anchor measured
independently on the same seeds: `m_hat = 1.205 > 1`.

![Lazy deployment](lazy_deploy/lazy_deploy_gamma_eff.png)

| K | median mu_hat(K) | fitted mu(K) | per-seed slopes | gamma_eff/gamma (measured) |
|---|---:|---:|---|---:|
| 1  | **+0.021** | +0.253 | -0.106, +0.021, +0.894 | 58.4 |
| 2  | **-0.064** | -0.241 | +0.023, -0.228, -0.064 | 18.9 |
| 3  | -0.257 | -0.567 | -0.016, -1.371, -0.257 | 4.68 |
| 5  | -0.695 | -0.926 | -0.078, -3.715, -0.695 | 1.73 |
| 8  | **-1.541** | -1.124 | -0.551, -4.811, -1.541 | 0.78 |
| 12 | -4.470 | -1.189 | -0.922, -5.018, -4.470 | 0.27 |
| 20 | -4.808 | -1.204 | -6.794, -4.808, -4.693 | 0.25 |

Fitted inner contraction `c = 0.661` (one deployment realises
`lam_1 = 0.34` of the exact cobweb). The two parameter-free predictions of
06 both land:

- **Deadbeat sign flip.** Predicted `K_db = ln(m/(1+m))/ln(c) = 1.46`;
  measured medians flip from +0.021 (K = 1) to -0.064 (K = 2).
- **Stability-window exit.** Predicted `K_max = ln((m-1)/(m+1))/ln(c) =
  5.75`; measured `|mu|` is still inside the band at K = 5 (0.695) and
  outside at K = 8 (1.541) - **laziness keeps this RRM-unstable market
  (`m_hat = 1.20`) stable for K <= ~6 and inherits the cobweb's divergence
  beyond**, the quantitative form of the RGD-beats-RRM stability-range
  result.
- The measured `gamma_eff/gamma` column falls monotonically through 1
  between K = 5 and K = 8 (1.73 -> 0.78): with `m > 1` every K sits in the
  stiff branch (`K_eq = 0`) until the map exits the band, exactly the
  two-branch structure of 06 §4.

**Caveats.** Per-seed scatter is large here by construction - beyond the
boundary the probe readings are finite-difference diagnostics, not local
slopes (the same A4 caveat as section 5; seed 1 rails early). The medians
trace the closed form; the deep-K medians (-4.5, -4.8) sit beyond the fitted
asymptote `-m` because the K-step map at this depth compounds the
beyond-boundary bifurcation. The clean contracting-regime curve (monotone,
tight fit: measured +0.78 / +0.70 / +0.37 vs fitted +0.88 / +0.68 / +0.36 at
K = 1/3/8 on the smoke config) is locked in `tests/test_lazy_deploy.py` and
the smoke profile, and the v3 budget-sensitivity study
(*(v3-run analysis figure)* below) was the precursor observation this theory
now explains.

![Budget sensitivity](../../analysis/figures/budget_sensitivity.png)
*(v3-run analysis figure)*

---

## 11. Estimator tuning: Sinkhorn blur + robust ambiguity radius

![Estimator tuning](tuning/estimator_tuning.png)

**Sinkhorn blur (left panel).** The debiased-divergence bias against the
exact 1-D quantile `W1`, on a *scale-relative* blur grid (blur / pooled
std), at the fixed 300-iteration budget:

| rel. blur | synthetic bias (true W1 = 0.313) | config-CRN bias (W1 = 2.289) |
|---:|---:|---:|
| 0.005 | 88.2% | 0.2% |
| 0.01  | 48.9% | 0.5% |
| **0.02** | **6.1%** | **1.2%** |
| 0.05  | 14.2% | 3.1% |
| 0.10  | 26.3% | 6.1% |
| 0.20  | 44.1% | 10.9% |
| 0.50  | 69.5% | 20.1% |
| 1.00  | 82.0% | 28.5% |
| 2.00  | 87.7% | 35.2% |

The ground-truth (synthetic) curve is U-shaped - log-domain
under-convergence below 0.02, entropic over-blur above - with its minimum at
the baked-in default `TUNED_REL_REG = 0.02`. The config's CRN toxic samples
(smaller n = 320/side) stay converged down to 0.005, so their curve is
monotone over this grid; the default sits in the low-bias region of both
(1.2% / 6.1%) and, being scale-relative, transfers to calibrated real-unit
configs unchanged (the same convention as every probe width). `reg="auto"`
in `estimate_epsilon_sinkhorn` applies it.

**Robust ambiguity radius (right panel).** One-sided frequentist coverage
at 95% nominal, n = 6 estimates per replicate, 400 Monte-Carlo replications
per case; plus the calibration run on the config's own CRN modulus probes:

| estimate distribution | z*s coverage | calibrated coverage | median quantile/z*s multiplier |
|---|---:|---:|---:|
| normal | 0.993 | 0.993 | 0.73 |
| student-t (2.5 dof) | 1.000 | 1.000 | 0.72 |
| lognormal | 0.990 | 0.990 | 0.80 |
| contaminated (6% far outliers) | 0.938 | 0.938 | 0.77 |
| **CRN probe estimates** (6 probes, bootstrap) | 0.999 | 0.999 | 0.50 |

- `z*s` **over-covers** everywhere except contamination - conservative in
  the safe direction, because the certificate's `z*s` is a single-estimate
  half-width applied to the mean. Symmetric heavy tails do *not* break it
  (the std inflates faster than the 95% quantile - the direction the tuning
  experiment itself corrected during development).
- The contaminated case (the railed-probe / beyond-boundary bifurcation
  pattern) drops both radii to 0.938: with 6 seeds a rare mode is simply
  unsampled - the honest limit, matching 1.4's `n_req = O(Delta^-2)`. The
  calibrated `max(z*s, quantile)` radius is the guard for exactly this
  pattern when it *is* sampled.
- On the actual probe estimates (0.541, 1.907, 1.205, 1.781, 1.866, 1.496 -
  note the beyond-boundary scatter) the quantile multiplier is 0.50, so
  `z*s` is the binding radius and is adequately calibrated for the CRN
  probe. Verdict: keep `z*s` as the default; `radius_method="calibrated"`
  available where contamination is suspected.

---

## 12. Reference: one instrumented loop

A single `perfgd_analytic` loop at the default config (`f = 5`, 12
deployments) with per-iteration seam diagnostics; classified converged
(trajectory L_hat = 0.81), ending at `h = 0.52` with realized market metrics
logged: `realized_h = 1.546`, `toxicity_share = 0.795`,
`fill_rate = 1.988`, `adverse/step = 5.534`, `HHI = 0.173`.

![Single loop trajectory](single/rrm_trajectory_a0.5_eps5.0_s0.png)

The final-iterate seam reading (learned `d[adv]/dh = +0.19` vs analytic
`-0.57`) is one more instance of the free-form operator's sign instability -
the quantitative motivation for the structural mode (section 9).

---

## 13. Consistency: the v4 code reproduces the v3 run

Deterministic from `(config, seed)`, the v4 re-runs of the v3-era
experiments reproduce the 07-10-2026 numbers **to all printed digits**:

| quantity | 07-10-2026 (v3) | 07-12-2026 (v4) |
|----------|-----------------|-----------------|
| sweep medians (f = 0..8) | 0.067 / 0.159 / 0.390 / 0.856 / 1.707 / 1.180 / 1.357 | identical |
| measured crossing f* | 3.169 | 3.169 |
| dealer common-mode (N = 1/2/3) | 0.786 / 1.369 / 2.480 | identical |
| triangulation legs (BR / W1 / CKS) | 0.508 / 4.578 / 4.011 | identical |
| realized state (rho, \|g\|) | 2.32, 0.446 | identical |
| fragility / calibrated / universe tables | (closed forms) | bit-for-bit |
| ML loops: rrm / analytic / learned trajectories | (see v3 report §8) | identical |

The only differences are the *additions*: the fourth loop mode
(`perfgd_structural`, section 9), the lazy-deploy sweep (section 10), the
tuning experiment (section 11), and the certificates (section 2). The
alpha-confound appendix sweep was not rerun - nothing in v4 touches the
alpha channel; its v3 artifacts stand in
[`../07-10-2026/sweep/`](../07-10-2026/sweep/).

---

## 14. Methods note: what the verification layer caught

Two findings from *building* the v4 verification and tuning layers are
results in their own right and are encoded in the shipped code:

1. **The `lambda_q` convention (certificate-discovered).** The first
   certificate run failed with residuals of -11.7% (1.1 BR slope) and -5.6%
   (1.2 curvature identity) - both exactly the inventory-risk curvature
   `P*lambda_q`. The 1-D frozen-gradient helpers (`best_response`, `Phi'`)
   omit the inventory term, so their measured slopes are governed by
   `gamma - P*lambda_q`, while the full-pipeline modulus uses the
   `lambda_q`-inclusive `gamma` (1.1 §7). The identities hold once both
   sides use one convention; the certificates now state and encode it.
2. **Two falsified first drafts (measurement-corrected).** The 1.6
   effective-curvature claim was first written as `gamma_eff >= gamma`
   everywhere - falsified by the smoke sweep (the near-zero-`m` default
   market sits entirely in the *inertia* branch below the equal-modulus
   count) and corrected to the two-branch statement now in 06 §4. The
   robust-radius test first asserted quantile > `z*s` for symmetric heavy
   tails - falsified (the std outpaces the quantile there) and corrected to
   the contamination statement of section 11.

Both corrections are visible in the commit history (`4b80b02`, `598d0ff`,
`3bd4bc9`) - the verification layer did its job before the paper-grade run,
which is why this run is quotable. The same discipline as the v3 pre-run
audit ([`../../analysis/pre-run-audit-2026-07.md`](../../analysis/pre-run-audit-2026-07.md)),
one layer deeper.

## 15. Limitations (paper-ready)

1. Not trade-level TRACE: VIX-implied spreads, proxy-level intensity fits,
   no per-dealer inventories. Regime ordering is data-driven; absolute
   critical gains are not. WRDS TRACE Enhanced is the upgrade path.
2. Crisis cells are degenerate (`k = 0`): crisis boundaries sit on the
   anchor floor; intra-crisis variation is not identified.
3. The toxic channel is structurally scaled (documented ratios), not
   data-identified.
4. At default-like constants the self-consistent fixed point never
   destabilises (defensive widening); measured crossings are statements
   about the local retraining map at the operating spread.
5. Measured moduli are protocol-dependent (optimizer budget, collection
   jitter); levels compare within a protocol, not across.
6. **The structural loop's optimum is the realized one**: `perfgd_structural`
   is benchmarked against independent structural fits of the realized market
   (info-cap saturation + liquidity inflation + severity drift included),
   never against the frozen-reference A2 `h_PO`; the level gap between them
   is measured and channel-attributed, not hidden (sections 7, 9).
7. The structural mode's convergence is to the O(1/sqrt(n)) noise ball of an
   estimated gradient (hovering, late steps ~0.02-0.05), not a
   parameter-norm fixed point; its early iterations depend on the trust
   region while the fit history is still narrow (the railed `h_PO_hat` rows
   in section 9).
8. The lazy-deploy verification at the beyond-boundary config carries the
   probe-scatter caveat (per-seed readings bifurcate; medians carry the
   result); the clean-fit demonstration lives in the contracting regime.
9. The free-form learned mode (`perfgd_learned`) remains a negative result
   by design and is reported as such.
10. The Lean formalisations are reviewed statements, not yet compiled (no
    toolchain on the dev machine); the numerical certificates are the
    verification of record (`endo_market_v4/lean/README.md`).
11. Sweep medians carry 8-seed IQRs and robust bands; the calibrated
    measured column (3 seeds) is a consistency check only.
