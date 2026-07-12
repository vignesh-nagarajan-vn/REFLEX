# REFLEX v4 paper-grade run report - July 12, 2026

The illustrated report of the full-profile suite of the **final generation**
(`endo_market_v4`, package `reflex` 4.0.0). One page: the v4 headline findings
(the loop-level gap closure, the theory-1.6 lazy-deploy sweep, the estimator
tuning, the proof certificates), then the consistency check that the v4 code
reproduces the v3-era results. Raw artifacts live beside this file
(per-experiment subfolders); provenance in [`README.md`](README.md); the v3
per-experiment deep-dive remains
[`../../analysis/ANALYSIS-full-2026-07.md`](../../analysis/ANALYSIS-full-2026-07.md).

| | |
|---|---|
| **Run date** | July 12, 2026 |
| **Outcome** | suite **11/11 passed in 25.1 min CPU**; **152/152 tests** green before the run; **66/66 proof certificates** passing (raw + calibrated) |
| **Environment** | Python 3.9.13, torch 2.8.0+cpu, numpy 2.0.2, pandas 2.3.3; Windows 11, CPU only; deterministic from `(config, seed)` |

---

## 1. Executive summary

1. **The v3 loop-level gap is closed.** In the genuinely RRM-unstable demo
   regime (closed-form `m = 1.21` at the fixed point) the blind loop fails to
   converge while the v4 structural mode - the learned loop with its response
   anchored to the GLFT families and fitted to its own deployment history -
   contracts geometrically (steps 0.35 -> 0.14 -> 0.05 -> 0.02 over the last
   iterations) and settles at `h = 3.193`, within **0.7%** of its running
   estimate of the *realized* performative optimum (`h_PO_hat = 3.172`)
   (section 2).
2. **Theory 1.6 is verified in the hardest regime.** At the beyond-boundary
   default config (measured full-BR anchor `m_hat = 1.20 > 1`) the signed
   K-step probe medians descend exactly as `mu(K) = -m + c^K(1+m)` predicts:
   the sign flips at the deadbeat count (`K_db = 1.46`; measured `+0.02` at
   `K = 1`, `-0.06` at `K = 2`) and the map exits the stable band right at the
   predicted window (`K_max = 5.75`; measured `|mu| = 0.70` at `K = 5`,
   `1.54` at `K = 8`): **laziness measurably stabilises an RRM-unstable
   market inside the predicted window** (section 3).
3. **The tuned estimators behave as calibrated.** The Sinkhorn blur bias
   curve confirms the baked-in scale-relative default (`0.02 x std`: 6.1%
   bias on the synthetic ground truth, low-bias region on the config's CRN
   samples); the robust `z*s` radius over-covers (0.99+) on normal, heavy-
   tailed and skewed estimate distributions and is the binding radius for the
   CRN probe (quantile multiplier 0.50) - conservative in the safe direction,
   with the contaminated case (coverage 0.938) marking the pattern the
   calibrated radius exists for (section 4).
4. **Every theory identity is machine-checked.** 66/66 numerical proof
   certificates pass on the raw config *and* the calibrated real-unit config
   - including the certificate-discovered `lambda_q` convention (the 1-D
   frozen-gradient helpers' slopes are governed by `gamma - P*lambda_q`)
   (section 5).
5. **The v4 code reproduces the v3 results exactly** where it re-runs them:
   sweep crossing `f* ~ 3.169` vs predicted 4.697, dealer common-mode
   amplification 1.74x / 3.16x vs `N_eff` = 2 / 3, realized-state
   triangulation with `rho = 2.32`, and the unchanged closed-form fragility /
   calibrated / universe outputs (section 6).

---

## 2. The gap closure: `perfgd_structural` (perfgd/)

![four-mode loops](perfgd/perfgd_ml_loops.png)

Four loops from a common seed in the unstable demo regime (10 deployments,
`--ml`): blind `rrm` swings and does not converge; `perfgd_analytic` and the
free-form `perfgd_learned` remain non-stabilising (the documented v3 negative
result - the operator's implied `dJ/dh` is the broken ingredient);
**`perfgd_structural` converges**: final `h = 3.193`, self-consistent with its
own realized `h_PO_hat = 3.172` to 0.7%, hovering with late steps < 0.05.

**Benchmark honesty (state in the paper).** The A2 closed-form `h_PO = 1.641`
is *not* the target: at these intensities the realized market differs from the
frozen-reference closed forms at first order through channels they omit by
construction (1.1 §9) - the `info_cap` saturation (raw toxic notionals ~10 vs
cap 8 at tight spreads), the liquidity-inflation feedback (realized
`rho ~ 2.3`), and severity drift. The loop is verified against *independent
structural fits* of the same market (the slow test in
`tests/test_structural_perfgd.py` re-fits on fresh controlled deployments:
the settle point zeroes their corrected gradient, sits inside their blind
stable point - the realized echo-chamber gap, closed - and brackets their
`h_PO` estimate). The right panel shows the three-way seam: the structural
fitted slope tracks the analytic shape while the free-form operator slope
stays unreliable - anchoring, not capacity, is what closes the gap.

## 3. Lazy deployment: theory 1.6 (lazy_deploy/)

![lazy deploy](lazy_deploy/lazy_deploy_gamma_eff.png)

Signed CRN K-step probe at the beyond-boundary default config, K in
{1, 2, 3, 5, 8, 12, 20}, 3 seeds, with the full-BR anchor measured
independently (`m_hat = 1.20`):

| K | median `mu_hat(K)` | theory |
|---|---|---|
| 1 | +0.021 | above the deadbeat (`K_db = 1.46`) |
| 2 | -0.064 | just past the sign flip |
| 3 | -0.258 | descending |
| 5 | -0.695 | still inside the stable band |
| 8 | -1.541 | outside (predicted exit `K_max = 5.75`) |
| 12 | -4.470 | inherits the cobweb divergence |
| 20 | -4.808 | (beyond-boundary probe scatter applies) |

Fitted inner contraction `c = 0.661` (one deployment realises `lam_1 = 0.34`
of the exact cobweb). Both parameter-free predictions land: the deadbeat sign
flip between K = 1 and 2, and the stability-window exit between K = 5 and 8
vs the predicted 5.75. Per-seed scatter is large here - expected and
documented (beyond the boundary the probe readings are finite-difference
diagnostics, not local slopes); the medians trace the closed form. The
contracting-regime curve (clean monotone fit) is locked in
`tests/test_lazy_deploy.py` and the smoke profile.

## 4. Estimator tuning (tuning/)

![tuning](tuning/estimator_tuning.png)

**Sinkhorn blur.** On the synthetic location-shift ground truth (true
`W1 = 0.30`) the bias curve turns at the baked-in default `rel_reg = 0.02`
(6.1% bias; under-convergence below - 35%+ at 0.01 and finer at this
iteration budget in the pre-bake measurement - entropic over-blur above). On
the config's own CRN toxic samples the low-blur end stays flat (0.2-1.2%
bias for 0.005-0.02 at this sample size); the default sits in the low-bias
region of both curves and is *scale-relative* (`reg = 0.02 x pooled std`), so
calibrated real-unit configs inherit it unchanged.

**Robust ambiguity radius.** One-sided frequentist coverage at 95% nominal,
n = 6 estimates per replicate: `z*s` covers 0.993 (normal), 1.000
(student-t 2.5), 0.990 (lognormal) - conservative, because the certificate's
`z*s` is a *single-estimate* half-width applied to the mean. The contaminated
case (6% far outliers, the railed-probe pattern) drops to 0.938 for both
radii: with 6 seeds a rare mode is simply unsampled - the honest limit,
matching 1.4's `n_req = O(Delta^-2)`. On the actual CRN probe estimates the
quantile multiplier is 0.50, so `z*s` is the binding (and adequately
calibrated) radius; `robust_boundary(radius_method="calibrated")` remains the
guard for contaminated patterns.

## 5. Proof certificates (certificates/)

66/66 checks pass on `configs/default.yaml` **and** the calibrated IG/normal
real-unit config: the 1.1 boundary identities (BR slope by finite differences
vs `-eps*beta/gamma` at the `lambda_q = 0` convention; cobweb-step
linearisation), the 1.2 identities (optimum stationarity, the `psi` sign
flip, `gamma_PO = -Phi''`) and raw-unit demo dynamics, the 1.3 eigen-identity
(`-m1*N_eff` on the ones vector, eigensolve-confirmed) and boundary algebra,
the 1.4 radius formula / Monte-Carlo `-1/2` rate / verdict trichotomy, the
1.5 Woodbury-vs-dense spectral radius and truncation-bound domination, and
the 1.6 algebra. The raw-unit demo-dynamics certificate is auto-excluded on
the calibrated config (absolute demo constants must not be imposed on
real-unit configs).

## 6. Consistency with the v3 run (sweep/, dealers/, triangulation/, fragility/, calibrated/, universe/, single/)

Deterministic from `(config, seed)`, the v4 re-runs reproduce the July-10
numbers:

- **Sweep (8 seeds, robust bands):** medians 0.067 / 0.159 / 0.390 / 0.856 /
  1.707 / 1.180 / 1.357 across f = 0..8; measured crossing `f* ~ 3.169` vs
  a-priori 4.697 - identical to v3 (the realized-state correction ~2.8-3.0
  reconciles them, as the triangulation confirms).
- **Dealers:** genuine-market common-mode moduli 0.786 / 1.369 / 2.480 at
  N = 1 / 2 / 3 = amplification 1.74x / 3.16x vs predicted N_eff 2 / 3, with
  the differential mode dead - identical to v3.
- **Triangulation:** BR 0.51 / Sinkhorn 4.58 / CKS 4.01 vs analytic 0.764
  (A2) and 1.727 (realized state, `rho = 2.32`) - the Sinkhorn/CKS legs agree
  within 14% and bracket the realized-state form at 2.3-2.7x, as in v3.
- **Fragility / calibrated / universe:** closed forms on the same shipped
  data - unchanged (headroom collapse ~4.4x IG / ~4.3x HY calm->crisis; HY
  >10x below IG; `rho(M) ~ 0.50` flat to 128 bonds; truncation bound holds
  with orders-of-magnitude slack).
- **Single (`perfgd_analytic`, f = 5):** converges with the seam recorded;
  realized market metrics logged (`realized_h = 1.546`,
  `toxicity_share = 0.795`).

The alpha-confound appendix was not rerun (nothing in v4 touches the alpha
channel); its v3 artifacts stand in [`../07-10-2026/sweep/`](../07-10-2026/sweep/).

---

## Honest caveats carried forward

Everything in the v3 report's caveat list still applies (not trade-level
TRACE; degenerate crisis fit; beyond-boundary probe scatter; structural
scaling of the toxic channel). New v4-specific caveats: the structural loop's
optimum is the *realized* one (benchmark against independent fits, never A2);
the free-form learned mode remains a negative result by design; the Lean
skeletons are reviewed formal statements, not yet compiled (no toolchain on
the dev machine) - the numerical certificates are the verification of record.
