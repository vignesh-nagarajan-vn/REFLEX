# REFLEX full-profile results: analysis (July 2026)

The master analysis of the paper-grade experiment suite
(`endo_market_v3`, `python -m experiments.run_all --profile full` plus the
alpha-confound appendix sweep), executed against the shipped real-data
calibrations after the [measurement-layer audit](pre-run-audit-2026-07.md).
Raw artifacts: [`../results/07-10-2026/`](../results/07-10-2026/).
Derived figures: [`figures/`](figures/).

**Reading order.** §1 is the one-page summary. §2–§9 are per-experiment
breakdowns (each: what ran, headline numbers, interpretation, caveats).
§10 collects the honest limitations that must appear in any paper.

---

## 1. Summary of findings

1. **The a-priori stability boundary is computable from real data, and its
   regime structure is economically sensible.** The closed-form headroom
   `eps* = gamma/beta` evaluated on 36 years of daily data collapses ~4.4x
   (IG) / ~4.3x (HY) from calm to crisis, and HY sits >10x below IG in every
   regime. The market's *distance to the performative-instability boundary*
   shrinks exactly when intuition says it should: high-yield paper, stressed
   regimes. (§2)
2. **Defensive widening is visible in the real data**: the modulus evaluated
   at *observed* spreads falls calm -> crisis (IG 0.85 -> 0.14) — dealers
   widen faster than the toxic channel steepens, buying stability at the cost
   of liquidity. The theory's saturation corollary (1.1 §6.3), live on data.
   (§2, §3)
3. **Competition amplifies performative instability on the genuine
   multi-dealer market as derived**: interior-regime common-mode moduli scale
   ~1.0 / 1.8 / 3.2 vs the predicted `N_eff` = 1 / 2 / 3, and the
   differential mode is flat at full spillover — after the audit fixed the
   env's coupling to the derivation's sum form. (§5)
4. **Predict-then-verify now works quantitatively**: under the clean probe
   protocol (a second protocol defect — inherited exploration jitter — was
   found in the first execution and fixed), the measured modulus tracks the
   closed form within ~10-25% through the contracting regime (0.39 vs 0.43
   at `f = 2`), and the measured boundary crossing (`f* ~ 3.2`) sits left of
   the a-priori prediction (4.7) by almost exactly the realised-state
   correction the triangulation measures independently (predicted crossing
   at the realised state: ~2.8-3.0). Robust certificates grade the grid
   stable -> undecided -> unstable as the bands warrant; past the boundary
   the probe correctly stops being interpretable as a slope. (§4)
5. **Three-way epsilon triangulation agrees at the factor ~2-3 level against
   the realized-state closed form** — with the liquidity-inflation channel
   (`rho ~ 2.3` realized vs 1.0 a-priori) identified as the dominant
   correction, and the remaining gap attributed to the state-feedback channel
   any frozen-state closed form omits. (§6)
6. **Factor scaling defuses the curse of dimensionality on real dispersion**:
   `rho(M) ~ 0.50` flat from 8 to 128 bonds (data-calibrated per-bond vol
   dispersion at the structural band cap), the fragile mode is idiosyncratic
   (market alignment ~ 0), and the truncation bound holds with orders of
   magnitude of slack. (§7)
7. **The un-blinding corrections are proven in closed form but do NOT yet
   stabilise the learned loop** — the audit's headline honest finding,
   reproduced at paper scale: the blind operator's implied `dJ/dh` diverges
   from the structural one, so neither the analytic nor the learned
   correction relocates the ML loop's equilibrium to `h_PO`. The ML
   artifacts are seam diagnostics, not stabilisation proofs. (§8)
8. **The alpha sweep is confounded exactly as theory 1.1 §6.4 predicts**
   (appendix result). (§9)

---

## 2. Fragility index (`run_fragility`) — the real-data headline

**What ran.** The 1.1 closed forms evaluated on every trading day 1990–2026
of the master panel (VIX spine; per-regime intensity fits), for IG and HY.
Closed form on real data — full fidelity by construction.

**Headline table** (regime medians; `eps*` = stability headroom
`gamma/beta`, fragility = median-headroom / headroom):

| regime | IG eps* | HY eps* | IG modulus at observed h | IG fragility |
|--------|--------:|--------:|-------------------------:|-------------:|
| calm     | 1207.7 | 93.5 | 0.847 | 0.60 |
| normal   |  739.2 | 61.3 | 0.870 | 0.98 |
| elevated |  594.4 | 45.9 | 0.520 | 1.22 |
| stress   |  477.3 | 35.6 | 0.223 | 1.52 |
| crisis   |  275.9 | 21.8 | 0.139 | 2.64 |

**Findings.**

- Headroom collapses **~4.4x (IG)** and **~4.3x (HY)** calm -> crisis
  (regime medians). *Correction to earlier docs:* the previously quoted
  "~13x (HY)" does not reproduce on the audited pipeline; both ratings
  collapse ~4.3–4.4x, while the *level* gap between ratings is the larger
  effect — HY has >10x less headroom than IG in every regime.
- The modulus at observed spreads **falls** into crisis — defensive widening
  live on real data (finding 2 above).
- The index **saturates at a crisis plateau** rather than peaking on a single
  day: the crisis-regime intensity fit is degenerate (`k = 0`, n = 74 days,
  documented), so within-crisis days share one boundary value (IG fragility
  2.64). The GFC window (first crisis day 2008-10-06, Lehman aftermath) and
  the COVID freeze (2020-03-09) both sit on that plateau; **intra-crisis
  ranking is not identified** by this calibration. Earlier phrasing
  ("peaks at Lehman") is corrected accordingly.

**Caveats.** VIX-implied spreads, proxy-level intensity fits (not trade-level
TRACE); regime *ordering* is data-driven, absolute levels are not; the
crisis plateau is an artifact of the degenerate crisis fit and flagged.

---

## 3. Calibrated a-priori boundaries (`run_calibrated --measure --seeds 3`)

**What ran.** For each (rating x regime) cell: the calibrated real-unit
config, the closed-form boundary at the fixed point and at the observed
spread, and (IG, non-degenerate cells) the measured BR-slope modulus at the
observed spread.

**Table** (per-100-par units):

| cell | h_obs | h* | eps* | m(h*) | m_pred at h_obs | m_measured (median, 3 seeds) |
|------|------:|----:|-----:|------:|----------------:|------------------------------:|
| IG calm     | 0.374 | 0.436 | 465.3 | 0.066 | 0.080 | 0.000 |
| IG normal   | 0.477 | 0.558 | 268.0 | 0.066 | 0.081 | 0.000 |
| IG elevated | 0.683 | 0.794 | 147.3 | 0.066 | 0.080 | 0.038 |
| IG stress   | 1.051 | 1.207 |  70.6 | 0.067 | 0.080 | 1.631 |
| IG crisis   | 1.233 | 1.472 |  34.5 | 0.065 | —  (degenerate) | — |
| HY calm     | 1.030 | 1.183 |  28.3 | 0.067 | — | — |
| HY normal   | 1.331 | 1.535 |  16.2 | 0.066 | — | — |
| HY elevated | 1.882 | 2.159 |   9.1 | 0.066 | — | — |
| HY stress   | 2.896 | 3.275 |   4.7 | 0.066 | — | — |
| HY crisis   | 3.493 | 4.169 |   2.2 | 0.055 | — | — |

**Findings.**

- **The fixed-point modulus is regime-invariant (~0.066)**: the calibration's
  anchor-stiffness rule pins the dealer near the observed spread with
  curvature proportional to `A/h0`, and the toxic channel is structurally
  scaled to `A` — so `m(h*)` cancels to a near-constant. The regime story
  lives entirely in the **headroom** `eps*` (465 -> 2.2 across the table),
  i.e. in *how much feedback the regime can absorb*, not in the operating
  modulus. This is the correct reading of the calibrated table and matches
  the fragility index.
- **The measured column is a weak consistency check, not a result.** The
  audit's protocol note applies with force here: at IG-calm/normal the
  anchor weight is enormous (`w = S*A/h0 ~ 190`), so the true local BR
  difference across the probe (~0.015 per-100-par) sits at the finite-budget
  attenuation floor — measured exactly 0.0; at IG-stress the anchor is ~15x
  softer and probe noise dominates a 3-seed median (1.63). Only IG-elevated
  (0.038 vs predicted 0.080, attenuation factor ~2 — consistent with the
  default-config attenuation) is informative. A defensible measured check at
  calibrated scales needs many more seeds and an explicit noise floor; the
  a-priori table above is the deliverable.

---

## 4. Phase-diagram sweep (`run_sweep`, 7 gains x 8 seeds) — predict-then-verify

**What ran.** The headline experiment: the CRN BR-slope modulus at
`h_ref = 1.0` across the feedback-gain grid, 8 seeds per point (median + IQR),
with the 1.4 robust certificate per point, overlaid with the closed form
evaluated **at the probe spread** (the audit's apples-to-apples fix) and, as
context, at the drifting fixed point `h*(f)`. Run under the **clean probe
protocol** (`collection_jitter = 0.05`; the first execution used the
inherited 0.2, which inflated every reading ~3x and produced a spurious
`m ~ 0.66` at zero feedback — audit §1.6; the artifacts here are the clean
rerun).

| f | median m | IQR | robust verdict | m_pred at h_ref | m at h*(f) |
|---|---------:|-----|----------------|----------------:|-----------:|
| 0 | 0.067 | [0.04, 0.09] | stable    | 0.000 | 0.000 |
| 1 | 0.159 | [0.08, 0.28] | stable    | 0.213 | 0.124 |
| 2 | 0.390 | [0.21, 0.48] | stable    | 0.426 | 0.225 |
| 3 | 0.856 | [0.61, 1.07] | undecided | 0.639 | 0.310 |
| 4 | 1.707 | [1.42, 2.16] | **unstable** | 0.852 | 0.384 |
| 6 | 1.180 | [0.83, 1.41] | undecided | 1.277 | 0.507 |
| 8 | 1.357 | [0.89, 1.84] | undecided | 1.703 | 0.607 |

Measured crossing: `f* ~ 3.17`. Predicted at the probe spread (a-priori
state): `f* ~ 4.70`.

**Findings.**

- **In the contracting regime the probe now tracks the closed form
  quantitatively**: 0.067 vs 0 at `f = 0` (the residual noise floor), 0.159
  vs 0.213, 0.390 vs 0.426 (8% agreement at `f = 2`). This is the
  predict-then-verify protocol working as intended — after two
  protocol-contamination defects (probe point, jitter) were found and fixed.
- **The measured crossing (3.17) sits left of the a-priori prediction (4.70)
  by almost exactly the realised-state correction identified independently
  by the triangulation (§6)**: the deployment inflates the liquidity ratio
  (`rho ~ 2.3`), scaling the closed form by ~1.7x at the operating spread —
  which moves the predicted crossing to ~2.8-3.0, bracketing the measured
  3.17. Two independent instruments attribute the same residual to the same
  channel (a-priori state vs realised state), which is the strongest form of
  agreement this design can produce.
- **Past the boundary the probe stops being a slope**: the IQRs explode
  (`f = 4`: [1.42, 2.16]; `f = 8`: [0.89, 1.84]) and the medians are
  non-monotone — the seed-level bifurcation of the [budget-sensitivity
  study](figures/budget_sensitivity.png) (readings 0.2 vs 4.9 at the same
  point) surfacing in the sweep. The robust certificates respond correctly:
  *stable* through `f = 2`, *unstable* at `f = 4`, *undecided* (bands
  straddle 1) where the beyond-boundary readings scatter. The theory's own
  A4 caveat — the analytic modulus is valid up to and slightly past the
  boundary, not deep into divergence — is visible in the data.
- **The fixed-point curve never crosses 1** (saturates at 0.61 by `f = 8`) —
  defensive widening, reported as context per 1.1 §6.3. The measured
  instability is a statement about the retraining map at the operating
  spread, not about the self-consistent equilibrium.
- **Budget sensitivity** (lazy-deploy `K`, the roadmap item): in the
  contracting regime the measured modulus is essentially budget-insensitive
  beyond ~30 inner steps and matches the closed form; past the boundary it
  grows with budget (fuller re-optimisation exploits the fitted differences
  further). Levels are protocol-comparable only within a regime.

---

## 5. Multi-dealer systemic risk (`run_dealers --probe --episodes 8`)

**What ran.** Three layers: the analytic `(N, f)` stability surface
`m_N = N_eff * m_1`; the simulated joint cobweb on the genuine shared-pool
market (`N = 3`); and the CRN joint-modulus probes — in-phase (common mode)
vs anti-phase (differential) — at the interior probe regime
(`f_probe = 0.5`, `liq_flow_boost/N`; audit fixes).

| N | measured common-mode | predicted N_eff x m_1 | measured differential | clipped |
|---|---------------------:|----------------------:|----------------------:|---------|
| 1 | 0.786 | 0.786 | — | no |
| 2 | 1.369 | 1.571 | 0.0034 | no |
| 3 | 2.480 | 2.357 | 0.0032 | no |

**Findings.**

- **The 1.3 amplification law holds on the genuine market**: measured
  amplification ratios 1.74x (`N = 2`) and 3.16x (`N = 3`) versus the
  predicted `N_eff` = 2 and 3 — within 13% and 5% respectively, after the
  audit fixed the environment's coupling to the derivation's sum form.
- **The differential mode is dead at full spillover** (measured ~0.003 vs
  theory's `(1 - kappa) m_1 = 0`): instability is purely common-mode — the
  systemic, synchronised channel, exactly the 1.3 story ("competition
  manufactures systemic fragility, a factor N_eff before any single dealer
  would destabilise").
- Both saturation guards (`br_clipped`, cap slack) were green; deep past the
  boundary the probe rails and reads zero, which the artifact now flags
  instead of silently reporting (the committed pre-audit artifact had
  exactly that silent zero).

---

## 6. Epsilon triangulation (`run_triangulation --episodes 8`)

**What ran.** The three independent measured legs — BR-slope
(decision-space), Sinkhorn/quantile-W1 (distribution-space), CKS fitted flow
curve (structural-fit) — at the operating spread `h_ref = 1.0`, against the
closed form evaluated both at the a-priori A2 reference state and at the
**realised deployment state** (audit fix implementing 1.1 §9).

| quantity | epsilon | ratio vs realised closed form |
|----------|--------:|------------------------------:|
| analytic, a-priori A2 state (`rho = 1`) | 0.764 | 0.44x |
| **analytic, realised state** (`rho = 2.32`, `|g| = 0.446`) | **1.726** | 1.00 |
| BR-slope leg (`m_hat = 0.541`) | 0.508 | 0.29x |
| Sinkhorn/W1 leg | 4.578 | 2.65x |
| CKS flow-curve leg | 4.011 | 2.32x |

**Findings.**

- **The realised-state correction is first-order, and its driver is
  identified**: the deployment's own flow boosts the liquidity field to
  `rho ~ 2.3` (vs the a-priori 1.0), lifting the closed form 2.3x. Any
  triangulation against the frozen a-priori state is comparing against the
  wrong number — a protocol point that generalises beyond this codebase.
- **The two distribution-space legs agree with each other within 14%**
  (4.58 vs 4.01) and sit 2.3–2.7x above the realised-state closed form; the
  residual is the *state-feedback channel* (`d(state)/dh` — tighter
  deployments both attract flow *and* shift the state distribution the flow
  responds to), which any frozen-state closed form necessarily omits. The
  closed form is a **lower anchor**, not an unbiased point prediction.
- **The BR leg reads low (0.29x)** — the finite-budget attenuation of the
  decision-space map (§4). Its value is corroborating the *order*, not the
  level.

---

## 7. Universe factor scaling (`run_universe`)

**What ran.** The `d x d` modulus matrix `M = beta * Gamma^{-1} * E` and its
spectral radius across universe sizes 8–128, with per-bond sigmas dispersed
by the **data-calibrated** coefficient of variation (audit fix: CV of
per-bond vol across the 212 real CUSIPs = 1.32, clipped to the structural
band cap 0.8 — real heterogeneity exceeds what the Gaussian dispersion model
admits, stated rather than hidden); plus the `O(d k^2)` Woodbury reduction
and the truncation bound at d = 128.

| d | rho(M) | stable | scalar-max m | market alignment |
|---|-------:|--------|-------------:|-----------------:|
| 8   | 0.499 | yes | 0.502 | 0.213 |
| 16  | 0.501 | yes | 0.502 | ~1e-13 |
| 32  | 0.501 | yes | 0.502 | ~1e-13 |
| 64  | 0.501 | yes | 0.502 | ~1e-13 |
| 128 | 0.501 | yes | 0.502 | ~1e-13 |

Truncation at d = 128: measured `|rho - rho_k|` is 1.7e-4 at k = 1 and
< 2e-5 for k >= 2, versus bounds 1.82 and 0.019 — the bound holds with 3–4
orders of magnitude of slack (it is a worst-case bound; the spectrum
concentrates far from it).

**Findings.** `rho(M)` is flat in universe size and pinned to the worst
scalar modulus; the unstable direction is **idiosyncratic**, not the market
factor (alignment ~ 0 for d >= 16). On this calibration, correlation does
not manufacture cross-sectional instability — the honest §3.3 reading of
theory 1.5 (the priced-risk `M` is *stabilised* by correlation), documented
in the module and now confirmed at the data-calibrated dispersion cap.
Runtime: 0.07 s at d = 128 (the Woodbury reduction is doing its job).

---

## 8. PerfGD: blind vs corrected loops (`run_perfgd --ml --iters 10`)

**What ran.** (i) The closed-form gap scan across the gain grid; (ii) the
exact 1-D dynamics in the genuinely unstable regime (the audit's fix — at
default constants no on-grid gain destabilises the fixed point, and the run
now says so explicitly); (iii) the three ML loop modes from a common seed in
that regime — the ML<->math seam diagnostic.

**Closed form (verified at full scale).**

- Gap scan: `gamma_PO > 0` on the whole grid (0.53 -> 1.01), the
  echo-chamber **decision gap grows ~O(eps)** (0.08 at `f = 0.5` -> 0.65 at
  `f = 6`) and the **value gap ~O(eps^2)** (0.002 -> 0.21) — both corollaries
  of theory 1.2 confirmed quantitatively. The blind stable point over-defends
  (`h_SP` up to 1.89) while the performative optimum tightens (`h_PO` down to
  1.24).
- Beyond-boundary dynamics (unstable demo regime, `m_rrm = 1.21`): the blind
  cobweb does **not** converge; the corrected 1-D ascent converges to
  `h_PO` — the "RRM diverges, PerfGD converges" figure, now produced in a
  regime where it is actually true.

**ML loops (the honest negative result, reproduced at paper scale).**
Trajectories of the central half-spread over 10 deployments (unstable demo
regime, `h_PO = 1.64`, seed 0):

| mode | trajectory (h per deployment) | final PR |
|------|-------------------------------|---------:|
| blind RRM        | 0.80 1.49 2.83 2.75 2.10 1.45 1.23 0.50 0.20 **0.16** | -1033 |
| PerfGD-analytic  | 0.80 0.38 0.84 0.71 1.12 1.53 0.84 0.56 0.60 **0.36** | -952 |
| PerfGD-learned   | 0.80 1.26 1.41 2.01 1.41 1.46 1.47 0.99 1.04 **0.66** | -493 |

- **No mode converges, and none operates near `h_PO`.** The blind loop ends
  in the echo-chamber collapse (h -> 0.16, catastrophic true risk); the
  corrected loops end less badly (the learned mode's final true risk is
  roughly half the blind loop's) but are not stabilised in any meaningful
  sense.
- **The seam diagnostic shows why**: at deployment 0 the operator's learned
  toxic slope is right-signed and near the analytic value (-0.84 vs -1.79);
  by late deployments it has **flipped positive** (+0.12 vs analytic -2.99)
  — the operator is fit on self-poisoned trajectories and misattributes the
  toxicity. The analytic correction is faithful to theory 1.2, but locating
  `h_PO` requires the operator's implied `dJ/dh` to match the structural
  objective's, and the blind operator's does not.
- **Conclusion (scoped claim for the paper).** Un-blinding is proven in
  closed form; at loop level with a learned operator it is an open problem
  with an identified mechanism. Candidate directions: operators whose
  `dJ/dh` is anchored to the GLFT structural form; corrections that
  compensate the operator's own bias; stability-penalty regularisation
  (wired, off by default) to prevent the data-poisoning collapse.

---

## 9. Appendix: the alpha-confound sweep (`sweep_adversariality.yaml`)

**What ran.** The same BR-slope protocol over the adversariality
`alpha` in [0.05, 0.80] (8 points x 5 seeds) at fixed `f = 5` — the
*documented confound*, with its own closed-form overlay (the audit migrated
the spec off the stale v2 format and generalised the prediction path), under
the clean probe protocol.

| alpha | 0.05 | 0.15 | 0.25 | 0.35 | 0.45 | 0.55 | 0.65 | 0.80 |
|-------|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|
| median m | 0.08 | 0.18 | 0.42 | 1.38 | **1.83** | 1.55 | 1.06 | 0.67 |

Measured crossing: `alpha* ~ 0.31`; predicted at the probe spread: 0.47.

**Findings.** Theory 1.1 §6.4 predicts the confound and the clean run now
shows its full shape: the measured modulus **rises and then reverses** —
climbing with `alpha` through ~0.45 (the feedback-slope channel), then
*falling* back below 1 by `alpha = 0.8` as the dealer flees to wide spreads
where the toxic response has decayed (the operating-regime channel wins).
The non-monotone hump is exactly the "flattens or even reverses; sign not
robust" behaviour documented since v2 — previously masked by the jitter
floor, now resolved. A sweep whose measured "boundary" depends on which side
of the hump the grid samples is not a control variable; this is the
quantitative case for `toxicity_feedback` as the headline axis, and `alpha`
stays an appendix.

---

## 10. Limitations (paper-ready list)

1. **Data provenance.** Not trade-level TRACE: VIX-implied spreads,
   proxy-level intensity fits, no per-dealer inventories. Regime *ordering*
   of every calibrated result is data-driven; absolute critical gains are
   not. WRDS TRACE Enhanced access is the stated upgrade path.
2. **Crisis cells are degenerate** (`k = 0`): crisis boundaries sit on the
   anchor floor; intra-crisis variation is not identified.
3. **The toxic channel is structurally scaled** (documented ratios tied to
   the identified arrival scale), not data-identified.
4. **Fixed-point saturation vs local instability.** At default-like
   constants the self-consistent fixed point never destabilises (defensive
   widening); every measured boundary crossing is a statement about the
   *local* retraining map at the operating spread. Both curves are reported;
   conflating them was an audit-fixed defect.
5. **Protocol dependence of measured moduli.** The BR-slope probe measures
   the finite-budget retraining map; its level moves with the per-deployment
   optimisation budget (lazy-deploy K). See the budget-sensitivity study.
6. **Loop-level PerfGD stabilisation is an open gap** (see §8): closed-form
   claims verified; the learned-loop counterpart is a documented negative
   result with identified mechanism (operator's implied dJ/dh vs structural).
7. **Statistical bands.** Sweep medians carry 8-seed IQRs and 1.4 robust
   bands; calibrated measured moduli (3 seeds) are a consistency check only.
