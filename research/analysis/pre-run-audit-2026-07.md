# Pre-run audit of the endo_market_v3 measurement layer (July 2026)

Before executing the paper-grade full-profile runs, the whole test suite was
exercised and the experiment/measurement code reviewed against the theory
documents. The **fast suite (103 tests) passed**, but **3 of the 7 slow
end-to-end guards failed** — and each failure traced to a genuine defect in
the *measurement layer* (probes, estimators, experiment protocols), not in the
theory modules. The committed smoke artifacts already contained the defective
numbers. Everything below was fixed and re-verified (110/110 tests pass)
before any paper-grade run was launched; the fixes landed in commit
`815425d`.

This document is the honest record: what was wrong, why, what changed, and
which *scientific* conclusions had to be reframed as a result.

---

## 1. Defects found and fixed

### 1.1 Multi-dealer environment contradicted its own derivation (theory 1.3)

**Symptom.** `test_joint_modulus_amplifies_with_dealers` failed: the measured
common-mode modulus at `N = 3` was exactly `0.0`; the committed smoke artifact
(`dealer_joint_moduli.csv`) showed measured `3.38` vs predicted `9.86` at
`N = 2` and measured `0.0` vs predicted `14.8` at `N = 3`.

**Two stacked root causes.**

1. *Coupling normalisation.* The derivation (1.3 §2.1, A2-multi) couples
   dealer `i`'s toxic responsiveness through the **sum** of competitors'
   responses, `exp(-c_t h_i) + kappa * Sum_{j != i} exp(-c_t h_j)` — that sum
   is precisely what produces the `N_eff = 1 + kappa (N - 1)` common-mode
   amplification. The environment implemented a **mean**-normalised pool,
   under which the common-mode slope is provably `m_1` for *every* `N` (the
   spillover redistributes rather than amplifies). Measured amplification
   ratios before the fix: 1.00 / 1.11 / 1.24 for `N` = 1 / 2 / 3 — flat, as
   the mean form predicts.
2. *Probe regime.* The probe ran at the default feedback gain (`f = 5`),
   deep past the boundary, where the closed-form best response to the measured
   toxic level rails at `policy.max_half_spread` in **every** probe arm for
   `N >= 2`. The finite difference of a clipped map is identically zero —
   the source of the artifact's `0.0`.

**Fix.** The environment now implements the derived sum coupling (the `N = 1`
bit-for-bit single-dealer reduction is unaffected — the spillover term
vanishes). Probing moved to the interior regime via
`interior_probe_config(cfg, N, f_probe)` (probe gain default 0.5 +
`liq_flow_boost / N` normalisation per the documented liquidity-inflation
gotcha), and `measure_joint_modulus_sim` now flags `br_clipped` and warns.

**Post-fix verification** (`f_probe = 0.5`, `h_ref = 1`, seed 0):

| N | measured common-mode ratio | predicted N_eff | clipped |
|---|---------------------------|-----------------|---------|
| 1 | 1.00 | 1 | no |
| 2 | 1.80 | 2 | no |
| 3 | 3.17 | 3 | no |

The 1.3 amplification law is now *verified on the genuine shared-pool market*.

### 1.2 The epsilon triangulation probed where its BR leg has no signal

**Symptom.** `test_triangulation_three_legs_agree` failed; the committed smoke
artifact carried `epsilon_br = 0.00055` vs analytic `0.215` — **390x off** —
while the README quoted only the two agreeing legs ("Sinkhorn/CKS within
~12%").

**Root cause.** The triangulation defaulted its probe spread to the analytic
fixed point `h* ~ 1.85`, but the learned pipeline *operates* at `h ~ 0.8`
(the blind operator underprices the toxic channel, so re-optimisation settles
near the quoting anchor). At `h*` the learned best-response map is nearly
flat — both probe arms land at `~0.8` regardless of the deployed spread, and
more optimisation budget makes the measured slope *smaller* (0.062 at 50
inner steps, 0.024 at 800). The probe was measuring a region the retraining
map never visits.

**Second issue (state drift).** Even at the operating spread, the a-priori
closed form understates the realised flow sensitivity several-fold: the
deployment itself inflates the liquidity ratio (`rho ~ 2.3` realised vs `1.0`
assumed) and lets mispricings build. Theory 1.1 §9 explicitly says the
constants are state-dependent and should be evaluated at the realised
reference state.

**Fix.** Default probe point is now the quoting anchor
(`reward.quote_anchor_ref` — in calibrated configs this is the *observed
market spread*), and the closed form is additionally evaluated at the
**realised deployment state** (`epsilon_analytic_realized`; new
`realized_reference()` measures mean |mispricing| and mean liquidity ratio
over the probes' own episodes). `run_calibrated --measure` probes at the
observed spread with a matching `m_pred_at_hobs` column.

**Post-fix numbers** (default config, `h_ref = 1.0`, seed 0, 8 episodes):

| quantity | value | ratio vs realised closed form |
|----------|-------|-------------------------------|
| analytic (a-priori A2 state) | 0.764 | 0.44x |
| analytic (realised state, §9) | 1.726 | 1.00 |
| Sinkhorn leg | 4.578 | 2.65x |
| CKS leg | 4.011 | 2.32x |
| BR-slope leg | 0.508 | 0.29x |

The distribution-space legs agree with the realised-state closed form within
a factor ~2.7; the remaining gap is the *state-feedback channel*
(`d state / d h`) that any frozen-state closed form omits. The BR leg sits
low by the finite-budget attenuation of the retraining map (§2.2 below) —
its test band is factor 5 and documented.

### 1.3 The sweep's predicted-vs-measured overlay compared two different spreads

**Symptom.** The predicted curve `m_pred(f)` in the sweep CSV never crosses 1
(saturates at ~0.51), while the measured curve reproduces v2's crossing at
`f ~ 1.3-2` — the headline "predicted vs measured crossing" would have shown
a prediction that predicts *no* crossing.

**Root cause.** The measurement probes the BR slope at a **fixed**
`h_ref = 1.0` (the loop's operating region); the prediction was evaluated at
the **drifting self-consistent fixed point** `h*(f) ~ 1.5-1.9`, where
defensive widening kills the toxic slope (theory 1.1 §6.3 saturation). Two
different spreads — an apples-to-oranges overlay. Theory §8 prescribes
evaluating both at the same point.

**Fix.** `run_sweep` now computes the closed form **at the probe's own
`h_ref`** (`predicted_epsilon_sweep_at`) as the overlay — this curve crosses
1 at `f* ~ 4.7` for the default microstructure — and keeps the saturating
fixed-point curve as explicitly-labeled context. Both appear in the CSV
(`m_pred`, `m_pred_hstar`) and the figure. The prediction path also
generalises to the `alpha` sweep, and `sweep_adversariality.yaml` was
migrated off the stale v2 spec format (`run_sweep` could not even parse it).

### 1.4 run_universe's "data-calibrated" dispersion was a no-op

**Symptom / root cause.** `rel_disp = min(max(disp / max(disp, 1e-9) * 0.35,
0.1), 0.8)` — the expression `disp / max(disp, 1e-9)` is identically 1, so
the "calibrated" per-bond sigma dispersion was silently the constant 0.35.
The aggregated G2 panel it read cannot identify per-bond vol heterogeneity in
the first place (one cross-sectional sigma per month mixes vol heterogeneity
with factor exposure).

**Fix.** v3 now ships the per-CUSIP monthly returns
(`reflex_G_bond_returns_monthly.csv`, 212 bonds) and computes the dispersion
honestly: drop stale zero-vol bonds (~5%, flat marks), winsorise the sigma
distribution at p05/p95 (the pipeline's own convention), take the coefficient
of variation. Measured CV = **1.32** — real per-bond vol heterogeneity is
*higher* than the structural band allows, so the value is clipped to the 0.8
band ceiling (stated in the experiment output rather than silently defaulted).

### 1.5 run_perfgd's "beyond the boundary" demo was not beyond any boundary

**Symptom / root cause.** With default constants the closed-form fixed-point
modulus never exceeds ~0.51 on the whole gain grid, so the demo's fallback
(`grid[-1]`) produced a "RRM diverges" figure at a gain where the closed-form
cobweb *converges*. The old slow test asserted loop-level stabilisation at
`f = 6` citing "far beyond epsilon* ~ 1.3" — but 1.3 is v2's **probe-point**
crossing, not the fixed-point boundary; at the fixed point that gain is
stable.

**Fix.** When no on-grid gain destabilises the fixed point, the demo now uses
the genuinely unstable regime (slow toxic decay `c_t = 0.8`, `alpha = 1`,
`I = 3`, `w = 0.15` — the same regime the closed-form tests verify;
`m(h*) = 1.205` there), honestly labeled in the figure and console. The ML
loop layer runs in the same regime.

### 1.6 The sweep's exploration jitter inflated the measured modulus ~3x (found during results analysis)

**Symptom.** In the first full-profile sweep the probe read `m ~ 0.66` at
**zero** feedback gain — where the structural modulus is exactly 0 — and the
measured curve crossed 1 near `f ~ 1`, far left of the closed form's
`f* ~ 4.7` at the same spread. Meanwhile the budget-sensitivity study (run
under `default.yaml`) showed the same probe tracking the structural modulus
within ~5% in the contracting regime.

**Root cause.** The sweep spec builds its config from the YAML `base` block,
with unlisted fields inheriting the *dataclass* defaults — and the dataclass
default for `rrm.collection_jitter` (exploration noise on quotes during data
collection) was **0.20**, while `default.yaml` uses **0.05**. A controlled
A/B on the sweep base confirms jitter is the driver:

| protocol | measured m at f = 0 | measured m at f = 2 (structural: 0.426) |
|----------|--------------------:|------------------------------------------:|
| jitter 0.20 (inherited) | 0.512 | 1.252 |
| jitter 0.05 (clean)     | 0.059 | **0.410** |

Under the clean protocol the probe agrees with the closed form to ~4% in the
contracting regime. **Consequence: v2's celebrated measured crossing at
`epsilon* ~ 1.3` was substantially a jitter artifact** — the trend with `f`
was real, the level and the crossing location were protocol-contaminated.

**Fix.** `collection_jitter` default aligned to 0.05 (and pinned explicitly
in both sweep specs with a comment); both sweeps rerun under the clean
protocol; the final analysis uses the clean-run artifacts.

**Beyond-boundary caveat that remains.** Past the boundary the seed-level
probe readings bifurcate (0.2 vs 4.9 at `f = 5` with large budgets): the
local-slope interpretation breaks down where the map is not a contraction —
exactly the theory's own A4 caveat. Measured "moduli" deep in the unstable
regime are finite-difference diagnostics, not slopes.

---

## 2. Scientific reframings (findings, not bugs)

The fixes above forced three honest reframings that the analysis and the
paper must carry.

### 2.1 Fixed-point saturation is structural; the empirical boundary is a local statement

At default-like constants the self-consistent fixed point `h*(f)` widens fast
enough that its modulus saturates **below 1** for every reasonable gain — a
consequence of the anchor reference sitting above the adverse severity
(`h_ref > psi`). Genuine fixed-point instability requires the toxic slope to
survive at the fixed point (slow decay `c_t < ~1`). The celebrated v2
crossing (`epsilon* ~ 1.3`) is therefore a statement about the **local
retraining map at the operating spread** (tighter than `h*`), i.e. "if the
loop is operated near `h = 1`, retraining feedback is locally expansive past
`f*`" — not "the equilibrium destabilises". Both curves are now reported.

### 2.2 The measured modulus is protocol-dependent

The BR-slope probe measures the *finite-budget* retraining map. Its level
depends materially on the per-deployment optimisation budget
(at `h_ref = 1`, `f = 5`: `m_hat` = 0.54 at 50 inner steps, 0.14 at 200) —
the lazy-deploy stabilisation knob of theory 1.1 §6.2, live in the
measurement itself. Probes use the loop's own budget; measured levels are
comparable within a protocol, not across protocols.

### 2.3 Loop-level PerfGD does not (yet) stabilise the learned loop

The 1.2 claims hold **in closed form** and are verified: in the unstable
regime the blind cobweb diverges while the corrected 1-D ascent converges to
`h_PO`. But in the full ML stack, at every tested scale/regime/update rule
(default and unstable regimes, `rgd` and full-reoptimisation rules, seeds
0/1):

- no mode (blind, analytic-corrected, learned-corrected) converges within
  the tested horizons;
- the analytically-corrected loop equilibrates far *below* `h_PO` — the
  correction term itself is faithful (`Delta = -beta (h - psi) eps(h)`,
  negative above `psi` by design), but locating `h_PO` requires the
  *operator's* implied `dJ/dh` to match the structural objective's, and the
  blind operator's does not;
- at high gain the blind loop exhibits an **echo-chamber collapse** (spreads
  driven toward 0 on self-poisoned data) rather than the theory's oscillatory
  divergence, and the on-trajectory learned `dD/dphi` flips sign.

The slow test now locks in exactly what is verified (all modes complete, the
blind loop's instability is real, seam diagnostics finite), and
`run_perfgd --ml` is framed as the **ML<->math seam diagnostic** it is. The
gap — making the corrected *learned* loop actually find `h_PO` — is a
concrete, honestly-stated open problem (candidate directions: operator
architectures whose `dJ/dh` is anchored to the GLFT structural form;
correction terms that compensate the operator's own bias, not just the
distribution shift).

### 2.4 Smoke artifacts are not results

The committed smoke artifacts contained the defective numbers above
(the 0.0 dealer modulus, the 390x-off BR leg). Smoke-profile outputs prove
the pipeline runs; only full-profile artifacts, produced after this audit,
are quotable.

---

## 3. Verification state after the audit

- **110/110 tests pass** (103 fast + 7 slow), including the recalibrated
  guards: interior-regime dealer amplification (ratios 1.80/3.17 vs predicted
  2/3), realised-state triangulation bands, loop-level integration in the
  genuinely unstable regime.
- Environment: Python 3.9.13, torch 2.8.0+cpu, numpy 2.0.2 (repo venv);
  Windows 11, CPU only.
- Fixes commit: `815425d`. The paper-grade full-profile suite was launched
  only after this state was green.
