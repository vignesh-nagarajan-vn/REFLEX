# 1.6 - Lazy Deployment: the K-Step Outer Map and the Effective Curvature

**STATUS: DONE (v4 addendum, derivation + code).** This document discharges the
last open "Training and tuning" item of [`../README.md`](../README.md) ("Sweep
lazy-deploy `K` and report effect on `gamma_eff`"). It derives, from the same
linearisation as [`01-analytic-stability-boundary.md`](01-analytic-stability-boundary.md),
the closed form of the outer retraining map when each deployment is followed by
only `K` inner gradient steps (the `rrm.update_rule = "rgd"` scheme with
`K = rrm.rgd_steps`) instead of an exact best response. The companion code is
[`reflex/theory/lazy_deploy.py`](../../endo_market_v4/reflex/theory/lazy_deploy.py)
with the CRN probe `measure_rgd_response` in
[`reflex/estimators/br_slope.py`](../../endo_market_v4/reflex/estimators/br_slope.py)
and the experiment `experiments/run_lazy_deploy.py`.

> **One-line thesis.** One deployment of a `K`-step retrainer realises only the
> fraction `lam_K = 1 - c^K` of the exact best-response cobweb (`c = 1 -
> eta_h*gamma` the inner per-step contraction), so the outer map's slope is the
> convex combination `mu(K) = (1-lam_K)*1 + lam_K*(-m)`. Laziness therefore
> *interpolates* between "do nothing" and the exact cobweb - it can deadbeat the
> loop at finite `K`, it keeps an RRM-unstable market (`m > 1`) stable for all
> `K` up to a computable `K_max`, and read through the plain-RRM identity it
> manifests as an **effective curvature** `gamma_eff(K) = gamma * m / |mu(K)|`
> with two branches: *inertia* (`gamma_eff < gamma`) below the equal-modulus
> count `K_eq`, *extra stiffness* (`gamma_eff > gamma`) above it, diverging at
> the deadbeat count and decaying to the true `gamma` as `K -> infinity`.

**Formatting note.** As in 1.1-1.5, all mathematics is in fenced code blocks /
backtick-ASCII (no LaTeX/MathJax). Symbols carry over from 1.1; new symbols are
defined in §0.

---

## 0. Carry-over and new notation

From [`01-analytic-stability-boundary.md`](01-analytic-stability-boundary.md) we
reuse the frozen-`T` objective `J(h; T)` (strongly concave in `h` with curvature
`gamma`), the toxic response `tau(h)` with sensitivity `epsilon = |d tau/dh|`,
the frozen best response `BR(T) = argmax_h J(h; T)`, the fixed point `h*`, and
the signed exact-cobweb slope at the fixed point

```
   BR'(h*) = - epsilon*beta/gamma = -m ,      m >= 0 .
```

| New symbol | Reading | Definition |
|------------|---------|------------|
| `K` | inner steps per deployment | `rrm.rgd_steps` |
| `eta_h` | inner step size *in spread units* | the image of `rrm.rgd_lr` under the policy parameterisation, §5 |
| `c` | inner per-step contraction | `c = 1 - eta_h*gamma`, assumed `0 <= c < 1` |
| `mu(K)` | K-step outer-map slope at `h*` | derived in §2 |
| `lam_K` | lazy-deploy weight | `lam_K = 1 - c^K` |
| `m_eff(K)` | effective outer modulus | `|mu(K)|` |
| `gamma_eff(K)` | effective curvature | `epsilon*beta / m_eff(K)`, §4 |
| `K_db` | deadbeat step count | zero of `mu`, §3.2 |
| `K_max` | largest stable `K` when `m > 1` | §3.3 |

---

## 1. The scheme

Exact RRM (analysed in 1.1) redeploys the full best response each round:

```
   h_{t+1} = BR(tau(h_t))                                (exact; slope -m at h*)
```

The implemented scheme (`rrm.update_rule = "rgd"`) is *lazy*: after deploying
`h_t` and refitting the operator on the induced data `D(h_t)`, the dealer takes
only `K` plain gradient steps on the frozen-`T` objective, warm-started from the
deployed spread:

```
   h^{(0)} = h_t ,
   h^{(j+1)} = h^{(j)} + eta_h * d1 J(h^{(j)}; tau(h_t)) ,   j = 0..K-1 ,
   h_{t+1} = h^{(K)} .
```

`K -> infinity` recovers exact RRM; `K = 1` is greedy repeated gradient descent
(RGD, Mendler-Duenner et al. 2020). The question 1.6 answers: what is the outer
map's contraction as a function of `K`, and what does it look like if one
insists on reading it through the plain-RRM identity `m = epsilon*beta/gamma`?

---

## 2. The K-step slope

**Lemma 1 (inner linearisation).** Linearise the inner iteration at the frozen
best response `B_t := BR(tau(h_t))`, where `d1 J(B_t; tau(h_t)) = 0` and
`d1^2 J = -gamma`:

```
   h^{(j+1)} - B_t = (1 - eta_h*gamma) * (h^{(j)} - B_t) = c * (h^{(j)} - B_t) ,
```

so after `K` steps

```
   h_{t+1} - B_t = c^K * (h_t - B_t) .                                   (2.1)
```

**Lemma 2 (outer linearisation).** At the fixed point `h*` (where `B(h*) = h*`),
`B_t - h* = BR'(h*) * (h_t - h*) = -m * (h_t - h*)` to first order (1.1 §3.3).

**Theorem 1 (K-step outer slope).** Combining (2.1) with Lemma 2,

```
   h_{t+1} - h* = (B_t - h*) + c^K * ( (h_t - h*) - (B_t - h*) )
                = [ -m + c^K * (1 + m) ] * (h_t - h*) ,
```

so the K-step deployment map has slope

```
   mu(K) = -m + c^K * (1 + m) .                                          (6.1)
```

**Corollary 1 (lazy-deploy interpolation, §2.2).** With `lam_K = 1 - c^K`,

```
   mu(K) = (1 - lam_K) * 1 + lam_K * (-m) :
```

one lazy deployment realises the fraction `lam_K` of the exact cobweb and keeps
the fraction `1 - lam_K` of "stay where you are". `mu(0) = 1`, `mu(K)` is
strictly decreasing in `K` (for `c in (0,1)`, `m + 1 > 0`), and
`mu(K) -> -m` as `K -> infinity` - the exact-RRM limit.

---

## 3. Consequences

### 3.1 Effective modulus

The outer loop contracts iff `m_eff(K) := |mu(K)| < 1`. Since `mu` decreases
monotonically from `1` toward `-m`:

* for `m < 1` every `K >= 1` is stable (`mu(K) in (-m, c(1+m)-m]`, and both
  endpoints lie in `(-1, 1)`);
* `m_eff` is **not** monotone in `K`: it first *falls* (the map slope drops
  from `~1` toward `0`), hits zero at the deadbeat count, then *rises* back to
  `m` - lazy loops between `K_db` and `infinity` are strictly *less*
  contractive than the deadbeat loop.

### 3.2 Deadbeat deployment

`mu(K) = 0  <=>  c^K = m/(1+m)`, i.e.

```
   K_db = ln( m/(1+m) ) / ln(c) .                                        (6.2)
```

At `K_db` the linearised outer loop converges in a *single* deployment
(deadbeat control): the inner under-shoot exactly cancels the cobweb
over-shoot. `K_db` is decreasing in `m` for fixed `c` and increasing in `c`.

### 3.3 Laziness stabilises an unstable market

For `m > 1` the exact cobweb diverges, but `mu(K) > -1` holds while
`c^K > (m-1)/(m+1)`:

```
   K < K_max = ln( (m-1)/(m+1) ) / ln(c) .                               (6.3)
```

Every `K <= floor(K_max)` keeps the loop linearly stable - the quantitative
version of the known qualitative fact that RGD converges on a strictly wider
regime than RRM (Perdomo et al. 2020, Thm 3.8 vs 3.5): *not* retraining to
convergence is a stabiliser. Beyond `K_max` the loop inherits the cobweb's
divergence.

### 3.4 What is genuinely predictive here

Given `m` (from 1.1) and `c`, all of §3 is parameter-free. `c` itself is not
a-priori (§5), so the falsifiable content is: (i) the *functional form* (6.1)
fits the measured `mu_hat(K)` with a single fitted `c`; (ii) the `K ->
infinity` asymptote equals the independently measured exact-BR modulus; (iii)
the curve is monotone decreasing and crosses zero iff `m > 0`.

---

## 4. The effective curvature `gamma_eff`

The to-do item asks for the effect of `K` on `gamma_eff`: read the measured
K-step modulus through the plain-RRM identity `m = epsilon*beta/gamma` as if
the loop were an exact retrainer,

```
   m_eff(K) = epsilon*beta / gamma_eff(K)
   =>  gamma_eff(K) = epsilon*beta / m_eff(K) = gamma * m / |mu(K)| .    (6.4)
```

The sign structure of §3 splits this into two branches, separated by the
**equal-modulus count** `K_eq` where the lazy modulus equals the exact one on
the positive branch (`mu(K) = +m`):

```
   c^K_eq = 2m/(1+m)   =>   K_eq = ln( 2m/(1+m) ) / ln(c) ,              (6.5)
```

(`K_eq = 0` when `m >= 1`: every `K >= 1` is already past it).

* **Inertia branch (`K < K_eq`).** The under-trained loop crawls:
  `m_eff(K) > m`, so `gamma_eff < gamma` - the lazy loop reads as a *softer*
  objective than the true one. An observer fitting the plain-RRM boundary here
  *over-estimates* the market's fragility (the measured modulus is inflated by
  under-training, not by performative feedback). At near-zero-`m` markets
  (e.g. the default calibration, `m ~ 0.03` at the probe spread) every
  practical `K` sits in this branch - the empirically relevant regime.
* **Stiffness branch (`K > K_eq`).** `m_eff(K) < m`, so `gamma_eff > gamma`:
  the overshoot-cancelling regime looks stiffer than the truth,
  `gamma_eff -> infinity` at the deadbeat count `K_db` (the map slope crosses
  zero - no finite curvature reproduces a one-shot-convergent retrainer) and
  decays to `gamma` *from above* as `K -> infinity`.

Either way `gamma_eff` must be reported *alongside* `m_eff`, never instead of
it, and the branch must be stated.

---

## 5. Protocol: measuring `mu(K)` and fitting `c`

`eta_h` is a step size in *spread units*. The implementation's `rrm.rgd_lr`
acts in *parameter* space (on the policy weights through softplus and the
state-dependent quote map), so `c = 1 - eta_h*gamma` is not computable from the
config - it must be identified from the loop itself. The audit-consistent
protocol (cf. `research/analysis/pre-run-audit-2026-07.md`):

1. **Signed CRN K-probe.** Deploy fixed spreads `h_ref +/- delta`, run the full
   collect -> refit pipeline for each, take `K` RGD steps *warm-started from
   the deployed policy* (the lazy convention - a fresh init would measure a
   different map), and form the **signed** difference quotient
   `mu_hat(K) = (out_plus - out_minus) / (2*delta)`. Signed, because the zero
   crossing at `K_db` is part of the prediction. Common random numbers across
   the paired probes, `collection_jitter = 0.05`, probe at the operating
   spread.
2. **Anchor `m`.** Measure the exact-BR modulus with the standard 1.1 probe on
   the same seeds - the `K -> infinity` asymptote. (The realized-state closed
   form of 1.1 §9 predicts it; the measured anchor keeps the K-curve
   comparison instrument-internal.)
3. **Fit `c`.** One-parameter least squares of (6.1) on the median
   `mu_hat(K)` curve (`fit_inner_contraction`). Report the fit residuals: a
   poor fit falsifies the linearised model, not just the parameter.
4. **Report** `m_eff(K)` and `gamma_eff(K) = gamma*m/m_eff(K)` (6.4), the
   deadbeat `K_db` (6.2), and - in an `m > 1` regime - the stability window
   `K <= K_max` (6.3).

**Caveats (state them in any write-up).** (i) The derivation is a local
linearisation: far from `h*`, or when the blowup guard trips, (6.1) is only a
first-order model. (ii) The probe's inner steps act through the operator
`T_theta`, so operator misfit perturbs `c` (it is a property of the *implemented*
loop, not of `T_true` alone). (iii) With `K` large and `eta_h` too big
(`c < 0`), the inner iteration itself oscillates; the implementation's default
step sizes keep `c in (0, 1)` and the fit constrains `c` there.

---

## 6. Code map

| Object | Where |
|--------|-------|
| `mu(K)`, `m_eff`, `lam_K` (6.1) | `reflex/theory/lazy_deploy.py` : `k_step_slope`, `effective_modulus`, `lazy_weight` |
| `K_db` (6.2), `K_max` (6.3) | `lazy_deploy.py` : `deadbeat_k`, `max_stable_k` |
| `gamma_eff(K)` (6.4), `K_eq` (6.5) | `lazy_deploy.py` : `gamma_eff`, `equal_modulus_k` |
| the `c` fit (§5.3) | `lazy_deploy.py` : `fit_inner_contraction` |
| assembled curve | `lazy_deploy.py` : `lazy_deploy_curve` |
| signed CRN K-probe (§5.1) | `reflex/estimators/br_slope.py` : `measure_rgd_response` |
| the sweep experiment | `experiments/run_lazy_deploy.py` (artifacts `lazy_deploy_sweep.csv`, `lazy_deploy_summary.csv`, `lazy_deploy_gamma_eff.png`) |
| tests | `tests/test_lazy_deploy.py` |
