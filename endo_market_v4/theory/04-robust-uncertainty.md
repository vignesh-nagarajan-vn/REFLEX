# 1.4 - Distributionally Robust Stability and the `O(1/sqrt(n))` Ambiguity Radius

**STATUS: DONE (Priority 4, derivation/theorem + code).** This document discharges
the *mathematical* content of methodology target **1.4** in
[`../README.md`](../README.md). It turns the *point* boundary of 1.1 (and its
multi-dealer lift, 1.3) into a **statistically defensible** one: because the
sensitivity `epsilon` is *estimated* from a finite simulation, the crossing
`epsilon = epsilon*` is a random event, and a bare point estimate of `epsilon*` is
not a claim. We

1. show the estimator `epsilon_hat_n` from `n` simulated episodes concentrates at
   the **parametric rate** `|epsilon_hat_n - epsilon| = O_p(1/sqrt(n))` -- and,
   crucially, that this rate hinges on the **common-random-numbers (CRN)**
   construction already in
   [`response_modulus.py`](../../archive/endo_market_v2/endo_market/analysis/response_modulus.py):
   a *naive* finite difference only achieves `O(n^{-1/3})` (§2);
2. build an **ambiguity set** `U_n` of radius `delta_n = O(1/sqrt(n))` around the
   estimate, as the `epsilon`-ball fit from cross-seed variance and, equivalently,
   a Wasserstein DRO ball via the Sinkhorn estimator (§3);
3. derive the **distributionally robust stability certificate** (the minimax stable
   point of Cao & Shi 2022): certify stability at confidence `1 - a` iff the
   *upper* confidence bound `epsilon_hat_n + delta_n < gamma/beta`, giving a robust
   boundary `epsilon*_rob = gamma/beta - delta_n` that is conservative and closes
   on the nominal one at rate `O(1/sqrt(n))` (Theorem 1, §4);
4. derive the **sample-complexity curve** `n_req = (z*sigma_eps / Delta)^2` to
   resolve a market a distance `Delta` from the boundary, so `n_req -> inf` as
   `Delta -> 0` -- the honest statement that the phase transition is
   statistically hard to pin exactly (§5);
5. compose it with 1.3 (robust *systemic* boundary) and separate **statistical**
   uncertainty (shrinks with data) from **structural/model** uncertainty (a fixed
   floor) -- the "off by 20%" sensitivity question (§6).

Every constant is inherited from 1.1/1.3; the new content is the *statistics* of
the existing estimator. The companion *code* --
[`analysis/robust_boundary.py`](../../archive/endo_market_v2/endo_market/analysis/robust_boundary.py),
the cross-seed variance harness, the certificate, and the sample-complexity /
log-log rate checks -- is now implemented (§8); the optional Sinkhorn `epsilon`
estimator remains the one deferred item.

> **One-line thesis.** 1.1 says "stable iff `epsilon < gamma/beta`." But you never
> know `epsilon`; you know `epsilon_hat_n +/- delta_n` from `n` episodes. The
> honest object is not the point boundary but the **certificate**: declare stable
> only when the whole ambiguity ball is on the stable side,
> `epsilon_hat_n + delta_n < gamma/beta`. The robust boundary sits a margin
> `delta_n = O(1/sqrt(n))` inside the nominal one; more simulation buys a tighter
> margin, and the sample size needed to resolve the crossing blows up as
> `(distance to boundary)^{-2}`. The CRN trick in the existing probe is exactly
> what makes that margin shrink at the parametric `1/sqrt(n)` rate rather than the
> `n^{-1/3}` of a naive finite difference.

**Formatting note.** As in 1.1-1.3, all mathematics is in fenced code blocks /
backtick-Unicode (no LaTeX/MathJax). Every structural symbol carries over from
1.1/1.3; new (statistical) symbols are defined in §0. A compilable LaTeX companion
lives in [`04-robust-uncertainty.tex`](04-robust-uncertainty.tex).

---

## 0. Carry-over and new notation

From [`01-analytic-stability-boundary.md`](01-analytic-stability-boundary.md) and
[`03-multi-dealer-systemic-risk.md`](03-multi-dealer-systemic-risk.md):

```
   m       = epsilon*beta/gamma          (single-dealer modulus, 1.1)
   epsilon* = gamma/beta                  (nominal boundary; multi-dealer: gamma/(N_eff*beta), 1.3)
   m_hat   = | BR(h+delta) - BR(h-delta) | / (2*delta)   (the CRN BR-slope probe, response_modulus.py)
```

| New symbol | Reading | Definition |
|------------|---------|------------|
| `n` | number of simulated episodes feeding one estimate | `rrm.eval_episodes` / rollouts, §1 |
| `S` | number of independent seeds (for cross-seed variance) | §3, §8 |
| `epsilon_hat_n` | estimator of `epsilon` from `n` episodes | §1 (via `m_hat*gamma/beta`, or Sinkhorn) |
| `sigma_eps` | per-episode asymptotic std of the estimator | §2 (delta-method / IPA variance) |
| `delta_n` | ambiguity radius (confidence half-width) | `= z*sigma_eps/sqrt(n) = O(1/sqrt(n))`, §2-§3 |
| `a`, `z` | miscoverage level; `z = z_{1-a}` normal quantile | §2 |
| `U_n` | ambiguity set around the estimate | `epsilon`-ball or Wasserstein ball, §3 |
| `epsilon*_rob` | robust boundary (minimax stable point) | `= epsilon* - delta_n`, §4 |
| `Delta` | distance of the truth to the boundary, `= |epsilon* - epsilon|` | §5 |
| `n_req` | episodes needed to certify at distance `Delta` | `= (z*sigma_eps/Delta)^2`, §5 |
| `lambda` | Sinkhorn entropic-regularisation strength | §3 (Sinkhorn route) |
| `eta_mod` | *structural* (model) relative ambiguity, a fixed floor | §6 |

---

## 1. The estimation problem: `epsilon*` is a random crossing

The nominal theory (1.1) is a deterministic inequality `epsilon < gamma/beta`. But
`epsilon` is never observed. It is *estimated* -- by the CRN BR-slope probe
`response_modulus.py`, or by a Sinkhorn/Wasserstein or CKS estimator (1.1 §8's
triangulation) -- from a finite number `n` of simulated episodes. Write the
estimator `epsilon_hat_n`; the probe actually returns the modulus `m_hat_n`, and
since `gamma`, `beta` are closed forms (1.1, known exactly given config),

```
   epsilon_hat_n = m_hat_n * gamma / beta ,        Var(epsilon_hat_n) = (gamma/beta)^2 * Var(m_hat_n) .
```

So all statements below hold verbatim for the *directly measured* modulus `m_hat_n`
with boundary `m = 1`; we phrase them in `epsilon` to match the methodology's
"ambiguity ball around estimated `epsilon`". The decision "stable vs. unstable" is
a hypothesis test `H0: epsilon >= epsilon*` against `H1: epsilon < epsilon*`, and
a point estimate `epsilon_hat_n` reported without its sampling error is exactly the
overclaim this priority exists to prevent.

---

## 2. Concentration at the parametric rate -- and why CRN is load-bearing

### 2.1 The estimator is a smooth functional of sample averages

The best response `BR(h_dep)` solves the FOC `F(h; theta) = 0`, where `theta`
collects the population flow moments at the deployment (arrival, imbalance,
informed-signal statistics of `clients.py`). Empirically, `theta` is replaced by a
sample average `theta_hat_n = (1/n) sum_i X_i` over `n` i.i.d. episodes, and
`BR_hat_n = BR(theta_hat_n)`. Because `F` is smooth with `F_h = -gamma != 0` on the
operating range, the implicit-function theorem makes `BR` a smooth function of
`theta`, so by the CLT + delta method (standard M-estimator asymptotics)

```
   sqrt(n) ( BR_hat_n - BR )  ->  N( 0 , grad_theta BR^T * Sigma_X * grad_theta BR )      (delta method)
```

i.e. each best response is `sqrt(n)`-consistent. The question is what the *finite
difference* of two such estimates does.

### 2.2 The naive finite difference: `O(n^{-1/3})`

Estimate `m` by an *independent-noise* symmetric difference (two separately seeded
deployments):

```
   m_hat = ( BR_hat_n(h+delta) - BR_hat_n(h-delta) ) / (2*delta) ,   the two BRs independent.
```

Two error sources compound:

```
   deterministic finite-difference bias :  E[m_hat] - m  =  (1/6) BR'''(h) delta^2 + O(delta^4)  =  O(delta^2)
   sampling variance (independent noise):  Var(m_hat)    =  ( Var(BR_hat_n) + Var(BR_hat_n) ) / (4 delta^2)
                                                         =  sigma_BR^2 / ( 2 * delta^2 * n )  =  O( 1/(delta^2 n) ) .
```

The variance **blows up as `delta -> 0`** -- the classic numerical-differentiation
bias/variance conflict. Minimising the MSE `O(delta^4) + O(1/(delta^2 n))` over
`delta` gives the optimum `delta ~ n^{-1/6}` and

```
   MSE_naive = O( n^{-2/3} )   ==>   | m_hat - m | = O_p( n^{-1/3} )      (naive rate) .
```

A naive probe therefore delivers an ambiguity radius shrinking only like
`n^{-1/3}` -- **not** the `1/sqrt(n)` the methodology promises.

### 2.3 Common random numbers restore `O(1/sqrt(n))`

`response_modulus.py` does **not** use independent noise: it drives both probe
deployments with the *same* generators (its docstring: "common random numbers ...
the operator-fit sampling noise largely cancels"). Under CRN both best responses
are functions of the *same* draw `omega`, so the difference is a **pathwise
(infinitesimal-perturbation-analysis) derivative**:

```
   D(delta, omega) := BR_hat_n(h+delta; omega) - BR_hat_n(h-delta; omega)
                    = 2*delta * d_h BR(h; omega)  +  O(delta^3)          (pathwise Taylor, shared omega)
```

so the leading `O(1)` sampling fluctuations of the two BRs **cancel in the
difference**, leaving a term that is itself `O(delta)`. Hence

```
   Var( m_hat ) = Var( D )/(4 delta^2) = Var( d_h BR(h; omega) ) + O(delta^2)
                = sigma_IPA^2 / n  +  O(delta^2) ,
```

which is `O(1/n)` **uniformly in `delta`** -- the variance no longer explodes as
`delta -> 0`. The CRN finite difference is an IPA-consistent gradient estimator
with *parametric* variance. Taking `delta` small kills the `O(delta^2)` bias for
free (no variance penalty), giving the parametric rate

```
   | m_hat_n - m | = O_p( 1/sqrt(n) ) ,        and likewise  | epsilon_hat_n - epsilon | = O_p( 1/sqrt(n) ) .   (*)
```

**This is the crux of 1.4:** the `O(1/sqrt(n))` robust radius the methodology
claims is *not* automatic -- it is bought by the variance-reduction design already
in the code. Stating this makes the priority a real result, not an assumption.

### 2.4 A finite-sample (non-asymptotic) certificate

For a *certificate* (not just an asymptotic CI) assume the per-episode IPA
derivative is sub-Gaussian with proxy `sigma_eps` (bounded flow, cap slack, A4).
Hoeffding/sub-Gaussian concentration gives, for every `n`,

```
   P( | epsilon_hat_n - epsilon | > t )  <=  2 * exp( - n t^2 / (2 sigma_eps^2) ) ,
```

so with probability `>= 1 - a`,

```
   | epsilon_hat_n - epsilon |  <=  delta_n := sigma_eps * sqrt( 2 * log(2/a) / n )  =  O( sqrt( log(1/a) / n ) ) .   (**)
```

Both (`*`) (asymptotic, tight constant `z*sigma_eps`) and (`**`) (finite-sample,
honest for a certificate) give `delta_n = O(1/sqrt(n))`; we use (`**`) for the
robust boundary and (`*`) for the sample-complexity constant.

---

## 3. The ambiguity set

### 3.1 The `epsilon`-ball, fit from cross-seed variance

The methodology asks for the ambiguity set to be *fit from the variance of
`epsilon` estimates across seeds*. Run the probe under `S` independent seeds, each
on `n` episodes, obtaining `epsilon_hat_n^{(1)}, ..., epsilon_hat_n^{(S)}`. The
cross-seed sample mean and variance

```
   ebar = (1/S) sum_s epsilon_hat_n^{(s)} ,     s_eps^2 = (1/(S-1)) sum_s ( epsilon_hat_n^{(s)} - ebar )^2
```

estimate `epsilon` and the estimator variance `sigma_eps^2/n` directly (no analytic
`sigma_eps` needed). The ambiguity set is the interval

```
   U_n = [ ebar - delta_n , ebar + delta_n ] ,     delta_n = z_{1-a} * s_eps / sqrt(?)      (see note) .
```

*Note on `S` vs `n`.* `s_eps` already estimates the *per-estimate* std
`sigma_eps/sqrt(n)`, so `delta_n = z_{1-a} * s_eps` is the confidence half-width of
a single `n`-episode estimate; the `1/sqrt(n)` scaling is then *tested*, not
assumed, by re-running at several `n` and checking `s_eps ~ n^{-1/2}` (§8). If
instead one certifies with the *pooled* `S*n`-episode estimate, the half-width is
`z_{1-a} * s_eps / sqrt(S)`, and the effective sample size is `S*n`. Either way the
radius is `O(1/sqrt(#episodes))`.

### 3.2 The Wasserstein ball and the Sinkhorn rate (paper #14)

The principled ambiguity set is a **Wasserstein-1 ball** around the empirical
induced distribution, `U_n^W = { Q : W_1(Q, D_hat_n) <= r_n }`, because Perdomo's
`epsilon` *is* a `W_1`-Lipschitz constant of `phi -> D(phi)`. Two facts pin the
radius:

- **Raw empirical `W_1` is cursed:** `E[ W_1(D_hat_n, D) ] = O(n^{-1/d})` in
  dimension `d` -- too slow to give a `1/sqrt(n)` radius in the multi-feature flow
  state.
- **The entropic/Sinkhorn estimator is parametric:** with fixed regularisation
  `lambda > 0`, the Sinkhorn divergence estimator has bias `O(lambda)` and
  **`O(1/sqrt(n))` standard error** (Cuturi & Peyré 2019, Ch. 7; Genevay et al.;
  Mena & Niles-Weed), i.e.

  ```
     | W_{1,lambda}(D_hat_n, D_hat'_n) - W_{1,lambda}(D, D') |  =  O_p( 1/sqrt(n) )      (fixed lambda) .
  ```

So the **Sinkhorn estimator is what realises the `1/sqrt(n)` radius** for the OT
route, exactly as the CRN construction realises it for the BR-slope route (§2.3).
The regularisation trades a controllable `O(lambda)` bias for the parametric
variance; picking `lambda` to balance bias against `sigma/sqrt(n)` is the tuning
task flagged in the methodology (`tune Sinkhorn entropic regularisation strength`).
By Wasserstein-DRO duality (worst-case risk over a `W_1` ball of radius `r` = nominal
risk + `r *` Lipschitz constant), the `W_1`-ball of radius `r_n = O(1/sqrt(n))`
inflates the effective sensitivity by `O(1/sqrt(n))`, coinciding to leading order
with the `epsilon`-ball of §3.1. We report both; they agree at the rate that
matters.

---

## 4. The distributionally robust stability certificate

We now state the minimax stable point of Cao & Shi (2022) in the scalar market
model.

**Definition (robust-stable).** A configuration is *robust-stable at confidence
`1 - a`* if the loop contracts for the **worst-case** `epsilon` in the ambiguity
set:

```
   sup_{ epsilon' in U_n }  m(epsilon') = sup_{ epsilon' in U_n } epsilon'*beta/gamma  <  1 .
```

Since `m` is increasing in `epsilon` and `U_n = [ebar - delta_n, ebar + delta_n]`,
the sup is attained at the upper endpoint, giving:

**Theorem 1 (robust boundary / sample-complexity of the certificate).** *With the
CRN BR-slope (or fixed-`lambda` Sinkhorn) estimator and radius
`delta_n = O(1/sqrt(n))` of §2-§3, a configuration is robust-stable at confidence
`1 - a` iff the upper confidence bound lies below the nominal boundary,*

```
   ebar + delta_n  <  gamma / beta                                               (robust certificate)
   <==>
   ebar  <  epsilon*_rob := gamma/beta - delta_n .                                (robust boundary)
```

*The robust boundary is strictly inside the nominal one, `epsilon*_rob < epsilon*`,
by the margin `delta_n`; the nominal-to-robust gap*

```
   epsilon* - epsilon*_rob = delta_n = O( 1/sqrt(n) )  ->  0      as  n -> inf ,
```

*so the robust boundary closes on the nominal one at the parametric rate.
Symmetrically, a configuration is robustly-**unstable** iff `ebar - delta_n >
gamma/beta`; the band `[gamma/beta - delta_n, gamma/beta + delta_n]` around the
nominal boundary is the "undecided" region where `n` episodes cannot yet call it.*

**Proof.** `m(epsilon') = epsilon' beta/gamma` is affine increasing, so
`sup_{U_n} m = (ebar + delta_n) beta/gamma`; requiring it `< 1` and solving gives
the certificate. `epsilon*_rob = epsilon* - delta_n < epsilon*` since `delta_n > 0`;
its limit follows from `delta_n = O(1/sqrt(n))` (§2). The undecided band is the set
of nominal `epsilon*` not separated from `ebar` by `delta_n`. [] 

This is exactly Cao & Shi's structural claim -- *the robust stable point lies inside
the nominal stable region and widens the margin from instability* -- made
quantitative here: the widening is precisely the confidence radius `delta_n`, and
it vanishes at `1/sqrt(n)`.

---

## 5. Sample complexity: resolving the phase transition

Theorem 1 inverts into a sample-complexity statement. Let `Delta := | epsilon* -
epsilon |` be the (unknown) distance of the truth to the boundary. To *decide*
stability (certify one side) at confidence `1 - a` we need the half-width below the
gap, `delta_n < Delta`, i.e. `z_{1-a} sigma_eps / sqrt(n) < Delta`:

```
   n_req( Delta )  =  ( z_{1-a} * sigma_eps / Delta )^2  =  O( Delta^{-2} ) .              (sample complexity)
```

Two honest consequences:

- **The transition is statistically hard.** As `Delta -> 0` (the truly critical
  regime), `n_req -> inf`: *no finite simulation resolves the exact crossing*. A
  single number "`epsilon* = 0.47`" is undefinable; only "stable / unstable /
  undecided at `n` episodes" is. This is the formal content of the methodology's
  "without this, a point estimate of `epsilon*` is not a statistically defensible
  claim."
- **The curve is plottable and falsifiable.** `n_req ~ Delta^{-2}` (inverse-square)
  and `delta_n ~ n^{-1/2}` (slope `-1/2` on log-log) are two concrete predictions
  the simulator must reproduce (§8); a measured slope far from `-1/2` falsifies the
  parametric-rate claim (and would flag a broken CRN coupling, §2.2).

---

## 6. Corollaries

### 6.1 Robust *systemic* boundary (compose with 1.3)
The certificate composes with the multi-dealer lift by replacing the nominal
boundary with 1.3's:

```
   robust-stable (N dealers)  <==>  ebar + delta_n  <  gamma / ( N_eff * beta ) ,
   epsilon*_rob,N = gamma/(N_eff*beta) - delta_n .
```

Since the nominal boundary is already a factor `N_eff` tighter (1.3), the *same*
absolute radius `delta_n` eats a **larger fraction** of the stable region as `N`
grows, and the distance-to-boundary `Delta_N = gamma/(N_eff beta) - epsilon`
shrinks with `N`, so by §5 the episodes needed to certify a competitive market,
`n_req ~ Delta_N^{-2}`, *increase* with the dealer count. **Systemic markets are
both more fragile and harder to certify** -- a sharpened systemic-risk message.
(The common-mode probe of 1.3 §10 supplies `m_N_hat`; its CRN in-phase
construction gives the same `1/sqrt(n)` radius by §2.3.)

### 6.2 Statistical vs. structural uncertainty ("off by 20%")
`delta_n` is **statistical** -- it shrinks with data. The methodology's separate
question "if our toxic-slope estimate is off by 20%, does the conclusion change?"
is **structural/model** uncertainty (misspecified `clients.py` intensity, wrong
`c_t`), a *fixed* relative ambiguity `eta_mod` that does **not** shrink with `n`.
The honest total ambiguity set is their union:

```
   U_n^total = [ ebar*(1 - eta_mod) - delta_n ,  ebar*(1 + eta_mod) + delta_n ] ,
   robust-stable  <==>  ebar*(1 + eta_mod) + delta_n  <  gamma/beta .
```

As `n -> inf`, `delta_n -> 0` but the `eta_mod` term remains: **more simulation
cannot buy away model misspecification.** Reporting `eta_mod` (a sensitivity
sweep, e.g. `+/-20%` on `c_t`/`I`) alongside `delta_n` is the complete UQ; conflating
them is the error to avoid.

### 6.3 The certificate is monotone and one-sided-correct
Because `m` is monotone in `epsilon`, the upper-confidence-bound rule (Theorem 1)
has *guaranteed* one-sided coverage: `P( declare stable | truly unstable ) <= a`.
The certificate never *over*-claims stability beyond the chosen `a` -- the correct
asymmetry for a risk statement (a false "stable" is the costly error).

---

## 7. Symbol -> config map

| Symbol | Meaning | Config / source | Status |
|--------|---------|-----------------|--------|
| `n` | episodes per estimate | `rrm.eval_episodes`, `policy.n_rollouts` | existing |
| `S` | seeds for cross-seed variance | experiment sweep (`--seed`) | existing pattern |
| `epsilon_hat_n`, `m_hat_n` | estimates | `response_modulus.py` (BR-slope) | existing |
| `sigma_eps` | per-episode estimator std | measured cross-seed (§3.1) | derived |
| `delta_n` | ambiguity radius | `z*sigma_eps/sqrt(n)` (§2-§3) | derived |
| `epsilon*_rob` | robust boundary | `gamma/beta - delta_n` (§4) | derived |
| `n_req` | episodes to resolve `Delta` | `(z*sigma_eps/Delta)^2` (§5) | derived |
| `lambda` | Sinkhorn regularisation | proposed Sinkhorn estimator | **proposed** |
| `eta_mod` | structural relative ambiguity | sensitivity sweep on `c_t`, `I` (§6.2) | **proposed** |

The only new machinery is a **cross-seed variance harness** around the *existing*
probe (no new estimator required for the headline `epsilon`-ball) and, optionally,
the Sinkhorn estimator (`lambda`) for the OT route. `gamma`, `beta`, `epsilon` are
1.1's closed forms.

---

## 8. Validation protocol and the remaining code task

**Predict (no new math).** From 1.1's closed forms compute `epsilon`, `epsilon* =
gamma/beta`, and hence `Delta`; predict `delta_n = z*sigma_eps/sqrt(n)`,
`epsilon*_rob = epsilon* - delta_n`, and `n_req = (z*sigma_eps/Delta)^2`.

**Measure.** For each `n` in a geometric grid and `S` seeds, run
[`measure_response_modulus`](../../archive/endo_market_v2/endo_market/analysis/response_modulus.py),
form `ebar`, `s_eps`, and `delta_n = z_{1-a} s_eps` (§3.1). Repeat across the
`sweep_feedback.yaml` grid.

**Compare -- three falsifiable checks.**
1. **Parametric rate:** `s_eps` vs `n` on log-log has slope `-1/2` (§2.3). A slope
   near `-1/3` would reveal the CRN coupling is broken (§2.2) -- a built-in
   diagnostic of the probe's variance reduction.
2. **Robust boundary:** the certificate `ebar + delta_n < gamma/beta` flips exactly
   at `epsilon*_rob`, and the nominal-vs-robust gap collapses as `1/sqrt(n)` (§4).
3. **Sample complexity:** the episodes to first certify a configuration a distance
   `Delta` from the boundary track `n_req ~ Delta^{-2}` (§5).

Report the phase diagram with the nominal boundary, the robust boundary, and the
**undecided band** shaded -- the median+IQR-with-error-bars figure the methodology
requires (and the ICAIF "uncertainty bands across seeds" line item).

**Code (DONE).** Implemented in
[`analysis/robust_boundary.py`](../../archive/endo_market_v2/endo_market/analysis/robust_boundary.py):
`empirical_radius` fits the cross-seed radius `delta_n = z*s`; `robust_certificate`
returns the `(mean, delta_n, epsilon*_rob, verdict in {stable, unstable, undecided})`
with the structural floor `eta_mod` (§6.2); `sample_complexity` gives
`n_req = (z*sigma/Delta)^2` (§5); `loglog_rate` / `rate_check` emit the
`slope ~ -1/2` diagnostic (§2.3); `finite_sample_radius` is the sub-Gaussian variant
(§2.4); and `robust_boundary` wraps the existing `measure_response_modulus` probe in
the cross-seed loop, reporting both modulus- and `epsilon`-space certificates.
Verified by `tests/test_robust_boundary.py`. Only the optional Sinkhorn OT-route
estimator (`analysis/sinkhorn_epsilon.py`, feeding 1.1's triangulation) is deferred;
no change to the market model was needed.

---

## 9. Honest caveats (for the paper's limitations paragraph)

- **The `1/sqrt(n)` rate is conditional on CRN / fixed-`lambda` Sinkhorn (§2.3, §3.2).**
  It is a property of the *estimator*, not a law of nature: a naive independent-noise
  probe gives only `n^{-1/3}` (§2.2), and raw empirical `W_1` gives `n^{-1/d}`. The
  claim is stated with the estimator attached; the log-log slope check (§8) is its
  falsification test.
- **Sub-Gaussianity assumes the cap is slack (A4).** Deep past the boundary the
  informed-flow cap engages and the IPA derivative can be heavy-tailed; the
  finite-sample certificate (`**`) then needs a Bernstein/median-of-means variant.
  The radius is trustworthy up to and slightly past the boundary -- the region of
  interest -- not arbitrarily far into divergence.
- **Cross-seed variance conflates estimator noise with seed-to-seed regime drift.**
  `s_eps` is a clean estimate of the sampling std only if `gamma`, `beta`, the
  reference state, and `h*` are held fixed across seeds; otherwise it also absorbs
  reference-state dispersion. Report `delta_n` as a per-reference-state quantity, as
  in 1.1 §9, and separate the two sources.
- **Structural ambiguity `eta_mod` does not shrink (§6.2).** The `O(1/sqrt(n))`
  result is about *statistical* uncertainty only. A confidently-stated `epsilon*_rob`
  is still conditional on the `clients.py` intensity being the right model; the
  sensitivity sweep on `(c_t, I)` must accompany it or the robustness is illusory.
- **Gaussian/`z` vs. finite-sample `sqrt(2 log(2/a))`.** The two radii (`*`)/(`**`)
  agree in rate but differ in constant; use the finite-sample one for any hard
  certificate and the CLT one only for large-`n` reporting.

---

## References

- T. Cao, R. Shi. *Distributionally Robust Performative Prediction.*
  arXiv:2206.01844, 2022. -- the minimax stable point / ambiguity-set formulation;
  the robust stable point lies inside the nominal region and widens the margin
  (Theorem 1). (Lit. #13.)
- M. Cuturi, G. Peyré. *Computational Optimal Transport.* 2019 (arXiv:1803.00567),
  Ch. 7. -- entropic/Sinkhorn estimation and its `O(1/sqrt(n))` sample complexity
  (with the `O(lambda)` bias), realising the Wasserstein-ball radius of §3.2. (Lit. #14.)
- J. Perdomo, T. Zrnic, C. Mendler-Dünner, M. Hardt. *Performative Prediction.*
  ICML 2020. -- `epsilon` as the `W_1`-Lipschitz sensitivity whose finite-sample
  estimation this document quantifies.
- Supporting concentration/estimation: sub-Gaussian (Hoeffding) tail for (`**`);
  the delta method / M-estimator CLT for (`*`); Genevay et al. and Mena & Niles-Weed
  for the entropic-OT parametric rate. See [`../references.bib`](../references.bib).
- Builds on [`01-analytic-stability-boundary.md`](01-analytic-stability-boundary.md)
  (closed-form `gamma`, `beta`, `epsilon`, and the CRN probe) and
  [`03-multi-dealer-systemic-risk.md`](03-multi-dealer-systemic-risk.md) (the robust
  systemic boundary of §6.1); see `literature/literature-raghav/README.md`
  (papers #13, #14) for the full citation map.
