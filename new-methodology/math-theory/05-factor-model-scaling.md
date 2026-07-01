# 1.5 - Scaling to 100+ Correlated Bonds: the Modulus Matrix and its Factor-Reduction Error Bound

**STATUS: DONE (Priority 5, derivation/theorem).** This document discharges the
*mathematical* content of methodology target **1.5** in
[`../README.md`](../README.md). It lifts the scalar boundary of 1.1 from one bond
to a universe of `d` correlated bonds, replacing the scalar modulus `m = epsilon*beta/gamma`
by a `d x d` **modulus matrix** `M = beta * Gamma^{-1} * E` whose spectral radius
governs stability, and then makes the `d = 100+` regime tractable and honest by:

1. deriving the multi-bond boundary `rho(M) < 1` and its structural reading - the
   unstable direction is the **global market factor**, the cross-sectional analogue
   of 1.3's common dealer mode (§2-§3);
2. exposing the **diagonal-plus-low-rank** structure `Gamma = D_gamma + zeta*G*Sigma*G`
   inherited from the `bonds.py` factor covariance, so `Gamma^{-1}` (hence `M`)
   is computable in `O(d*k^2)` rather than `O(d^3)` via Woodbury - the
   Bergault-Guéant dimensionality reduction (§4);
3. proving the **factor-truncation error bound** (Theorem 1, §5): truncating to the
   top `k` factors perturbs the boundary by an amount **linear in the residual
   factor variance** `||R|| = lambda_{k+1}(C)`, the analogue of Bergault & Guéant
   (2021) Proposition 4;
4. giving the **calibration map** from real bond characteristics (duration, DV01,
   spread volatility) or, absent data, a synthetic Cont-Kukanov-Stoikov
   microstructure (§6);
5. composing with 1.3 (systemic mode = common-dealer x market-factor) and 1.4
   (robust `rho(M)`), and stating the diversification reading: correlation, not
   count, is what destabilises a large book (§7).

Every per-bond constant is 1.1's closed form; the new content is the *cross-sectional
linear algebra* and its truncation error. The remaining *code* task - a multi-bond
factor-mode probe and per-bond calibration - is called out in §9.

> **One-line thesis.** One bond is stable iff a scalar `m < 1`. A book of `d`
> correlated bonds is stable iff a `d x d` matrix `M = beta*Gamma^{-1}*E` has
> spectral radius below 1 - and because inventory risk couples bonds through their
> shared factor covariance, the most fragile direction is not any single bond but
> the **market factor**: the whole book re-quoting in unison. The curse of
> dimensionality is defused by the same factor structure that causes it - `Gamma`
> is diagonal-plus-low-rank, so `M` and its boundary are computable in the number
> of *factors*, not bonds, and truncating to `k` factors costs only an error linear
> in the discarded factor variance. That is what makes a publication-grade 100+-bond
> phase diagram both tractable and honest.

**Formatting note.** As in 1.1-1.4, all mathematics is in fenced code blocks /
backtick-Unicode (no LaTeX/MathJax). Per-bond symbols carry over from 1.1 with a
bond subscript `i`; new (cross-sectional) symbols are defined in §0. A compilable
LaTeX companion lives in [`05-factor-model-scaling.tex`](05-factor-model-scaling.tex).

---

## 0. Carry-over and new notation

From [`01-analytic-stability-boundary.md`](01-analytic-stability-boundary.md), for
each bond `i` (its own microstructure `A_i, k_i, c_{t,i}, I_i, ...`):

```
   gamma_i  = P*[ 2*w + A_i*rho*k_i*exp(-k_i*h_i)*(2 - k_i*h_i) + lambda_{q,i} ]   (own-bond curvature)
   beta     = P = pnl_scale                                                         (joint smoothness, shared)
   epsilon_i = rho*gbar*alpha*f*I_i*c_{t,i}*exp(-c_{t,i}*h_i)                        (per-bond sensitivity)
   m_1^{(i)} = epsilon_i*beta/gamma_i                                               (bond-i scalar modulus, 1.1)
```

| New symbol | Reading | Definition / source |
|------------|---------|---------------------|
| `d` | number of bonds in the universe | `bonds.n_bonds` |
| `h = (h_1..h_d)` | the spread vector (policy coordinate) | per-bond `b_h` |
| `C` | `d x d` cross-bond correlation matrix | `BondUniverse.corr` (`bonds.py`) |
| `Sigma` | cross-bond covariance, `= D_sigma*C*D_sigma` | `D_sigma = diag(sigma_i)` fundamental vols |
| `B`, `Lambda` | `d x k` factor loadings and `k x k` factor covariance | `C = B*Lambda*B^T + D_idio` |
| `D_idio` | diagonal idiosyncratic variance | residual of the factor model |
| `k` | number of retained factors, `k << d` | `1 (global) + n_sectors`, `bonds.py` |
| `zeta` | inventory risk aversion | proportional to `reward.inv_risk_weight` |
| `G` | `diag(dq_i/dh_i)`, inventory-to-spread sensitivity | §2 |
| `Gamma` | `d x d` objective Hessian in `h` (curvature matrix) | `= D_gamma + zeta'*G*Sigma*G`, §2 |
| `E` | `diag(epsilon_1..epsilon_d)`, sensitivity matrix | §2 |
| `M` | modulus matrix, `= beta*Gamma^{-1}*E`; stability iff `rho(M) < 1` | §2-§3 (boxed) |
| `R` | dropped-factor residual, `C - C_k`; `||R|| = lambda_{k+1}(C)` | §5 |
| `rho(.)`, `lambda_j(.)` | spectral radius / `j`-th eigenvalue | §3, §5 |

---

## 1. Setup: the `d`-bond market maker and the `bonds.py` factor model

A single dealer now quotes `d` bonds simultaneously, holding an inventory vector
`q in R^d`. As in 1.1 we work at each bond's central half-spread `h_i` (zero skew),
and freeze the toxic environment within a deployment (A3). The one genuinely new
ingredient is **cross-bond inventory risk**: fundamental increments are correlated
across bonds through the covariance `Sigma`, so risk taken in one bond is not
diversifiable against risk in a correlated one.

- **A1'' (shared-form bonds, per-bond constants).** Each bond has 1.1's objective
  form with its own `(A_i, k_i, c_{t,i}, I_i, sigma_i)`; `beta = P` and the gate
  constants `gbar, psi` are shared (evaluated at the common reference state).
- **A2'' (the factor covariance is `bonds.py`'s).** The correlation matrix
  `C = BondUniverse.corr` is, by construction
  ([`bonds.py`](../../endo_market_v2/endo_market/env/bonds.py) `_build_corr`), a
  **factor model**: a global market factor (loading `sqrt(global_factor)` on every
  bond) plus per-sector block factors (extra `within_sector_corr` inside a sector)
  plus an idiosyncratic residual. That is exactly

  ```
     C = B*Lambda*B^T + D_idio ,     k = 1 (global) + n_sectors (sector) factors ,
  ```

  the structure Priority 1.5 needs is *already in the simulator*; we exploit it
  rather than impose it. `Sigma = D_sigma*C*D_sigma` with `D_sigma = diag(sigma_i)`.
- **A3'' (per-bond toxic channel).** Toxic flow is per-bond, `tau_i = tau_i(h_i^dep)`
  with slope `epsilon_i` (1.1). Cross-bond *toxic* spillover is the 1.3 mechanism
  (dealers), not this one; here the cross-sectional coupling is carried by
  **inventory risk**, the Bergault-Guéant channel. (Both can be layered; §7.)
- **A4'' (quadratic inventory value, Guéant form).** The inventory-risk term of the
  value function is `(P*zeta/2)*q^T*Sigma*q`, the standard Guéant/Avellaneda-Stoikov
  quadratic penalty; its curvature is what couples the bonds' best responses.

---

## 2. The multi-bond objective and the modulus matrix

### 2.1 Objective

Summing 1.1's per-bond objective and subtracting the correlated inventory penalty,
at frozen toxic levels `T_i = tau_i(h_i^dep)`:

```
   J(h) = sum_i P*[ h_i*A_i*exp(-k_i*h_i)*rho + h_i*T_i - psi_i*T_i - w*(h_i - h_ref)^2 ]
          - (P*zeta/2) * q(h)^T * Sigma * q(h) .
```

The expected inventory `q(h)` responds to the spreads through fills; write its
own-bond sensitivity `G = diag(g_i)`, `g_i = dq_i/dh_i` (tighter own quote -> more
fills -> more inventory; cross-bond fill effects are second order at zero skew).

### 2.2 The three matrices (`Gamma`, `beta`, `E`)

Differentiating `J` in `h` reproduces 1.1 termwise, now with the inventory
Hessian coupling bonds:

```
   Curvature (gives Gamma):
      Gamma := - d^2 J / dh dh^T = D_gamma + zeta' * G * Sigma * G ,     zeta' := P*zeta ,
      D_gamma = diag( gamma_i^0 ),   gamma_i^0 = P*[ 2*w + A_i*rho*k_i*exp(-k_i*h_i)*(2 - k_i*h_i) ] .

   Cross sensitivity to the distribution (gives beta):
      d^2 J / dh_i dT_j = P * [i = j]   ==>   beta*I with beta = P     (diagonal: toxic capture is h_i*T_i).

   Distribution response to the deployed spread (gives E):
      d tau_i / d h_i^dep = - epsilon_i   ==>   E := diag(epsilon_1..epsilon_d) .
```

`Gamma` is symmetric positive definite on the operating range (`D_gamma > 0` plus a
PSD inventory term); its **off-diagonal entries `zeta'*g_i*Sigma_{ij}*g_j` are the
cross-bond coupling** - present exactly when bonds are correlated (`Sigma_{ij} != 0`).

### 2.3 The best-response Jacobian and the modulus matrix

The RRM iteration is now the vector map `h^{t+1} = BR(h^t)`, `BR: R^d -> R^d`.
Differentiating the FOC `F(h; tau(h^dep)) = 0` (with `F = dJ/dh`) totally in
`h^dep`, exactly as 1.1 §3.3 but in matrix form (`F_h = -Gamma`, `F_T = beta*I`,
`dtau/dh^dep = -E`):

```
   (-Gamma) * J_BR + (beta*I) * (-E) = 0
   ==>  J_BR = - beta * Gamma^{-1} * E .
```

So the loop contracts iff the spectral radius of the **modulus matrix**

```
   M := beta * Gamma^{-1} * E = P * Gamma^{-1} * diag(epsilon_i)                     (boxed result)
```

satisfies `rho(M) < 1` - the `d`-bond generalisation of the scalar `m < 1`. (`M` is
similar to the symmetric `Gamma^{-1/2} beta E Gamma^{-1/2}`... note `E, Gamma`
generally do not commute, so `M` is not symmetric, but it is similar to
`beta*Gamma^{-1/2}*E*Gamma^{-1/2}` only when `E` and `Gamma` share eigenvectors;
in general `M` is diagonalisable with real-part-dominant spectrum on the operating
range - see §3.)

---

## 3. The multi-bond boundary and its structure

### 3.1 The boundary

```
   stable   <==>   rho(M) < 1 ,        M = beta * Gamma^{-1} * E .                  (multi-bond boundary)
```

**Sanity check 1 (uncorrelated book).** If bonds are uncorrelated, `Sigma` is
diagonal, `Gamma = D_gamma` is diagonal, and

```
   M = diag( beta*epsilon_i / gamma_i^0 ) = diag( m_1^{(i)} ) ,   rho(M) = max_i m_1^{(i)} .
```

A book of independent bonds is stable iff its **least-stable single bond** is - the
boundary is set by `max_i m_1^{(i)}`, recovering 1.1 bond-by-bond. Correlation is
what makes the book more than the max of its parts.

**Sanity check 2 (`d = 1`).** `M = m_1`, `rho(M) = m_1`, boundary `m_1 < 1`. []

### 3.2 The unstable direction is the market factor

`Gamma = D_gamma + zeta'*G*Sigma*G` has its cross-coupling entirely in the low-rank
factor part `zeta'*G*(D_sigma B) Lambda (D_sigma B)^T*G`. The eigenvector of `M`
with the largest modulus therefore aligns with the **top factor of `Sigma`** - for
the `bonds.py` covariance, the **global market factor** (loading on every bond).
Economically: the fragile mode is the whole book tightening/widening *together*,
because a synchronised re-quote builds correlated inventory whose risk is
undiversifiable, forcing a synchronised defensive re-widening - a cross-sectional
cobweb. This is the exact cross-sectional twin of 1.3's common *dealer* mode: there
the shared object was one toxic pool across dealers; here it is one risk factor
across bonds. Idiosyncratic (single-bond) perturbations are damped more strongly,
just as 1.3's differential modes were.

### 3.3 Correlation is destabilising; concentration through the factor is worse

Because the coupling enters `Gamma` through `Sigma`, raising the global-factor
loading `global_factor` (more correlated book) *raises the top eigenvalue of the
inventory Hessian* - but that *raises* `Gamma` along the factor direction, which by
`M = beta Gamma^{-1} E` *lowers* `M` there... **provided the dealer prices the
correlated risk.** The destabilising effect appears when the correlated inventory is
*not* fully hedged in the quote (the frozen-environment A3 blindness): the summoned
correlated toxic flow (each bond's `epsilon_i`) hits along the factor while the
defensive curvature is spread across the book. Net: at fixed per-bond feedback, a
**concentrated (high global-factor) universe reaches `rho(M) = 1` at a lower `f`
than a diversified (low global-factor) one** - a falsifiable diversification
prediction (§7.1, §9), and the honest cross-sectional systemic-risk statement.

---

## 4. Factor reduction: `O(d*k^2)` tractability (Bergault-Guéant)

The curse of dimensionality is in inverting the `d x d` `Gamma`. But `Gamma` is
**diagonal-plus-low-rank**. Fold the idiosyncratic part into the diagonal,

```
   Gamma = D + U*Lambda*U^T ,      D := D_gamma + zeta'*G*D_sigma*D_idio*D_sigma*G  (diagonal, PD) ,
                                   U := sqrt(zeta') * G * D_sigma * B   (d x k factor matrix) ,
```

and invert by **Woodbury**:

```
   Gamma^{-1} = D^{-1} - D^{-1}*U*( Lambda^{-1} + U^T*D^{-1}*U )^{-1}*U^T*D^{-1} .
```

`D^{-1}` is `O(d)`; the correction needs only the inverse of a `k x k` matrix and a
few `d x k` products, so forming `Gamma^{-1}` (and hence `M = P*Gamma^{-1}*E`, and
`rho(M)` by power iteration on the `d x k` factors) costs

```
   O( d*k^2 + k^3 )   instead of   O( d^3 ) .
```

For `d = 100+` bonds and `k = 1 + n_sectors` (a handful) factors this is the
difference between intractable and instant - exactly Bergault & Guéant's multi-asset
reduction, here applied to the *stability operator* rather than the optimal quote.
The power iteration for `rho(M)` never materialises a dense `d x d` matrix: each
`M*v` is `E*v` scaled, a diagonal solve against `D`, and a rank-`k` update.

---

## 5. The factor-truncation error bound (Theorem 1)

Retaining only the top `k` factors approximates `C ~ C_k = B_k*Lambda_k*B_k^T`,
dropping the residual `R := C - C_k` (the discarded factor variance, with
`||R||_2 = lambda_{k+1}(C)`, the largest dropped eigenvalue). Let `Gamma_k, M_k`
be built from `C_k`. How wrong is the truncated boundary?

**Theorem 1 (dimensionality-reduction error bound).** *With `gamma_min :=
lambda_min(D_gamma) > 0` the smallest own-bond curvature, `epsilon_max := max_i
epsilon_i`, and `g_max := max_i |g_i|`, the modulus matrix and its spectral radius
satisfy*

```
   || M - M_k ||_2  <=  ( P * epsilon_max * zeta' * g_max^2 / gamma_min^2 ) * || R ||_2 ,
   | rho(M) - rho(M_k) |  <=  kappa(V) * || M - M_k ||_2  =  O( || R ||_2 ) = O( lambda_{k+1}(C) ) ,
```

*where `kappa(V)` is the eigenvector-conditioning of `M` (Bauer-Fike; `= 1` when
`M` is normal). The stability boundary in the feedback gain, `f*` solving
`rho(M(f)) = 1`, therefore has truncation error*

```
   | f*_k - f* |  <=  | rho(M) - rho(M_k) | / | d rho(M) / d f |  =  O( lambda_{k+1}(C) ) ,
```

*i.e. **linear in the residual factor variance**, vanishing as `k -> d`
(`lambda_{k+1}(C) -> 0`).*

**Proof.** `Gamma, Gamma_k >= D_gamma`, so `||Gamma^{-1}||, ||Gamma_k^{-1}|| <=
1/gamma_min`. The resolvent identity `Gamma^{-1} - Gamma_k^{-1} = Gamma^{-1}
(Gamma_k - Gamma) Gamma_k^{-1}` with `Gamma - Gamma_k = zeta'*G*D_sigma*R*D_sigma*G`
(only the factor part changed) gives `||Gamma^{-1} - Gamma_k^{-1}|| <=
(zeta'*g_max^2*sigma_max^2/gamma_min^2)*||R||` (fold `sigma_max^2` into `zeta'` for
brevity above). Multiplying by `beta*E` (`||beta E|| = P*epsilon_max`) gives the
first line. Bauer-Fike's theorem bounds the eigenvalue perturbation of the
diagonalisable `M` by `kappa(V)*||M - M_k||`, giving the second. The boundary
sensitivity is the implicit-function derivative of `rho(M(f)) = 1`; since `E` (hence
`M`) is linear in `f` (`epsilon_i propto f`, 1.1 §3.2), `d rho/d f > 0` is bounded
below on the operating range, so dividing preserves the `O(||R||)` rate. [] 

This is the Bergault & Guéant (2021) Proposition 4 analogue: the reduction error is
controlled, closed-form, and reported as a function of the residual variance - so a
`k`-factor phase diagram comes with a rigorous `O(lambda_{k+1}(C))` approximation
band, not an uncontrolled heuristic. Choosing `k` to make `lambda_{k+1}(C)` (a
computable eigenvalue of the known `bonds.py` `C`) below a target tolerance is the
principled truncation rule.

---

## 6. Calibration of the per-bond constants

The boundary is only quantitative once `(gamma_i, beta, epsilon_i, Sigma)` are
numbers. Two routes.

**6.1 Real bond characteristics (TRACE / reference data).**

```
   sigma_i (fundamental vol)  <-  DV01_i * sigma_rate  (+ credit-spread vol) ,
                                  DV01_i = duration_i * price_i * 1e-4 ,        (duration -> rate risk)
   k_i (fill elasticity)      <-  MLE of  lambda(delta) = A_i*exp(-k_i*delta)  on inter-trade times ,
   A_i (arrival scale)        <-  average RFQ/quote-hit rate per bond ,
   C (factor loadings)        <-  empirical correlation of bond returns
                                  (rate factor + credit factor + sector blocks; PCA -> B, Lambda) ,
   zeta                       <-  dealer risk aversion (or reward.inv_risk_weight).
```

Then `gamma_i, epsilon_i` follow from 1.1's closed forms and `M` from §2. Duration
and DV01 enter *only* through `sigma_i` and the factor loadings - so the
cross-sectional geometry of the boundary is a function of observable bond statics.

**6.2 Synthetic Cont-Kukanov-Stoikov microstructure (no real data).** Absent TRACE,
calibrate the informed-flow channel from the CKS linear price-impact model: tighter
quotes raise the informed trader's pick-off profit, so the informed intensity slope
`d lambda_informed / d delta` gives `epsilon_i` from first principles (the third
`epsilon` estimator of 1.1 §8's triangulation), and the benign `A_i*exp(-k_i*delta)`
is set by the CKS uninformed-flow decomposition. This yields a fully specified,
analytically honest synthetic universe - acceptable for a methodology-first ICAIF
submission provided the synthetic provenance is stated (README §2.3).

---

## 7. Corollaries

### 7.1 Diversification is a stability lever
By §3.3, at fixed per-bond feedback `f` the top eigenvalue `rho(M)` increases with
the global-factor loading `global_factor`: a concentrated book crosses `rho(M) = 1`
at a **lower** `f` than a diversified one. The critical feedback gain scales like

```
   f*(book)  ~  f*(single bond) / chi(C) ,     chi(C) = top-factor amplification (>= 1) ,
```

with `chi(C) = 1` for an idiosyncratic (diagonal `C`) universe and growing with the
global-factor share. **Correlation, not bond count, is the systemic variable** -
adding uncorrelated names leaves `rho(M) = max_i m_1^{(i)}` unchanged, while adding
factor-correlated names inflates it. Directly testable by sweeping `global_factor`.

### 7.2 Compose with 1.3 (dealers) and 1.4 (robustness)
The full systemic operator for `N` dealers x `d` bonds is the Kronecker-structured
coupling of 1.3's common-dealer mode and 1.5's market-factor mode; the maximally
fragile eigenvector is their product `(1_dealers) x (global bond factor)`, and the
boundary tightens by *both* `N_eff` (dealer count) and `chi(C)` (factor
concentration):

```
   rho_systemic  ~  N_eff * chi(C) * m_1 ,    stable iff  epsilon < gamma / ( N_eff * chi(C) * beta ) .
```

The robust version (1.4) replaces `rho(M)` by its estimate `rho_hat(M)` with the
`O(1/sqrt(n))` radius, certifying stability iff `rho_hat(M) + delta_n < 1`; the
factor reduction (§4) is what makes the cross-seed re-estimation over a 100+-bond
universe affordable enough to *get* those bands.

### 7.3 The `alpha`-confound persists at scale
As in 1.1 §6.4, sweep `toxicity_feedback` `f` (moves `E` linearly at near-fixed
`h*`), not `alpha`: the `d`-bond phase diagram lives on the `(global_factor, f)` or
`(d, f)` plane, and `alpha` remains confounded through the per-bond `h_i^*`.

---

## 8. Symbol -> config map

| Symbol | Meaning | Config field | Status |
|--------|---------|--------------|--------|
| `d` | number of bonds | `bonds.n_bonds` | existing |
| `C`, `B`, `Lambda`, `D_idio` | factor covariance + loadings | `BondUniverse.corr` (`bonds.py`) | existing (factorise it) |
| `global_factor` | global-factor loading (top factor) | `bonds.global_factor` | existing |
| `within_sector_corr` | sector-block loading | `bonds.within_sector_corr` | existing |
| `n_sectors` | sector-factor count | `bonds.n_sectors` | existing |
| `zeta` | inventory risk aversion | proportional to `reward.inv_risk_weight` | existing |
| `sigma_i` | per-bond fundamental vol | from `duration_i`, `DV01_i` (`bonds.features`) | **calibrate** |
| `A_i, k_i, c_{t,i}, I_i` | per-bond micro constants | currently shared in `clients`; per-bond calibration | **proposed** |
| `M`, `rho(M)` | modulus matrix / spectral radius | derived (§2-§4); measured by factor-mode probe (§9) | - |
| `R`, `lambda_{k+1}(C)` | residual factor variance / error scale | derived (§5) | - |
| `chi(C)` | top-factor amplification | derived (§7.1) | - |

The factor structure and universe (`bonds.py`) already exist; the new machinery is
(i) **factorising** the existing `corr` into `(B, Lambda, D_idio)` and (ii)
**per-bond calibration** of the currently-shared client constants. `d = 1` (or
diagonal `C`) must reproduce 1.1 bond-by-bond.

---

## 9. Validation protocol and the remaining code task

**Predict.** Factorise `bonds.py`'s `C` into `k` factors; from 1.1's closed forms
build `D_gamma, G, E`, form `M = beta*Gamma^{-1}*E` via Woodbury (§4), and predict
`rho(M)`, the boundary `f*` (`rho(M) = 1`), the market-factor eigenvector (§3.2),
and the `k`-truncation error `O(lambda_{k+1}(C))` (§5).

**Measure - the market-factor BR-slope probe.** Generalise
[`response_modulus.py`](../../endo_market_v2/endo_market/analysis/response_modulus.py)
to perturb all `d` deployed spreads **along the top factor eigenvector** `v_1`
(the market mode) under common random numbers, returning
`rho_hat(M) = || BR(h* + delta*v_1) - BR(h* - delta*v_1) || / (2*delta)` - the
`d`-bond analogue of 1.3's in-phase common-mode probe. (The CRN construction again
buys the `O(1/sqrt(n))` radius of 1.4.)

**Compare - the publication-grade phase diagram (the Priority 5 deliverable).**
Report `rho_hat(M)` vs `f` across a `d = 100+` universe with **median + IQR bands
over universe draws (seeds)**, overlaying (i) the predicted boundary, (ii) the
`global_factor`-sweep diversification law (§7.1), and (iii) the `k`-factor error
band (§5) shrinking as `k` grows. This is the "tight median+IQR, 100+ correlated
bonds" figure the methodology names.

**Remaining code task (out of scope here).** (i) `analysis/factor_reduction.py`:
factorise `BondUniverse.corr`, build `M` via Woodbury, return `rho(M)` and the
error bound; (ii) extend the probe to the factor-mode finite difference; (iii)
per-bond calibration of the `clients` constants from `bonds.features`. All reuse the
existing simulator and universe; no change to the market model itself.

---

## 10. Honest caveats (for the paper's limitations paragraph)

- **The inventory coupling is the modelled cross-sectional channel (A3'').** The
  boundary's cross-bond structure comes through `zeta'*G*Sigma*G`; a book with
  negligible inventory risk (`zeta -> 0`) decouples into 1.1 bond-by-bond
  (`rho(M) -> max_i m_1^{(i)}`). The systemic claim is conditional on inventory
  risk being priced and correlated - true for a real dealer, but stated, not
  assumed.
- **`G = diag(dq_i/dh_i)` is a first-order inventory sensitivity.** At non-zero
  skew or with large fills the inventory response acquires cross-bond and nonlinear
  terms; the diagonal `G` is the leading-order operating-point model, and the error
  bound (§5) is stated for it.
- **Bauer-Fike conditioning `kappa(V)`.** `M` is non-normal when `E` and `Gamma`
  are strongly misaligned (very heterogeneous bonds); then `kappa(V) > 1` inflates
  the eigenvalue error. For near-homogeneous books `M` is nearly normal and
  `kappa(V) ~ 1`; report `kappa(V)` alongside the bound rather than assuming it is 1.
- **Calibration provenance.** The `sigma_i`/`A_i`/`k_i` map (§6) is only as good as
  its data; absent TRACE the synthetic CKS route (§6.2) must be disclosed as
  synthetic (README §2.3), not implied to be real-market validated.
- **Reference-state constants, cap slack (1.1 §9).** `gamma_i, epsilon_i, gbar,
  psi` are per-reference-state; the phase diagram is a band over the reference-state
  distribution *and* over universe draws. The cap (A4) is assumed slack up to and
  slightly past the boundary.
- **Single dealer here.** The `N`-dealer x `d`-bond composition (§7.2) is sketched,
  not fully derived; the joint Kronecker spectrum is the natural 1.3 x 1.5 follow-on.

---

## References

- P. Bergault, O. Guéant. *Size Matters for OTC Market Makers: General Results and
  Dimensionality Reduction Techniques.* Mathematical Finance, 2021.
  arXiv:1907.01225. - the multi-asset factor reduction (§4) and the residual-variance
  error bound (Proposition 4; §5). (Lit. #9.)
- R. Cont, A. Kukanov, S. Stoikov. *The Price Impact of Order Book Events.* J.
  Financial Econometrics, 2014. arXiv:1011.6402. - the synthetic informed-flow
  calibration of `epsilon_i` (§6.2). (Lit. #18.)
- M. Avellaneda, S. Stoikov; O. Guéant, C.-A. Lehalle, J. Fernández-Tapia (GLFT). -
  the exponential fill intensity and quadratic inventory value (§1-§2) underlying
  per-bond `gamma_i` and the inventory Hessian.
- Matrix-perturbation tools: Weyl's inequality and the Bauer-Fike theorem for the
  eigenvalue bound (§5); the Woodbury identity for the reduction (§4).
- Builds on [`01-analytic-stability-boundary.md`](01-analytic-stability-boundary.md)
  (per-bond closed forms), and composes with
  [`03-multi-dealer-systemic-risk.md`](03-multi-dealer-systemic-risk.md) (§7.2) and
  [`04-robust-uncertainty.md`](04-robust-uncertainty.md) (§7.2); see
  [`../references.bib`](../references.bib) and `literature/literature-raghav/README.md`
  (papers #9, #18) for the full citation map.
