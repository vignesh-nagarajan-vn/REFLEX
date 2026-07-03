# 1.3 - Multi-Dealer Competition and Systemic Performative Instability

**STATUS: DONE (Priority 3, derivation/theorem + code).** This document discharges
the *mathematical* content of methodology target **1.3** in
[`../README.md`](../README.md). It lifts the single-dealer closed forms of
[`01-analytic-stability-boundary.md`](01-analytic-stability-boundary.md) to `N`
competing dealers who share a common informed-flow pool, and derives:

1. the **performatively stable Nash equilibrium (PSNE) boundary** `epsilon < gamma/(N*beta)`
   for `N` symmetric, fully-coupled dealers (Theorem 1, §4) -- the target theorem;
2. its honest interpolation `epsilon < gamma/(N_eff*beta)` with
   `N_eff = 1 + kappa*(N-1)` for partial toxic spillover `kappa in [0,1]` (so
   `kappa=0` recovers `N` independent single-dealer loops and `kappa=1` gives the
   headline `N`-boundary);
3. **existence and uniqueness** of the PSNE (§5);
4. **linear joint-RRM convergence** at rate `m_N = N_eff*epsilon*beta/gamma`,
   with the `O(1/k)` stochastic rate under joint strong monotonicity `gamma_joint
   = gamma - N_eff*epsilon*beta > 0` (§6, after Narang et al. 2022);
5. the **mean-field `N -> inf` limit** under both strong (`kappa` fixed) and
   mean-field (`kappa = c/N`) coupling (§7, after Lacker 2016);
6. the **systemic-risk reading**: instability is carried by the *common mode*
   (all dealers tightening together), with a critical dealer count `N_c =
   gamma/(beta*epsilon)` past which individually-rational quoting destabilises the
   whole market (§8).

The single-dealer modulus `m_1 = epsilon*beta/gamma` of 1.1 is the building block;
*every* constant below is one of 1.1's closed forms, so this lift requires **no new
estimation** -- only the `N`-dealer linear algebra. The companion *code* --
[`analysis/multi_dealer_modulus.py`](../../endo_market_v2/endo_market/analysis/multi_dealer_modulus.py),
the closed-form boundary, the joint cobweb, and the common-mode / anti-phase probes --
is now implemented (§10).

> **One-line thesis.** A single dealer is stable iff `m_1 = epsilon*beta/gamma < 1`.
> Put `N` of them in a market that shares one informed-flow pool and the
> destabilising feedback no longer acts through each dealer's *own* re-tightening
> alone: when *any* dealer tightens it summons toxic flow that picks off *all* of
> them, so the joint best-response map is a rank-one common-mode coupling whose
> unstable eigenvalue is `N` times larger. The market crosses into instability at
> `epsilon = gamma/(N*beta)` -- a factor `N` *before* any individual dealer would.
> Competition manufactures a synchronised cobweb that is more fragile than any
> dealer in isolation. That is the systemic-risk result.

**Formatting note.** As in 1.1 and 1.2, all mathematics is in fenced code blocks /
backtick-Unicode (no LaTeX/MathJax) so it renders in any Markdown viewer. Every
symbol carries over from 1.1; new symbols are defined in §0. A compilable LaTeX
companion lives in [`03-multi-dealer-systemic-risk.tex`](03-multi-dealer-systemic-risk.tex).

---

## 0. Carry-over and new notation

From [`01-analytic-stability-boundary.md`](01-analytic-stability-boundary.md) we
reuse, verbatim, the single-dealer one-step objective at frozen toxic level `T`
and its three closed-form constants (all functions of the config, §6/§7 of 1.1):

```
   J(h; T) = P * [ h*A*exp(-k*h)*rho  +  h*T  -  psi*T  -  w*(h - h_ref)^2  -  lambda_q ]
   P = pnl_scale

   gamma   = - d^2 J / dh^2   = P * [ 2*w + A*rho*k*exp(-k*h)*(2 - k*h) + lambda_q ]   (strong convexity)
   beta    = | d^2 J / dh dT | = P                                                      (joint smoothness)
   epsilon = | d tau / d h |   = rho*gbar*alpha*f*I*c_t*exp(-c_t*h)                      (distribution sensitivity)
   m_1     = epsilon*beta/gamma                                                          (single-dealer modulus, 1.1 §3.4)
```

| New symbol | Reading | Definition |
|------------|---------|------------|
| `N` | number of competing dealers | proposed config `clients.n_dealers`, §1 |
| `h_i` | half-spread quoted by dealer `i` | per-dealer policy coordinate |
| `h = (h_1,...,h_N)` | the joint quote profile (bold in the `.tex`) | strategy profile in `R^N` |
| `h_i^dep` | deployed spread of dealer `i` (sets the frozen toxic env.) | §1, A3' |
| `tau_i` | informed (toxic) notional hitting dealer `i` | §2 |
| `kappa` | cross-dealer toxic spillover, `in [0,1]` | proposed config `clients.toxic_spillover`, §2 |
| `T_i` | frozen toxic level seen by dealer `i`, `= tau_i(h^dep)` | §2 |
| `BR_i` | dealer `i`'s frozen-environment best response | §3 |
| `J_BR` | Jacobian of the joint map `BR: R^N -> R^N` at the PSNE | §3 |
| `N_eff` | effective dealer count, `= 1 + kappa*(N-1)` | §4 |
| `m_N` | joint contraction modulus, `= N_eff*m_1 = spectral radius of J_BR` | §4 |
| `gamma_joint` | joint effective curvature (worst-case orientation), `= gamma - N_eff*epsilon*beta` | §6 |
| `N_c` | critical dealer count at the boundary, `= gamma/(beta*epsilon) = 1/m_1` | §8 |
| `1` | the all-ones vector in `R^N`; `1 1^T` the all-ones matrix | §4 |
| `c` | aggregate spillover intensity in the mean-field scaling `kappa = c/N` | §7 |

---

## 1. Setup: `N` symmetric dealers sharing one informed-flow pool

We extend the single-representative-bond, zero-skew analysis of 1.1 to `N`
dealers quoting the *same* bond simultaneously. As in 1.1 we work at the level of
each dealer's central half-spread `h_i` (the bias `b_h` of its `LinearPolicy`),
the dominant coordinate of the iterate map.

- **A1' (symmetric dealers, single bond, zero skew).** All `N` dealers share the
  microstructure constants `(A, k, w, h_ref, P, lambda_q)` and the toxic
  parameters `(I_b, I, alpha, f, c_t, sigma_s)` of 1.1. Heterogeneity is a
  documented extension (§11). One bond, `skew = 0` for every dealer.
- **A2' (reference state).** Curvature and the gate constants `gbar`, `psi` are
  evaluated at the common probe reference state `s0` with inventory `q0 ~ 0`,
  mispricing `g`, liquidity ratio `rho` -- identical to 1.1 A2. Expectations are
  over the same client-flow noise.
- **A3' (frozen-environment, simultaneous best response).** Carries 1.1's A3 to
  the game: within one deployment round the learned operator conditions on a
  **frozen** `policy_summary` of the *whole deployed profile* `h^dep =
  (h_1^dep,...,h_N^dep)`. Each dealer `i`, computing its best response, sees the
  toxic environment frozen at the level induced by the deployed profile (write
  `T_i = tau_i(h^dep)`), while its *own* benign demand-elasticity channel still
  responds to its candidate `h_i`. All dealers redeploy together -- a *simultaneous*
  repeated-retraining (RRM) round, exactly the multiplayer setting of Brown et al.
  (2021) and Narang et al. (2022).
- **A4' (saturation non-binding).** The informed-flow cap `info_cap*tanh(.)` is
  slack at the operating point (1.1 A4); we carry the uncapped form and flag the
  saturated deep-unstable regime as a corollary.
- **A5' (own-franchise benign flow).** Each dealer's *uninformed* notional
  `A*exp(-k*h_i)*rho` responds to its **own** spread only -- a captured client
  franchise. The competitive coupling is carried entirely by the **toxic**
  channel (A2 below), which is where the *performative* feedback lives. Layering
  competitive splitting of benign flow on top (a Bertrand/Cournot quote game) only
  *adds* own-`h` curvature -- it raises `gamma`, is stabilising, and does not change
  the performative coupling structure; we treat it as out of scope (§11).

Assumption A5' is the modelling choice that keeps the lift clean: it isolates the
one channel through which a dealer's quote moves the *distribution other dealers
face*, which is the only channel that can produce performative instability.

---

## 2. Coupled toxic flow and the per-dealer objective

### 2.1 The shared informed-flow pool

In 1.1 the toxic notional hitting the single dealer was
`tau(h_dep) = rho*gbar*(I_b + alpha*f*I*exp(-c_t*h_dep))`: tighter quotes
(`h_dep` small) summon more informed flow, slope `epsilon = |d tau/d h|`.

With `N` dealers the informed pool is *shared*: informed traders profit from
picking off *whichever* dealer is exploitable, and a tighter market -- *any*
dealer tightening -- raises the attractiveness of the venue to informed flow,
increasing the toxic notional routed at *every* dealer. We model the toxic
notional hitting dealer `i` as its own-spread response plus a spillover from
every competitor:

```
   tau_i(h^dep) = rho*gbar * ( I_b  +  alpha*f*I * [ exp(-c_t*h_i^dep)
                                       +  kappa * Sum_{j != i} exp(-c_t*h_j^dep) ] )      (A2-multi)
```

`kappa in [0,1]` is the **toxic spillover coefficient**:

- `kappa = 0`: each dealer's toxic flow responds only to its *own* spread -- `N`
  independent single-dealer problems (a sanity-check limit).
- `kappa = 1`: **full spillover** -- the informed pool is a single shared resource
  whose size tracks the *aggregate* market attractiveness
  `Sum_{j=1}^N exp(-c_t*h_j^dep)`, split symmetrically across dealers. This is the
  "shared induced distribution" of the multiplayer-performativity literature
  (Brown et al. 2021) and yields the README's target `N`-boundary.

`kappa` is the structural primitive of this document, just as `f` was for 1.1; it
measures how much one dealer's tightening contaminates the flow its competitors
face. The two key partial sensitivities of `tau_i` to the *deployed* profile are

```
   d tau_i / d h_i^dep = rho*gbar*alpha*f*I*(-c_t)*exp(-c_t*h_i^dep) = - epsilon       (own slope)
   d tau_i / d h_j^dep = kappa * (- epsilon)                              (j != i, cross slope)
```

evaluated at the symmetric point `h_j^dep = h*` for all `j`, with
`epsilon = rho*gbar*alpha*f*I*c_t*exp(-c_t*h*)` -- the **same** single-dealer
`epsilon` of 1.1. The cross-slope is a `kappa`-discounted copy of the own-slope.

### 2.2 The per-dealer objective is structurally 1.1's

Under A1'/A5', dealer `i`'s expected one-step objective at its frozen toxic level
`T_i = tau_i(h^dep)` is *identical in form* to 1.1's `J`, with `h -> h_i`,
`T -> T_i`:

```
   J_i(h_i; T_i) = P * [ h_i*A*exp(-k*h_i)*rho  +  h_i*T_i  -  psi*T_i
                         -  w*(h_i - h_ref)^2  -  lambda_q ] .
```

The competitor spreads enter `J_i` *only* through the frozen scalar `T_i` -- they
do not appear in dealer `i`'s own benign or quoting-cost terms (A5'). This is what
makes the game tractable: each dealer solves a 1.1-shaped scalar problem; the
coupling is entirely in how the `T_i` are wired to the deployed profile (§2.1).

### 2.3 The three single-dealer derivatives carry over unchanged

Because `J_i` has 1.1's form, differentiating in `(h_i, T_i)` reproduces 1.1
§3.2 verbatim:

```
   d^2 J_i / d h_i^2   = - gamma            (curvature in own spread; same gamma as 1.1)
   d^2 J_i / d h_i d T_i = + beta = + P     (cross sensitivity to own frozen toxic level)
```

The adverse term `-psi*T_i` is constant in `h_i`, so -- exactly as in 1.1 §2.6 --
it sets the profit *level* (and drives the multi-dealer echo-chamber gap, a 1.2 x
1.3 cross-product noted in §11) but **does not enter the stability boundary**.
Only `gamma`, `beta`, and now the *coupled* `tau_i` matter for stability.

---

## 3. The joint best-response map and its Jacobian

### 3.1 The map

Each dealer best-responds to the frozen profile by maximising `J_i` in `h_i`:

```
   BR_i(h^dep) = argmax_{h_i}  J_i( h_i ; tau_i(h^dep) ) ,
```

with first-order condition `F_i(h_i, T_i) := d J_i/d h_i = 0`,

```
   F_i = P * [ A*rho*exp(-k*h_i)*(1 - k*h_i) + T_i - 2*w*(h_i - h_ref) ] = 0 .
```

The simultaneous-RRM iteration is the joint map `h^{t+1} = BR(h^t)`,
`BR = (BR_1,...,BR_N): R^N -> R^N`. A **PSNE** `h*` is a fixed point
`h* = BR(h*)`: every dealer's deployed spread is its own best response *given the
toxic environment that same profile induces*. Linearising at `h*`,

```
   h^{t+1} - h*  ~  J_BR * (h^t - h*) ,        J_BR := [ d BR_i / d h_j^dep ]_{i,j} ,
```

so the joint loop contracts iff the **spectral radius** `rho(J_BR) < 1` -- the
product-space generalisation of the single-dealer `|BR'| < 1` (Brown et al. 2021,
the joint map must contract in the product space).

### 3.2 The Jacobian by the implicit-function theorem

Differentiate the per-dealer fixed-point identity `F_i(BR_i(h^dep), tau_i(h^dep)) = 0`
totally in `h_j^dep`. Since `F_i` depends on the profile only through `h_i`
(via `BR_i`) and through the frozen `T_i = tau_i(h^dep)`:

```
   F_{i,h_i} * (d BR_i / d h_j^dep)  +  F_{i,T_i} * (d tau_i / d h_j^dep)  =  0 ,
```

with `F_{i,h_i} = d^2 J_i/d h_i^2 = -gamma`, `F_{i,T_i} = d^2 J_i/d h_i dT_i = +beta`,
and the §2.1 slopes `d tau_i/d h_j^dep = -epsilon` (`j=i`) or `-kappa*epsilon`
(`j != i`). Solving,

```
   d BR_i / d h_i^dep = - F_{i,T_i} * (d tau_i/d h_i^dep) / F_{i,h_i}
                      = - (beta)*(-epsilon)/(-gamma)          = - beta*epsilon/gamma   = - m_1   (diagonal)

   d BR_i / d h_j^dep = - (beta)*(-kappa*epsilon)/(-gamma)    = - kappa*beta*epsilon/gamma = - kappa*m_1   (off-diag)
```

The common P&L scale `P = beta` cancels against `gamma` exactly as in 1.1 §3.4.
So at the symmetric PSNE the joint Jacobian is the **`N x N` matrix**

```
   J_BR = - m_1 * [ (1 - kappa)*I  +  kappa * 1 1^T ] ,          m_1 = epsilon*beta/gamma .      (J_BR)
```

Its diagonal entries are `-m_1` (a dealer responding to its *own* deployment, the
single-dealer self-coupling) and its off-diagonal entries are `-kappa*m_1` (a
dealer responding to a *competitor's* deployment through the shared toxic pool).
`J_BR` is **symmetric** -- a structural consequence of the symmetric shared pool --
which (§4) makes its spectral radius equal to its operator 2-norm, so the
contraction is a genuine Euclidean-norm contraction, not merely asymptotic.

---

## 4. Spectrum, the PSNE boundary, and the headline theorem

### 4.1 Eigenstructure of `J_BR`

`J_BR` is a rank-one update of a scaled identity, so its spectrum is explicit. The
all-ones matrix `1 1^T` has eigenvalue `N` (eigenvector `1`, the **common mode**:
all dealers move together) and eigenvalue `0` with multiplicity `N-1`
(eigenvectors orthogonal to `1`, the **differential modes**: dealers move against
each other, zero net market move). Hence the eigenvalues of `J_BR` are

```
   common mode      (eigvec 1):       lambda_+ = - m_1 * [ (1-kappa) + kappa*N ] = - m_1 * (1 + kappa*(N-1))
   differential modes (eigvec _|_ 1): lambda_- = - m_1 * (1 - kappa)              (multiplicity N-1)
```

Define the **effective dealer count**

```
   N_eff := 1 + kappa*(N-1)          (so  N_eff = 1 at kappa=0,  N_eff = N at kappa=1) .
```

Then `lambda_+ = -m_1*N_eff` and `lambda_- = -m_1*(1-kappa)`. Since
`N_eff >= 1 >= 1 - kappa >= 0` for `kappa in [0,1]`, the **common mode dominates**:

```
   rho(J_BR) = m_1 * N_eff =: m_N        (joint contraction modulus) .                    (m_N)
```

### 4.2 The boundary (Theorem 1)

**Theorem 1 (PSNE stability boundary).** *For `N` symmetric dealers sharing a
toxic pool with spillover `kappa in [0,1]`, the simultaneous-RRM loop contracts to
the PSNE iff*

```
   m_N = N_eff * epsilon * beta / gamma  <  1
   <==>
   epsilon  <  gamma / ( N_eff * beta ) ,        N_eff = 1 + kappa*(N-1) .                 (boxed result)
```

*In particular, under full spillover `kappa = 1` (`N_eff = N`) this is the target*

```
   epsilon  <  gamma / ( N * beta ) ,      with market-wide instability at  N*epsilon*beta/gamma = 1 ,
```

*and at `kappa = 0` it reduces to the single-dealer boundary `epsilon < gamma/beta`
of 1.1, applied independently to each of the `N` decoupled dealers.*

**Proof.** `BR` is `C^1` near the symmetric fixed point (`J_i` is smooth and
strongly concave in `h_i` by `gamma > 0`, so each `BR_i` is a smooth implicit
function, §5). The iteration `h^{t+1} = BR(h^t)` therefore contracts in a
neighbourhood of `h*` iff `rho(J_BR) < 1` (Ostrowski / Banach for the linearised
map). By §4.1, `rho(J_BR) = m_1*N_eff`. Substituting `m_1 = epsilon*beta/gamma`
and rearranging gives the boundary. Because `J_BR` is symmetric, `rho(J_BR)` is
its 2-norm, so for `m_N < 1` the map is a strict Euclidean contraction and the
convergence is global on the basin where the linearisation's sign structure holds
(`gamma(h) > 0` on the operating range, as in 1.1 §4). [] 

The mechanism is sharp: every entry of `J_BR` carries a copy of the *same*
single-dealer feedback `-m_1`, but the **common-mode eigenvalue sums `N_eff` of
them in phase**. The destabilising direction is not any one dealer's own loop -- it
is the synchronised market-wide tightening that summons a synchronised toxic surge.
Competition does not add new feedback channels; it lets the *existing* feedback
resonate across dealers.

### 4.3 Common mode unstable, differential modes *more* stable

A corollary of §4.1 worth stating on its own: the differential-mode eigenvalue
`|lambda_-| = m_1*(1-kappa) <= m_1 < m_N`. So as the market is driven through the
boundary, the *first* (and only) mode to go unstable is the common mode; dealers
deviating idiosyncratically from the pack are **damped more strongly** than a lone
dealer would be. Instability here is intrinsically *systemic* -- it lives in the
aggregate, synchronised direction, never in dealer-specific dispersion. (At
`kappa = 1` the differential modes are *neutrally* damped, `lambda_- = 0`: pure
relative repositioning neither grows nor decays at the boundary, all the action is
in the aggregate.)

---

## 5. Existence and uniqueness of the PSNE

**Existence.** Each strategy space `h_i in [0, max_half_spread]` is compact and
convex; the joint space is the compact convex box `B = [0, max_half_spread]^N`.
Under A1' each `J_i(.; T_i)` is continuous and strongly concave in `h_i`
(`-d^2J_i/dh_i^2 = gamma > 0` on the operating range, 1.1 §4), and `tau_i` is
continuous in `h^dep`, so each `BR_i` is a continuous single-valued map and
`BR: B -> B` is a continuous self-map of a nonempty compact convex set. By
**Brouwer's fixed-point theorem** a PSNE `h* = BR(h*)` exists. (This is the
market-making instance of Brown et al. (2021)'s PSNE existence result; their
"the joint map is a continuous self-map of the product space" hypothesis is
discharged here by the per-dealer strong concavity inherited from 1.1.)

**Uniqueness in the stable regime.** When `m_N < 1` (Theorem 1), `BR` is a
contraction in the 2-norm on `B` (§4.2), so the fixed point is **unique** and
globally attracting on `B` by Banach. Outside the stable regime multiplicity is
possible (the saturated cap A4' re-convexifies and can pin auxiliary equilibria);
the unique interior PSNE is the object of interest and is located, by symmetry, by
the *single* scalar fixed-point equation of 1.1 §4 with the toxic term replaced by
the coupled `tau_i` evaluated at `h_i^dep = h*` for all `i`:

```
   A*rho*exp(-k*h*)*(1 - k*h*)  +  rho*gbar*( I_b + N_eff*alpha*f*I*exp(-c_t*h*) )  -  2*w*(h* - h_ref)  =  0 ,
```

i.e. the symmetric PSNE half-spread solves a 1.1-shaped scalar equation with the
toxic baseline-plus-slope amplified by the *same* `N_eff` (each dealer at the
symmetric profile sees its own attractiveness plus `kappa*(N-1)` copies of its
neighbours'). Solve by Newton/bisection on `[0, max_half_spread]`, exactly as 1.1.

---

## 6. Convergence rate of the joint loop

### 6.1 Linear rate from the contraction

Theorem 1 already gives the deterministic rate. With `J_BR` symmetric and
`m_N < 1`, the linearised iteration is a Euclidean contraction with ratio `m_N`:

```
   || h^t - h* ||_2  <=  m_N^t * || h^0 - h* ||_2 ,        m_N = N_eff*epsilon*beta/gamma .
```

So the joint RRM loop converges **linearly at the joint modulus** -- and slows as
`N` (or `kappa`) grows, since `m_N = N_eff*m_1` increases toward the boundary. The
number of rounds to reach tolerance `delta` scales like
`log(delta)/log(m_N) ~ 1/(1 - m_N)` near the boundary: markets with more
fully-coupled dealers not only destabilise sooner, they *converge more slowly*
while still stable -- a critical-slowing-down signature directly testable in the
simulator.

### 6.2 Strong-monotonicity / stochastic-gradient view (Narang et al. 2022)

The gradient-dynamics counterpart (the multiplayer analogue of 1.1 §5's RGD
orientation) governs the *stochastic* loop in which each dealer takes damped
gradient steps. The joint Nash gradient operator's strong-monotonicity constant is
the worst-case-orientation **joint effective curvature**

```
   gamma_joint := gamma - N_eff*epsilon*beta ,           gamma_joint > 0  <==>  epsilon < gamma/(N_eff*beta) ,
```

which at `kappa = 1` is `gamma_joint = gamma - N*epsilon*beta` -- exactly the
README's `gamma_joint ~ gamma - N*epsilon*beta`, and the `N`-dealer generalisation
of 1.1 §5's `gamma_eff = gamma - epsilon*beta` (recovered at `N_eff = 1`). Under
`gamma_joint > 0` the joint game is strongly monotone, so Narang et al. (2022)
give:

- **stochastic gradient** (noisy operator fits): `O(1/k)` on the squared distance
  to the PSNE, `E|| h^k - h* ||^2 = O(1/(gamma_joint * k))` -- mirroring the
  single-agent `O(1/k)` of 1.2 §4;
- **deterministic / full-gradient**: linear at ratio `(1 - eta*gamma_joint)` for
  step `eta <= 1/L_joint`, the `N`-dealer twin of 1.2's `(1 - eta*gamma_PO)`.

When `gamma_joint <= 0` but the weaker **variational-stability** condition of
Narang et al. holds (possible in the defensive-widening regime where `gamma` is
small but the game is still monotone in aggregate), the loop still converges, but
to a *weaker* equilibrium concept -- an honest caveat for the systemic-risk reading
(the market can "converge" to a fragile, low-curvature aggregate, §11).

---

## 7. The mean-field limit `N -> inf`

How the boundary behaves as the market grows depends on how the shared pool scales
-- two structurally different regimes, both worth reporting (Lacker 2016 supplies
the rigorous `N`-player -> continuum passage).

### 7.1 Strong coupling (`kappa` fixed > 0): systemic collapse

If the spillover `kappa` is held fixed as `N` grows, `N_eff = 1 + kappa*(N-1) ~
kappa*N -> inf`, so the boundary collapses:

```
   epsilon*(N) = gamma / ( N_eff*beta )  ~  gamma / ( kappa*N*beta )  ->  0      as  N -> inf .
```

**Any** positive performative feedback `epsilon > 0` destabilises a sufficiently
large fully-coupled market. This is the systemic-risk alarm in its starkest form:
a fixed per-dealer toxic spillover makes the stable region vanish as the number of
competitors grows -- competition is *unboundedly* destabilising when the informed
pool is a fixed shared resource.

### 7.2 Mean-field coupling (`kappa = c/N`): a finite continuum boundary

The economically natural large-market scaling is the **mean-field** one, in which
each individual dealer's contribution to the shared pool is `O(1/N)` (informed
flow responds to the *empirical distribution* of quotes, not to any single
dealer): set `kappa = c/N` for a fixed **aggregate spillover intensity** `c >= 0`.
Then

```
   N_eff = 1 + (c/N)*(N-1)  ->  1 + c          as  N -> inf ,
```

and the boundary converges to a **finite, well-defined mean-field limit**

```
   epsilon  <  gamma / ( (1 + c)*beta )  =:  gamma / ( beta*N_eff^MF ) ,        N_eff^MF = 1 + c .       (MFG boundary)
```

Equivalently the **mean-field effective sensitivity** is `epsilon_MF = (1 + c)*epsilon`:
each dealer behaves as if facing `(1 + c)` copies of its own feedback -- its own
plus the aggregate `c` summoned by the population. In this scaling the aggregate
attractiveness `(1/N) Sum_j exp(-c_t*h_j^dep)` concentrates on its population mean
`E_{h ~ mu}[exp(-c_t*h)]` (a law-of-large-numbers / propagation-of-chaos statement,
Lacker 2016), so the `N`-dealer phase diagram has a clean `N -> inf` limit and the
`(N, epsilon)` stability region is *well-posed* in the continuum -- the theoretical
licence to scale Priority 1.5's phase diagram from a handful of dealers to a
realistic 10+-dealer, 100+-bond market without the boundary degenerating.

The two regimes bracket reality: §7.1 (fixed `kappa`) is the worst case where
toxic capacity does not grow with the venue count; §7.2 (`kappa = c/N`) is the
benign case where adding dealers dilutes each one's individual footprint. Which
holds is an empirical microstructure question -- and, usefully, the simulator can
distinguish them by measuring how `m_N` scales with `N` at fixed `epsilon` (§10).

---

## 8. Corollaries (each a falsifiable prediction)

### 8.1 Critical dealer count
At full spillover, the boundary is crossed not only by raising `epsilon` but by
**adding dealers**. Holding the structural regime fixed, the market is stable iff
`N < N_c` with

```
   N_c = gamma / ( beta*epsilon ) = 1 / m_1 .
```

A market stable as a monopoly (`m_1 < 1`) is destabilised once the number of
fully-coupled dealers exceeds `1/m_1`. This converts the headline into a *count*:
e.g. `m_1 = 0.5` -> `N_c = 2` (a duopoly already sits at the boundary);
`m_1 = 0.2` -> `N_c = 5`. Directly testable: sweep `N` at fixed config and locate
the integer crossing of `m_N = 1`.

### 8.2 Joint modulus is linear in `N_eff` (hence in `N` at `kappa = 1`)
`m_N = N_eff*m_1` is **exactly linear** in the effective dealer count at fixed
structural regime, so a plot of measured joint modulus against `N` (full spillover)
should be a straight line of slope `m_1` through the origin -- a stringent
one-parameter check of the whole construction against the simulator.

### 8.3 Critical slowing down near the boundary
By §6.1 the convergence ratio is `m_N -> 1` as `(N, epsilon)` approach the
boundary, so the relaxation time `~ 1/(1 - m_N)` diverges. The simulator should
show RRM taking *progressively more rounds* to settle as `N` is increased toward
`N_c` -- the dynamical early-warning signature of an approaching systemic
transition, even before outright divergence.

### 8.4 Spillover `kappa` and the policy lever
Because the boundary depends on `N` only through `N_eff = 1 + kappa*(N-1)`, the
*regulatory/structural* knob with leverage is `kappa`, not `N`: fragmenting the
informed pool (lowering `kappa`, e.g. via segmented RFQ, last-look, or
information-leakage controls) lowers `N_eff` and re-stabilises a market that "too
many dealers" had pushed past the boundary. This is the systemic-risk policy
reading and a concrete `(kappa, N)`-sweep prediction.

### 8.5 The boundary is `epsilon`-driven, not `alpha`-driven, here too
Exactly as 1.1 §6.4, sweeping `toxicity_feedback` `f` (which moves `epsilon`
linearly at near-fixed `h*`) is the clean axis; sweeping `alpha` confounds
`epsilon` and `h*`. The multi-dealer phase diagram should therefore be drawn on
the `(N, f)` (equivalently `(N, epsilon)`) plane, not `(N, alpha)`.

---

## 9. Symbol -> config map

| Symbol | Meaning | Config field | Status |
|--------|---------|--------------|--------|
| `N` | number of dealers | `clients.n_dealers` | **added** (default `1`) |
| `kappa` | toxic spillover across dealers | `clients.toxic_spillover` | **added** (default `0.0`) |
| `epsilon`, `beta`, `gamma`, `m_1` | single-dealer constants | derived in 1.1 (see its §7 map) | existing |
| `N_eff` | effective dealer count `1+kappa(N-1)` | derived (§4) | - |
| `m_N` | joint modulus `N_eff*m_1` | derived (§4); measured by common-mode probe (§10) | - |
| `gamma_joint` | joint curvature `gamma - N_eff*epsilon*beta` | derived (§6) | - |
| `N_c` | critical dealer count `1/m_1` | derived (§8.1) | - |
| `c` | mean-field aggregate spillover (`kappa = c/N`) | derived (§7.2) | - |

The two **new** fields (`clients.n_dealers`, `clients.toxic_spillover`, now added
to `ClientsConfig`) are the only new configuration this priority requires; every
other quantity is a closed form in the *existing* 1.1 config. `n_dealers = 1` (or
`toxic_spillover = 0`) reproduces the present single-dealer behaviour bit-for-bit
-- checked by the `kappa = 0` regression test in `tests/test_multi_dealer.py` (§10).

---

## 10. Validation protocol and the remaining code task

**Predict (no new math).** For each `(N, kappa, f)` on the
`sweep_feedback.yaml`-style grid: solve the symmetric PSNE `h*` (§5 scalar
root-find), evaluate `m_1 = epsilon*beta/gamma` from 1.1's closed forms at `h*`,
then predict `m_N = N_eff*m_1`, the boundary `epsilon < gamma/(N_eff*beta)`, and
the critical count `N_c = 1/m_1`.

**Measure -- the common-mode BR-slope probe.** The natural multi-dealer
generalisation of [`response_modulus.py`](../../endo_market_v2/endo_market/analysis/response_modulus.py)'s
common-random-number finite difference is to perturb **all `N` deployed spreads in
phase** (along the unstable eigenvector `1`) and read off the joint best response:

```
   m_N^meas = | BR_i( h* + delta*1 )  -  BR_i( h* - delta*1 ) | / ( 2*delta )      (any i, by symmetry)
```

driving both probes with shared RNG (operator init, collection noise, optimiser
noise) so the sampling noise cancels and the difference isolates the common-mode
eigenvalue `lambda_+ = -m_N`. (Perturbing a *single* dealer's spread instead would
mix `lambda_+` and `lambda_-`; the in-phase perturbation is what cleanly returns
the unstable modulus.) A *second*, anti-phase probe along an eigenvector
orthogonal to `1` returns `m_1*(1-kappa)` and so **identifies `kappa` empirically**
from the ratio of the two measured slopes -- a free calibration of the spillover.

**Compare -> the systemic phase diagram (the Priority 3 deliverable).** Report
`m_N^pred` vs `m_N^meas` across the `(N, epsilon)` grid with cross-seed median+IQR
bands; confirm (i) the linear-in-`N` law (§8.2), (ii) the integer crossing at
`N_c` (§8.1), and (iii) the critical-slowing-down of round-count near the boundary
(§8.3). The 2-D `(N, epsilon)` stability region *is* the paper's systemic-risk
figure: competitive quoting destabilises the market a factor `N_eff` before any
single dealer's loop would.

**Code (DONE).** Implemented in
[`analysis/multi_dealer_modulus.py`](../../endo_market_v2/endo_market/analysis/multi_dealer_modulus.py),
with the config fields `clients.n_dealers` and `clients.toxic_spillover` added
(defaults `1` / `0.0`, reproducing `endo_market_v2` exactly). `multi_dealer_boundary`
returns the closed-form `m_N`, `gamma_joint`, `N_c` and boundary; `joint_jacobian`
builds `J_BR` and its spectrum; `run_joint_rrm` runs the genuine `N`-dimensional
coupled-`tau_i` best-response cobweb; `mean_field_boundary` / `strong_coupling_limit`
give the §7 limits; and `common_mode_probe` (with `measure_common_mode_modulus` /
`measure_differential_modulus`) runs the in-phase / anti-phase probes and identifies
`kappa`. The empirical probes **reuse the single-dealer machinery** exactly as
prescribed: the symmetric in-phase common mode is provably a single-dealer problem
with the toxic slope amplified by `N_eff` (`effective_config`), so
`measure_response_modulus` is called on that effective config — no `N`-body
simulator surgery is required. (Absolute agreement of the *learned-operator*
measurement with the linear `m_N = N_eff*m_1` law holds only in the responsive,
non-saturated regime — the documented saturation/attenuation caveat, §11.) Verified
by `tests/test_multi_dealer.py`.

---

## 11. Honest caveats (for the paper's limitations paragraph)

- **Full spillover `kappa = 1` is the strong assumption.** The headline
  `epsilon < gamma/(N*beta)` holds only when the informed pool is a single shared
  resource tracking aggregate attractiveness. Real OTC toxic flow is partially
  segmented (relationship trading, tiered RFQ, last-look), i.e. `kappa < 1`; the
  honest statement is the interpolated `epsilon < gamma/(N_eff*beta)` with `kappa`
  *measured* from the anti-phase probe (§10), not assumed. We report the boundary
  as a function of `kappa`, not a single `N`-law.
- **Symmetric dealers (A1').** With heterogeneous `(gamma_i, epsilon_i)`, `J_BR`
  has entries `-beta*(spillover_{ij})*epsilon_j/gamma_i` and the spectral radius is
  a weighted common-mode eigenvalue dominated by the *least-curved* (most
  defensive, smallest `gamma_i`) dealer -- so a single fragile dealer can set the
  systemic boundary. The symmetric result is the clean base case; the heterogeneous
  spectral bound is a stated extension.
- **Own-franchise benign flow (A5').** Competitive splitting of *uninformed* flow
  (a Bertrand quote game) is omitted; including it adds own-`h` curvature (raises
  `gamma`, stabilising) and a benign off-diagonal coupling, shifting `h*` and
  `N_eff`'s prefactor but not the `common-mode-resonance` mechanism. It is the
  natural next layer, not a correction to the present result.
- **Frozen-environment, first-order (A3').** As in 1.1/1.2 the operator is blind
  to `dD/dphi` *within* a deployment; the multi-dealer PerfGD correction (each
  dealer crediting itself for the cross-dealer flow its tightening summons) is the
  1.2 x 1.3 cross-product -- it would replace `gamma_joint` by an objective
  curvature `gamma_PO,joint` and, exactly as 1.2 §5, can stabilise the loop
  *beyond* the PSNE boundary. Deriving it is the clean follow-on; here we establish
  the *blind* multi-dealer boundary that PerfGD would then beat.
- **Cap slack (A4'), reference-state constants.** `gbar`, `psi`, `lambda_q`, and
  the cap are evaluated at the probe reference state; deep past the boundary the
  cap re-convexifies and the linearisation (and uniqueness, §5) cease to hold. The
  modulus is valid up to and slightly past the boundary -- the region of interest --
  not arbitrarily far into divergence. The phase diagram must report the boundary
  as a band over the reference-state distribution (1.1 §9), now also over `(N,
  kappa)`.
- **Mean-field regime selection (§7).** Whether the large-market limit is the
  collapsing §7.1 or the finite §7.2 depends on the unobserved scaling of toxic
  capacity with venue count; the simulator can *measure* which holds (slope of
  `m_N` in `N` at fixed `epsilon`), but the paper must not assert one without that
  measurement.

---

## References

- G. Brown, S. Hod, I. Kalemaj. *Performative Games.* arXiv:2106.09784, 2021. --
  performatively stable Nash equilibria and their existence; the joint-map
  contraction in product space that §3-§5 instantiate. (Lit. #11.)
- A. Narang, E. Faulkner, D. Drusvyatskiy, M. Fazel, L. J. Ratliff. *Multiplayer
  Performative Prediction: Learning in Decision-Dependent Games.* arXiv:2207.05630,
  2022. -- convergence rates (`O(1/k)` under joint strong monotonicity; variational
  stability), giving §6's joint `gamma_joint = gamma - N_eff*epsilon*beta`. (Lit. #12.)
- D. Lacker. *A General Characterization of the Mean Field Limit for Stochastic
  Differential Games.* Probab. Theory Relat. Fields, 2016. arXiv:1510.01408. -- the
  `N -> inf` propagation-of-chaos limit underlying §7's mean-field boundary. (Lit. #17.)
- J. Perdomo, T. Zrnic, C. Mendler-Dünner, M. Hardt. *Performative Prediction.*
  ICML 2020. -- the single-agent `m_1 = epsilon*beta/gamma` boundary that `N_eff`
  multiplies.
- Builds on [`01-analytic-stability-boundary.md`](01-analytic-stability-boundary.md)
  (the closed-form `gamma`, `beta`, `epsilon`, `m_1`) and
  [`02-perfgd-correction.md`](02-perfgd-correction.md) (the un-blinding cross-product
  of §11); see [`../references.bib`](../references.bib) and
  `literature/literature-raghav/README.md` (papers #11, #12, #17) for the full
  citation map.
