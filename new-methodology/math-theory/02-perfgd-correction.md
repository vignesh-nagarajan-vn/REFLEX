# 1.2 - Un-blinding the Operator: the PerfGD-Corrected Loop

**STATUS: DONE (Priority 2, derivation/theorem).** This document discharges the
*mathematical* content of methodology target **1.2** in
[`../README.md`](../README.md): it builds on the closed forms of
[`01-analytic-stability-boundary.md`](01-analytic-stability-boundary.md) to derive
the Performative Gradient Descent (PerfGD) correction in closed form, prove it
converges to the **performative optimum (PO)** rather than the **stable point
(SP)**, show it **remains stable for `epsilon` beyond the RRM boundary
`epsilon* = gamma/beta`**, and quantify the **echo-chamber gap** between the
stable and optimal quotes. The remaining *code* task - wiring this gradient into a
training loop as `equilibrium/perfgd_loop.py` - is called out in §8 and is
deliberately out of scope for the math-theory deliverable.

> **One-line thesis.** Blind RRM (the loop analysed in 1.1) finds the spread that
> is its own best response *given* the induced flow - the stable point. It is
> blind to the fact that its own quoting *moves* that flow. PerfGD adds exactly
> the missing term `dJ/dT * dtau/dh`, which 1.1 already gives in closed form, so
> no estimation is needed. The corrected loop optimises the true objective and is
> governed by the objective's *own* curvature `gamma_PO`, not by the cobweb
> modulus `epsilon*beta/gamma` - which is why it converges in regimes where RRM
> oscillates and diverges.

**Formatting note.** As in 1.1, all mathematics is in fenced code blocks /
backtick-Unicode (no LaTeX/MathJax). Every symbol carries over from 1.1; new
symbols are defined in §0.

---

## 0. Carry-over and new notation

From [`01-analytic-stability-boundary.md`](01-analytic-stability-boundary.md) we
reuse, verbatim, the expected one-step objective at frozen toxic level `T`:

```
   J(h; T) = P * [ h*A*exp(-k*h)*rho  +  h*T  -  psi*T  -  w*(h - h_ref)^2  -  lambda_q ]
   P = pnl_scale
```

and the three constants derived there (all closed-form functions of the config):

```
   gamma   = - d^2 J / dh^2   = P * [ 2*w + A*rho*k*exp(-k*h)*(2 - k*h) + lambda_q ]   (strong convexity)
   beta    = | d^2 J / dh dT | = P                                                      (joint smoothness)
   epsilon = | d tau / d h |   = rho*gbar*alpha*f*I*c_t*exp(-c_t*h)                      (distribution sensitivity)
   tau(h)  = rho*gbar*( I_b + alpha*f*I*exp(-c_t*h) ) = C0 + C1*exp(-c_t*h),
            C0 = rho*gbar*I_b ,  C1 = rho*gbar*alpha*f*I ,  so  epsilon(h) = c_t*C1*exp(-c_t*h) .
```

| New symbol | Reading | Definition |
|------------|---------|------------|
| `Phi(h)` | performative objective (to maximise) | `Phi(h) = J(h; tau(h)) = E_{D(h)}[objective]` |
| `PR(h)` | performative risk (to minimise) | `PR(h) = - Phi(h)` |
| `h_SP` | stable point (RRM fixed point) | zero of the *blind* gradient, §1 |
| `h_PO` | performative optimum | maximiser of `Phi`, §1 |
| `G(h)` | blind / decoupled gradient | `G(h) = d1 J(h; tau(h))` (partial in 1st arg only) |
| `Delta(h)` | performative correction term | `Delta(h) = dT J * (d tau/dh)` |
| `gamma_PO` | curvature of the objective at the PO | `gamma_PO = - Phi''(h_PO)`, §4 |
| `psi` | per-unit adverse severity (from 1.1 §2.5) | `psi = sigma_s*u*a(u)/gbar(u) > 0` |

---

## 1. Two solution concepts: stable point vs. performative optimum

Following Perdomo et al. (2020), a decision-dependent problem has two distinct
stationary objects.

**Stable point `h_SP` (what RRM finds).** The dealer best-responds to the *frozen*
distribution, then redeploys. A fixed point zeroes the **blind gradient** - the
derivative of `J` in its first (decision) argument only, with `T` held fixed and
*then* evaluated self-consistently at `T = tau(h)`:

```
   G(h) := d1 J(h; tau(h)) = P * [ A*rho*exp(-k*h)*(1 - k*h) + tau(h) - 2*w*(h - h_ref) ] ,
   h_SP solves  G(h_SP) = 0 .
```

This is exactly the FOC of 1.1 §3–4; `h_SP` is the `h*` of that document.

**Performative optimum `h_PO` (what we actually want).** The maximiser of the true
objective `Phi(h) = J(h; tau(h))`, whose stationarity uses the **total**
derivative - the blind gradient *plus* the term that accounts for how `h` moves
the distribution:

```
   Phi'(h) = G(h) + Delta(h) ,    Delta(h) := dT J(h; tau(h)) * (d tau / d h) ,
   h_PO solves  Phi'(h_PO) = 0 .
```

The only difference between the two FOCs is `Delta`. **Computing `Delta` is the
whole game**, and 1.1 already did it.

---

## 2. The analytic performative gradient (no estimation)

The correction needs two ingredients, both closed-form from 1.1.

**(a) Marginal value of toxic flow `dT J`.** Differentiate `J` in the toxic level
`T`. The toxic spread-capture term `h*T` contributes `+h`; the adverse term
`-psi*T` contributes `-psi` (this term was *constant in `h`* and dropped out of
the 1.1 boundary, but it is **not** constant in `T`, so it returns here):

```
   dT J = P * ( h - psi ) .
```

Interpretation: a marginal unit of toxic flow earns the half-spread `h` but costs
`psi` to adverse selection; its net value is `h - psi`. Toxic flow is net
*profitable* where `h > psi` and net *toxic* where `h < psi`.

**(b) Flow response `d tau/dh = -epsilon(h)`** (from 1.1 §3.2). Hence

```
   Delta(h) = dT J * (d tau/dh) = P*(h - psi) * ( - epsilon(h) ) = - P*(h - psi)*epsilon(h) .
```

So the full PerfGD ascent direction is

```
   Phi'(h) = G(h) + Delta(h)
           = P * [ A*rho*exp(-k*h)*(1 - k*h) + tau(h) - 2*w*(h - h_ref)
                   - (h - psi)*epsilon(h) ] .                                    (PerfGD gradient)
```

Every term is a closed-form function of the config and `h`. **No finite-difference
probing of `dD/dphi` is required** - the contrast with Izzo et al. (2021), who
*estimate* the distribution-response by perturbation and inherit its bias and
variance. Here 1.1 supplies it exactly. This is the concrete pay-off of having
derived the boundary analytically.

---

## 3. The PerfGD update

```
   PerfGD:   h_{k+1} = h_k + eta * Phi'(h_k) ,        (gradient ascent on the true objective)
```

with `Phi'` from §2 and step `eta > 0`. Contrast the two loops:

```
   RRM  / blind:   h_{k+1} = h_k + eta * G(h_k)              -> converges to  h_SP   (if epsilon < gamma/beta)
   PerfGD:         h_{k+1} = h_k + eta * [ G(h_k)+Delta(h_k) ] -> converges to  h_PO  (under §4 condition)
```

Because the *only* added work is evaluating the scalar `Delta(h_k) =
-P*(h_k-psi)*epsilon(h_k)`, PerfGD costs the same as RRM per step.

---

## 4. Convergence to the performative optimum

PerfGD is ordinary gradient ascent on `Phi`. Its behaviour is set by the
curvature of `Phi` at the optimum, which we now compute exactly.

### 4.1 The objective curvature `gamma_PO`

Differentiate `Phi(h) = J(h; tau(h))` twice. Write the pieces (using
`tau' = -epsilon`, `tau'' = c_t*epsilon`, all from §0):

```
   uninformed:        d^2/dh^2 [ h*A*rho*exp(-k*h) ]      = - A*rho*k*exp(-k*h)*(2 - k*h)
   toxic capture:     d^2/dh^2 [ h*tau(h) ]               = 2*tau' + h*tau'' = epsilon*(c_t*h - 2)
   adverse:           d^2/dh^2 [ -psi*tau(h) ]            = -psi*tau'' = -psi*c_t*epsilon
   quoting cost:      d^2/dh^2 [ -w*(h - h_ref)^2 ]       = -2*w
```

Summing and multiplying by `P`:

```
   Phi''(h) = P * [ -A*rho*k*exp(-k*h)*(2 - k*h) - 2*w + epsilon*(c_t*h - 2) - psi*c_t*epsilon ]
            = - gamma + beta*epsilon*( c_t*h - 2 - c_t*psi ) .
```

Therefore the **curvature of the true objective at the optimum** is

```
   gamma_PO := - Phi''(h_PO)
             = gamma  +  beta*epsilon*( 2 + c_t*psi - c_t*h_PO ) .                 (gamma_PO)
```

### 4.2 Rate

If `Phi` is `L`-smooth (`|Phi''| <= L`) and concave on the operating interval,
gradient ascent with `eta <= 1/L` gives the standard convex rate on the value
gap,

```
   Phi(h_PO) - Phi(h_k)  <=  ( h_0 - h_PO )^2 / ( 2*eta*k )  =  O(1/k) ,
```

matching the methodology's stated `O(1/k)`. If, in addition, `gamma_PO > 0`
(strong concavity - §4.3), the rate upgrades to **linear**:

```
   | h_k - h_PO |  <=  ( 1 - eta*gamma_PO )^k * | h_0 - h_PO | ,     for  eta <= 1/L ,
```

so the analytic-gradient PerfGD converges geometrically at ratio
`(1 - eta*gamma_PO)`.

### 4.3 When is `Phi` strongly concave?

From `gamma_PO` above, `Phi` is strongly concave at the optimum iff

```
   gamma + beta*epsilon*( 2 + c_t*psi - c_t*h_PO )  >  0 .
```

A clean sufficient condition is `c_t*h_PO < 2 + c_t*psi`, i.e. the optimum sits in
the regime where the performative second-order term is *stabilising*; then
`gamma_PO > gamma > 0` for **every** `epsilon` (the correction only adds
curvature). This is the regime of interest near the operating point; the
defensive wide-spread regime where it can flip is treated in §7/§9.

---

## 5. Headline: PerfGD is stable beyond the RRM boundary `epsilon*`

This is the theorem that justifies the whole correction.

```
   RRM   stable  <==>  m = epsilon*beta/gamma < 1   <==>   epsilon < epsilon* := gamma/beta .   (from 1.1)
   PerfGD stable <==>  gamma_PO > 0  and  eta < 2/L .                                            (§4)
```

The two stability conditions are governed by **different** quantities. RRM's
modulus `epsilon*beta/gamma` grows linearly in `epsilon` and crosses 1 at
`epsilon* = gamma/beta`; past that, the cobweb oscillates and diverges. PerfGD's
condition is `gamma_PO = gamma + beta*epsilon*(2 + c_t*psi - c_t*h_PO) > 0`. Under
the §4.3 condition `c_t*h_PO < 2 + c_t*psi`,

```
   gamma_PO  =  gamma + (positive) * epsilon   >  0     for all epsilon >= 0 ,
```

so **PerfGD converges for arbitrarily large `epsilon`, including `epsilon` well
beyond `epsilon*` where RRM has already diverged.** Intuitively: RRM diverges not
because the *objective* lost its optimum but because the *fixed-point iteration*
overshoots it (a negative-slope cobweb, 1.1 §3.3); PerfGD descends the objective
directly and never iterates the cobweb, so the cobweb modulus is irrelevant to
it. This is precisely the methodology's claim - "converges to the PO at rate
`O(1/k)` and remains stable for `epsilon` beyond the RRM boundary `epsilon*`" -
now derived.

---

## 6. The echo-chamber gap (stable vs. optimal)

How far is the naive stable point from the optimum? Linearise the PO FOC about
the SP. With `G(h_SP) = 0` and `G'(h_SP) = d1d1 J + d1dT J * tau' = -gamma -
beta*epsilon = -(gamma + beta*epsilon)`:

```
   0 = G(h_PO) + Delta(h_PO)
     ~ G(h_SP) + G'(h_SP)*(h_PO - h_SP) + Delta(h_SP)
     = 0 - (gamma + beta*epsilon)*(h_PO - h_SP) + Delta(h_SP) .
```

Solving with `Delta(h_SP) = -beta*epsilon*(h_SP - psi)`:

```
   DECISION gap (spread inflation):
       h_SP - h_PO  =  beta*epsilon*(h_SP - psi) / ( gamma + beta*epsilon )  =  O(epsilon) .   (6a)
```

So when toxic flow is net profitable at the stable spread (`h_SP > psi`), the
**stable point quotes strictly wider than the optimum** - the dealer over-defends
because, blind to `dD/dphi`, it never credits itself for the flow its own
tightening would summon. The gap is `O(epsilon)` to leading order and vanishes as
`epsilon -> 0` (no performativity, SP = PO).

The corresponding loss of *value* is quadratically smaller, because `Phi` is flat
to first order at its optimum (`Phi'(h_PO) = 0`):

```
   VALUE gap (performative-risk suboptimality of the SP):
       Phi(h_PO) - Phi(h_SP)  ~  (1/2)*gamma_PO*(h_SP - h_PO)^2
                              =  (1/2)*gamma_PO * [ beta*epsilon*(h_SP - psi)/(gamma+beta*epsilon) ]^2
                              =  O(epsilon^2) .                                                 (6b)
```

**Reconciliation with the README.** The methodology README quotes the
echo-chamber gap as `O(epsilon^2)`. That `O(epsilon^2)` is the *value* gap (6b) -
the standard performative-prediction suboptimality result. The raw *spread*
inflation (6a) is `O(epsilon)` to leading order. We report both explicitly and
label which is which; conflating them is a common slip.

---

## 7. Corollaries

### 7.1 The correction is self-financing exactly at `h = psi`
`Delta(h) = -beta*(h - psi)*epsilon(h)` vanishes at `h = psi`: when the quoted
half-spread equals the adverse severity, toxic flow is value-neutral and PerfGD
and RRM momentarily agree. The performative correction has a *sign flip* there -
it pulls toward *tighter* quotes when `h > psi` (chase the profitable summoned
flow) and toward *wider* quotes when `h < psi` (suppress the toxic summoned flow).

### 7.2 Gap scales linearly in adversariality
Since `epsilon` is linear in `alpha*f` (1.1 §3.2), so is the decision gap (6a) to
leading order: `h_SP - h_PO ~ (alpha*f) * [ rho*gbar*I*c_t*exp(-c_t*h_SP)*(h_SP -
psi) / gamma ]`. The echo chamber widens linearly as the market gets more
adversarial - a directly testable prediction.

### 7.3 PerfGD curvature vs. RRM curvature, side by side
```
   RRM    effective curvature:   gamma_eff = gamma - beta*epsilon         (worst-case, 1.1 §5)
   PerfGD objective curvature:   gamma_PO  = gamma + beta*epsilon*(2 + c_t*psi - c_t*h_PO)
```
The `epsilon`-coefficients differ in both sign and size: RRM *loses* curvature as
feedback grows, PerfGD (in the operating regime) *gains* it. This is the formal
statement of "un-blinding stabilises the loop".

---

## 8. Validation protocol and the remaining code task

**Validation (no new math).** With the same harness used in 1.1 §8:

1. Compute `h_SP` (1.1 §4 root-find) and `h_PO` (root of §2's `Phi'`), then the
   analytic gap (6a) and value gap (6b), across the `sweep_feedback.yaml` grid.
2. Run RRM and PerfGD loops from a common start; confirm RRM diverges and PerfGD
   converges for `epsilon` chosen just above `epsilon* = gamma/beta`.
3. Confirm the measured `h_SP - h_PO` matches (6a) (`O(epsilon)`) and the measured
   performative-risk gap matches (6b) (`O(epsilon^2)`), with cross-seed IQR bands.

**Remaining code task (out of scope here).** Implement
`equilibrium/perfgd_loop.py`: identical to the RRM loop but adding the scalar
correction `Delta(h_k) = -P*(h_k - psi)*epsilon(h_k)` to the policy gradient,
with `epsilon(h)` and `psi` imported from a shared analytic module (the
`analytic_boundary.py` recommended in 1.1 §8). The math here fixes that module's
formulas completely; no estimator is needed.

---

## 9. Honest caveats

- **`gamma_PO > 0` is conditional (§4.3).** In the defensive wide-spread regime
  `c_t*h_PO > 2 + c_t*psi` the performative second-order term turns destabilising
  and `gamma_PO` can fall below `gamma`; PerfGD then needs a smaller step or a
  trust region. The "stable beyond `epsilon*`" claim is stated with this
  condition attached, not as unconditional.
- **`dT J = P*(h - psi)` assumes the frozen-`T` objective of 1.1 (A3).** Once the
  operator is genuinely un-blinded inside a deployment, the within-deployment
  marking changes; the correction derived here is the *first-order* un-blinding
  (the leading `dD/dphi` term), which is exactly what PerfGD targets, but
  higher-order self-consistency (the full implicit `D = T(D, phi)` solve of the
  fixed-point objective) is a further step.
- **Single bond, zero skew, reference-state `psi`/`gbar`/`lambda_q`** - same scope
  limits as 1.1 §9; the gap (6a–6b) is a per-state quantity to be reported as a
  band over the reference-state distribution.
- **Linearised gap.** (6a)/(6b) are leading-order in `epsilon`; for large
  `epsilon` solve the two FOCs exactly (both are 1-D root-finds) rather than using
  the linearisation.

---

## References

- J. Perdomo, T. Zrnic, C. Mendler-Dünner, M. Hardt. *Performative Prediction.*
  ICML 2020. - stable point vs. performative optimum; the `O(epsilon^2)`
  suboptimality of the stable point.
- Z. Izzo, L. Ying, J. Zou. *How to Learn when Data Reacts to Your Model:
  Performative Gradient Descent.* ICML 2021. - the PerfGD algorithm, here run with
  the *exact* analytic `dD/dphi` from 1.1 instead of an estimated one.
- C. Mendler-Dünner, J. Perdomo, T. Zrnic, M. Hardt. *Stochastic Optimization for
  Performative Prediction.* NeurIPS 2020. - convergence rates for the corrected
  dynamics.
- Builds on [`01-analytic-stability-boundary.md`](01-analytic-stability-boundary.md);
  see [`../references.bib`](../references.bib) and
  `literature/literature-raghav/README.md` for the full citation map.
