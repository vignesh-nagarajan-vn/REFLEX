# 1.1 — Analytic Stability Boundary for a Performative Market Maker

**STATUS: DONE (Priority 1).** This document discharges methodology target
**1.1** in [`../README.md`](../README.md). It derives, in closed form, the
performative-prediction contraction modulus `m` of the REFLEX
policy-distribution loop directly from the microstructure primitives of the
`endo_market_v2` simulator, and obtains the stability boundary

```
   stable   <==>   m = epsilon * beta / gamma  <  1   <==>   epsilon < gamma / beta
```

with **`gamma`, `beta`, and `epsilon` each given as a closed-form function of the
model parameters** — none swept, none tuned. The result is then turned into a
falsifiable cross-check against the existing model-free estimator in
[`analysis/response_modulus.py`](../../endo_market_v2/endo_market/analysis/response_modulus.py).

> **Why this is the headline contribution.** Perdomo et al. (ICML 2020) prove
> that the repeated-retraining map contracts iff `epsilon < gamma/beta`, but
> treat `gamma` (strong convexity), `beta` (joint smoothness) and `epsilon`
> (distribution sensitivity) as *abstract Lipschitz constants of an unspecified
> loss*. For a structural market-making model these constants are not free: they
> are pinned by the fill-intensity curve, the adverse-selection channel, and the
> quoting-cost friction. We compute them. The deliverable is an **a-priori
> stability boundary** — predict whether a given bond/dealer configuration
> converges *before running a single RRM loop*, then verify the prediction
> against the model-free best-response (BR) slope probe. That predict-then-verify
> loop is the ICAIF-grade evidentiary bar.

**Formatting note.** This file deliberately uses fenced code blocks and
backtick-Unicode for all mathematics so it renders identically in every Markdown
viewer (no LaTeX/MathJax dependency). Greek letters are written out as Unicode
glyphs (`gamma`, `beta`, `epsilon`, ...) inside `code` spans; multi-line
derivations live in ```` ``` ```` blocks. Every symbol maps to a config field in
the table in §7.

---

## 0. Notation

| Symbol | Reading | Definition / source |
|--------|---------|---------------------|
| `h` | central half-spread (the policy coordinate we analyse) | bias `b_h` of `LinearPolicy` |
| `h*` | self-consistent fixed-point half-spread | root of the FOC, §4 |
| `h_dep` | *deployed* half-spread that sets the toxic environment | §1, A3 |
| `J(h; T)` | expected one-step dealer objective at toxic level `T` | §2 |
| `tau(h)` | expected informed (toxic) notional at spread `h` | §2 |
| `T` | frozen toxic level inside one deployment, `T = tau(h_dep)` | §1, A3 |
| `epsilon` | distribution sensitivity, `= |d tau / d h|` | §3.1 |
| `beta` | joint smoothness, `= |d^2 J / dh dT|` | §3.2 |
| `gamma` | strong convexity, `= - d^2 J / dh^2` | §3.3 |
| `m` | contraction modulus, `= |BR'(h*)| = epsilon*beta/gamma` | §3.4 |
| `rho` | liquidity ratio `liquidity / liq_mean` at the reference state | A2 |
| `g`, `u` | mispricing `v - m`, and standardized `u = g / sigma_s` | §2 |
| `gbar(u)` | gate mean, a 1-D Gaussian integral | §2 |

---

## 1. Setup and assumptions

We analyse the loop at the level of the **central half-spread** `h` — the bias
coordinate `b_h` of the dealer policy
([`policy/dealer_policy.py`](../../endo_market_v2/endo_market/policy/dealer_policy.py)),
which the convergence study identifies as the dominant coordinate of the
policy-to-distribution iterate map. The model-free probe
[`response_modulus.py`](../../endo_market_v2/endo_market/analysis/response_modulus.py)
deploys *constant-`h`, zero-skew* policies; we adopt the same restriction so the
analytic and empirical moduli measure the same object.

- **A1 (single representative bond, zero skew).** One bond, `skew = 0`. The
  cross-sectional factor structure (`bonds.py`) rescales but does not change the
  scalar boundary; the multi-bond lift is Priority 1.5.
- **A2 (reference state).** Curvature is evaluated at the probe's reference state
  `s0 = simulator.reset(...)` with inventory `q0 ~ 0`, mispricing `g = v - m`,
  and liquidity ratio `rho = liquidity / liq_mean`. Expectations below are over
  the client-flow noise (arrival, imbalance, informed signal) at that state.
- **A3 (frozen-environment best response).** This is the structural content of
  "the operator is blind to `dD/dphi`" (CLAUDE.md; `operator/`). Within a single
  deployment the learned operator `T_theta` conditions on a **frozen**
  `policy_summary` of the deployed regime. So when the dealer computes its best
  response it sees the *toxic-flow environment* as fixed at the level induced by
  the **deployed** spread `h_dep` (write `T = tau(h_dep)`), while the benign
  demand-elasticity channel still responds to its candidate `h`. The performative
  feedback enters only *across* deployments — exactly the cobweb the loop
  iterates. Made precise in §3.
- **A4 (saturation non-binding).** The informed-flow cap
  `info_cap * tanh(. / info_cap)` (`clients.py`) is in its linear regime at the
  operating point; it "rarely binds" by construction. We carry the uncapped form
  and flag the saturated regime as a documented corollary (§6.3).

---

## 2. The expected single-step objective `J(h)`

### 2.1 Components from the simulator

From [`simulator.py`](../../endo_market_v2/endo_market/env/simulator.py), at zero
skew, with `S` = dealer sells (clients lift the ask), `B` = dealer buys (clients
hit the bid), `q_after = q0 + B - S`, `v` = fundamental, `v'` = next fundamental,
`m` = mid:

```
   spread_capture          = (S + B) * h
   inventory_pnl           = q_after * (v' - v)
   adverse_selection_loss  = (B - S) * (m - v)
```

The dealer objective
([`objective/reward.py`](../../endo_market_v2/endo_market/objective/reward.py))
is

```
   J = pnl_scale * [ spread_capture + inventory_pnl
                     - adverse_selection_loss
                     - inv_risk_weight * q_after^2
                     - w * (h - h_ref)^2 ]
```

with `w = quote_anchor_weight`, `h_ref = quote_anchor_ref`. Write
`P := pnl_scale` for brevity. We now take the expectation of each term over the
flow noise at the reference state, using the flow model in
[`clients.py`](../../endo_market_v2/endo_market/env/clients.py).

### 2.2 Uninformed (benign) flow

Benign notional has the Guéant–Lehalle–Fernández-Tapia / Avellaneda–Stoikov
exponential fill intensity

```
   U(h) = A * exp(-k*h) * rho ,        A = base_arrival_rate ,  k = demand_elasticity
```

split roughly 50/50 into buys/sells with a zero-mean idiosyncratic imbalance.
Hence

```
   E[S_uninf] = E[B_uninf] = (1/2) U(h) ,        E[S_uninf - B_uninf] = 0 .
```

### 2.3 Informed (toxic) flow

With `signal = g + sigma_s * eta`, `eta ~ N(0,1)`, `sigma_s = info_signal_noise`,
the gate is `gate = tanh(|signal| / sigma_s)`. The informed notional is

```
   tau(h) = rho * gbar * ( I_b + alpha*f*I * exp(-c_t*h) )
```

where

```
   I_b   = info_base_intensity      (alpha-independent baseline toxic level)
   I     = info_intensity           (scale of the spread-responsive slope term)
   alpha = alpha                    (adversariality)
   f     = toxicity_feedback        (explicit performative-feedback gain)
   c_t   = info_spread_decay        (decay rate of toxic responsiveness in h)
   gbar  = E_eta[ tanh(|g + sigma_s*eta| / sigma_s) ]   (the gate mean; see 2.5)
```

Informed traders pick the dealer off, so the signed informed flow is
`E[(B - S)_inf] = - tau(h) * sign(g)` (modulo signal noise), and the **expected
adverse-selection loss** is

```
   E[adverse] = psi * tau(h) ,
```

where `psi > 0` is the per-unit adverse severity defined in 2.5.

### 2.4 Inventory terms vanish or are second order

The fundamental increment `v' - v = sigma_v * shock` is drawn *after*, and
independently of, `q_after`, so

```
   E[inventory_pnl] = E[q_after] * E[v' - v] = E[q_after] * 0 = 0     (every h).
```

The quadratic risk `E[q_after^2]` is `O(q0^2)` plus a flow-variance term that is
second order at the `q0 ~ 0` reference. We carry it as a small additive curvature
`lambda_q >= 0`; its only effect below is to *raise* `gamma` (it is stabilising)
and it never changes the boundary's functional form.

### 2.5 The two Gaussian integrals (gate mean and adverse severity)

Standardize the mispricing as `u = g / sigma_s`. Because `tanh` is odd,
`tanh(|x|)*sign(x) = tanh(x)`, the two state constants reduce to clean 1-D
Gaussian integrals in `u` alone:

```
   gbar(u) = E_{eta~N(0,1)} [ tanh(|u + eta|) ]          (gate mean)
   a(u)    = E_{eta~N(0,1)} [ tanh(u + eta) ]            (signed gate mean)
   psi     = sigma_s * u * a(u) / gbar(u)  >  0          (adverse severity)
```

Both `gbar` and `a` are smooth, bounded, monotone-in-`|u|` integrals evaluated by
1-D quadrature (or, for the half-normal moments, in terms of `erf`). They are
*state* constants — they do not depend on `h` — so they pass through every
`h`-derivative untouched. This is what makes the boundary a clean function of `h`.

### 2.6 Assembled objective

Substituting 2.2–2.5 into 2.1 and writing the toxic level as the frozen
`T = tau(h_dep)` (A3):

```
   J(h; T) = P * [   h * A * exp(-k*h) * rho      (uninformed spread capture)
                   + h * T                        (toxic spread capture)
                   - psi * T                      (adverse selection, const in h)
                   - w * (h - h_ref)^2            (quoting cost)
                   - lambda_q ]                   (inventory-risk curvature)
```

**Key structural fact.** The adverse term `- psi*T` is *constant in `h`* (the
toxic level is frozen within a deployment), so it drops out of every
`h`-derivative below. It sets the *level* of profit — the "echo-chamber gap" of
Priority 1.2 — but **adverse selection does not enter the stability boundary**,
only the optimal-vs-stable profit gap. This is itself a clean, reportable result.

---

## 3. The best-response map and its slope

### 3.1 The map

The dealer best-responds by maximising `J(h; T)` over `h` at the frozen toxic
level `T = tau(h_dep)`:

```
   BR(h_dep) = argmax_h  J( h ; tau(h_dep) ) .
```

The RRM iteration is `h_{k+1} = BR(h_k)`. Linearising at a fixed point `h*`,

```
   h_{k+1} - h*  ~  BR'(h*) * (h_k - h*) ,
```

so the loop contracts iff the **modulus** `m := |BR'(h*)| < 1` — precisely the
object `response_modulus.py` estimates by common-random-number finite
differences.

### 3.2 First-order condition and the three derivatives

Differentiate `J` in `h`. Define `F(h, T) := dJ/dh`:

```
   dJ/dh = F(h, T)
         = P * [ A*rho*exp(-k*h)*(1 - k*h)  +  T  -  2*w*(h - h_ref) ] .       (FOC: F = 0)
```

We need three second-order quantities. Compute each explicitly.

**(i) Curvature in `h` (gives `gamma`).** Differentiate `F` in `h`. Only the
uninformed term and the quoting cost depend on `h` (the toxic term `h*T` is
linear in `h`, so its second derivative is 0):

```
   d/dh [ A*rho*exp(-k*h)*(1 - k*h) ]
       = A*rho * [ (-k)*exp(-k*h)*(1 - k*h) + exp(-k*h)*(-k) ]
       = A*rho*exp(-k*h) * [ -k*(1 - k*h) - k ]
       = A*rho*exp(-k*h) * [ k^2*h - 2*k ]
       = A*rho*k*exp(-k*h) * (k*h - 2) .

   d/dh [ -2*w*(h - h_ref) ] = -2*w .

   ==>  d^2 J / dh^2 = P * [ A*rho*k*exp(-k*h)*(k*h - 2) - 2*w ] .
```

Therefore the strong-convexity constant is

```
   gamma := - d^2 J / dh^2
          = P * [ 2*w  +  A*rho*k*exp(-k*h)*(2 - k*h)  +  lambda_q ] .          (*)
```

**(ii) Cross sensitivity to the distribution (gives `beta`).** Differentiate `F`
in the frozen toxic level `T`. Only the toxic spread-capture term `h*T`
contributes a `T`-dependence to the gradient (coefficient `+1`); the constant
adverse term `-psi*T` contributes nothing to `dJ/dh`:

```
   d^2 J / dh dT = d/dT [ P*( A*rho*exp(-k*h)*(1-k*h) + T - 2*w*(h-h_ref) ) ]
                 = P * 1 = P .

   ==>  beta := | d^2 J / dh dT | = P = pnl_scale .                            (**)
```

The smoothness constant *is* the global P&L scale — the cleanest possible
identification.

**(iii) Sensitivity of the distribution to the deployed spread (gives
`epsilon`).** Differentiate the toxic level `tau` in the deployed spread:

```
   d tau / d h = rho*gbar * alpha*f*I * (-c_t) * exp(-c_t*h) .

   ==>  epsilon := | d tau / d h | = rho * gbar * alpha * f * I * c_t * exp(-c_t*h) . (***)
```

`epsilon` is the slope of the toxic-flow response to the deployed spread —
**tighter quotes summon more toxic flow** — and is *linear in `alpha` and in
`toxicity_feedback` `f`*, confirming the `clients.py` design claim that `f` is
"the explicit gain of toxicity-vs-spread feedback (~epsilon scaler)".

### 3.3 Implicit-function theorem: the BR slope

Differentiate the fixed-point identity `F( h*(h_dep), tau(h_dep) ) = 0` totally
in `h_dep`:

```
   F_h * (d h* / d h_dep)  +  F_T * (d tau / d h_dep)  =  0
```

with `F_h = d^2 J/dh^2 = -gamma`, `F_T = d^2 J/dh dT = +beta`,
`d tau/d h_dep = -epsilon`. Solving,

```
   d h* / d h_dep = - F_T * (d tau / d h_dep) / F_h
                  = - (beta) * (-epsilon) / (-gamma)
                  = - beta * epsilon / gamma .
```

So `BR'(h_dep) = - beta*epsilon/gamma` — **negative slope**. The cobweb is
*oscillatory*: deploying wider invites a narrower best response and vice versa.

### 3.4 The modulus and the boundary

Taking magnitudes and evaluating at the fixed point `h*`:

```
   m(h*) = | BR'(h*) | = epsilon * beta / gamma                                  (boxed result)

         =        rho * gbar * alpha*f*I * c_t * exp(-c_t*h*)
            -----------------------------------------------------------
              2*w  +  A*rho*k*exp(-k*h*)*(2 - k*h*)  +  lambda_q
```

(the common factor `P = pnl_scale` cancels between `beta` in the numerator and
`gamma` in the denominator). The loop is **stable iff**

```
   epsilon < gamma / beta
   <==>
   rho*gbar*alpha*f*I*c_t*exp(-c_t*h*)  <  2*w + A*rho*k*exp(-k*h*)*(2 - k*h*) + lambda_q .
```

Because every quantity on both sides is a closed form in the config and `h*`,
this inequality is an **a-priori predictor**: given `(config)` and the fixed
point `h*` (§4), evaluate the boundary without running the loop. Note `m = |BR'|`
is exactly what `response_modulus.py` returns (symmetric finite difference of
`BR`), so `m(h*)` above is the directly comparable prediction (§8).

---

## 4. Locating the fixed point `h*`

The boundary is evaluated at the deployed fixed point: the spread that, when
deployed, is its own best response. Setting the FOC `F(h, tau(h)) = 0` with the
toxic level evaluated self-consistently at the same `h`:

```
   A*rho*exp(-k*h*)*(1 - k*h*)  +  tau(h*)  -  2*w*(h* - h_ref)  =  0 ,
   with  tau(h*) = rho*gbar*( I_b + alpha*f*I*exp(-c_t*h*) ) .
```

This is a smooth scalar equation; solve by Newton or bisection on
`[0, max_half_spread]`. A unique interior root exists wherever `gamma(h) > 0` on
the operating range (strict concavity of `J` in `h`) — which is exactly the
stability condition holding with room to spare.

---

## 5. Orientation of the performative coupling (honest treatment)

The *magnitude* `m = epsilon*beta/gamma` and the boundary `epsilon < gamma/beta`
are unambiguous (§3). The *sign/orientation* of the coupling — whether feedback
damps or amplifies a given update rule — deserves an honest statement, because
the codebase itself documents that "the [modulus's] sign isn't robust to universe
size" (CLAUDE.md).

- **Full best response (RRM / the BR-slope probe).** `BR'(h*) = -beta*epsilon/gamma`
  is negative: an *oscillatory cobweb* that converges when `beta*epsilon/gamma < 1`
  and diverges (with growing oscillations) when it exceeds 1. This is exactly the
  Perdomo `m < 1` criterion and is what `response_modulus.py` measures.

- **Damped gradient deployment (RGD, the default `update_rule="rgd"`).** One
  gradient step is `M(h) = h + eta * G(h)`, with `G(h) = dJ/dh` evaluated at the
  current deployment (frozen toxic level `T = tau(h)`). Differentiating,

  ```
     G'(h) = F_h + F_T * (d tau / d h) = -gamma + beta*(-epsilon) = -(gamma + beta*epsilon)
     M'(h*) = 1 + eta * G'(h*) = 1 - eta*(gamma + beta*epsilon) .
  ```

  In *this* spread-capture-dominated single-step objective the coupling is
  **stabilising** for RGD: a small damped step is contractive and the feedback
  only helps. The destabilising orientation `gamma_eff = gamma - beta*epsilon`
  quoted in the methodology README arises when the *inventory-carry and
  liquidity-degradation* channels (multi-step: `liq_overtighten_decay`,
  adverse-selection-driven inventory) dominate the single-step spread-capture
  channel. Perdomo's sufficient condition bounds the coupling by `|.| <=
  beta*epsilon` and therefore uses the worst-case orientation

  ```
     gamma_eff = gamma - beta*epsilon  >  0   <==>   epsilon < gamma/beta ,
  ```

  recovering the *same* boundary conservatively. **Takeaway:** the boundary
  `epsilon < gamma/beta` is robust (it is the magnitude condition); which side of
  a damped update the feedback lands on is model-channel-dependent, and the
  BR-slope probe measures the net orientation empirically.

This honesty is faithful to the project's own documented behaviour and is the
correct thing to state for an ICAIF reviewer, who will otherwise (correctly)
object that a single sign cannot be asserted across regimes.

---

## 6. Corollaries (each a falsifiable prediction)

### 6.1 Linear-in-adversariality scaling
At fixed `h*`, `m` is linear in the product `alpha*f`. The critical feedback gain
at which `m = 1` is

```
   (alpha*f)*  =  gamma  /  ( rho * gbar * I * c_t * exp(-c_t*h*) ) .
```

This predicts the `toxicity_feedback` value at which `sweep_feedback.yaml`
crosses the boundary, *before the sweep is run*.

### 6.2 Lazy-deploy raises effective curvature
Taking `K` gradient steps before redeploying compounds the per-step contraction
`|M'|^K` against a fixed environment, so larger `K` enlarges the stable region in
`(eta)` — the `gamma_eff = gamma - beta*epsilon` mechanism of §5, now derived
from `M'`.

### 6.3 The modulus saturates rather than blows up (documented gotcha, explained)
As the dealer defends by widening, `h*` grows and the GLFT curvature term
`A*rho*k*exp(-k*h*)*(2 - k*h*) -> 0`, so `gamma -> 2*w*P` (a *floor*, not zero)
while `epsilon = rho*gbar*alpha*f*I*c_t*exp(-c_t*h*)` also decays. Their ratio
approaches a finite plateau,

```
   m_inf  ~  epsilon(h*) / (2*w)        (wide-spread regime, gamma at its floor),
```

rather than diverging — the analytic explanation of the empirically observed
"modulus saturates (~1.25) past the boundary" behaviour noted in `CLAUDE.md`.
(Note the GLFT term even turns *negative* once `k*h* > 2`, lowering `gamma`; this
is the defensive-widening low-curvature region.)

### 6.4 Why `epsilon` (not `alpha`) is the clean control variable
Sweeping `alpha` moves *both* `epsilon` (numerator, proportional to `alpha`) and
`h*` (the dealer widens), which feeds back through `exp(-c_t*h*)` and the `gamma`
curvature term — so the net `m(alpha)` is confounded and its sign is not robust
to universe size. Sweeping `toxicity_feedback` `f` at a fixed structural regime
moves `epsilon` *linearly* with `h*` far less perturbed, isolating the
performative channel. This re-derives, from the closed form, the `CLAUDE.md`
instruction to use `sweep_feedback.yaml` for the headline result.

---

## 7. Symbol -> config map

| Symbol | Meaning | Config field | Default |
|--------|---------|--------------|---------|
| `A` | uninformed arrival scale | `clients.base_arrival_rate` | 1.0 |
| `k` | demand elasticity (GLFT fill decay) | `clients.demand_elasticity` | 1.5 |
| `I_b` | toxic baseline level | `clients.info_base_intensity` | 0.60 |
| `I` | toxic slope scale | `clients.info_intensity` | 1.4 |
| `alpha` | adversariality | `clients.alpha` | 0.15 |
| `f` | performative feedback gain | `clients.toxicity_feedback` | 0.22 |
| `c_t` | toxic spread-decay | `clients.info_spread_decay` | 1.5 |
| `sigma_s` | informed signal noise | `clients.info_signal_noise` | 0.6 |
| `w` | quoting-cost convexity | `reward.quote_anchor_weight` | 0.25 |
| `h_ref` | quoting-cost anchor | `reward.quote_anchor_ref` | 1.0 |
| `P = pnl_scale` | global P&L scale (= `beta`) | `reward.pnl_scale` | 1.0 |
| `lambda_q` | inventory-risk curvature (proportional to `inv_risk_weight`) | `reward.inv_risk_weight` | 0.05 |
| `rho` | liquidity ratio at reference | state (`liquidity/liq_mean`) | ~1 |
| `gbar`, `psi` | gate mean / adverse severity | derived Gaussian integrals in `u = g/sigma_s` (§2.5) | — |

---

## 8. Validation protocol (predict -> verify)

The closed form is only a contribution if it agrees with the model-free probe.
The check requires **no new infrastructure**:

1. **Predict.** For each `toxicity_feedback` `f` in `sweep_feedback.yaml`,
   solve for `h*` (§4 root-find), then evaluate `m_pred(f) = epsilon*beta/gamma`
   from (`*`), (`**`), (`***`) in §3.
2. **Measure.** Run
   [`measure_response_modulus`](../../endo_market_v2/endo_market/analysis/response_modulus.py)
   at the same `h_ref ~ h*` to obtain the common-random-number BR-slope estimate
   `m_meas(f)`.
3. **Compare.** Report `m_pred` vs `m_meas` across the sweep with cross-seed IQR
   bands (the median+IQR requirement of the methodology). The evidentiary claim
   is: the predicted boundary `(alpha*f)*` of §6.1 coincides with the measured
   crossing `m_meas = 1`, and `m_pred ~ m_meas` in the contracting regime
   (`m < 1`), diverging only in the saturated regime where §6.3 predicts the
   plateau.

A reference implementation of steps 1–3 belongs in
`endo_market_v2/endo_market/analysis/` as `analytic_boundary.py` (a pure function
of `Config`), reusing `response_modulus.py` for step 2 — the recommended next
code task now that this derivation is accepted.

---

## 9. Honest caveats (for the paper's limitations paragraph)

- **`beta = pnl_scale` is exact only under the frozen-environment FOC (A3).** If a
  future operator models `dD/dphi` (the PerfGD fix, Priority 1.2), the toxic
  spread-capture term `h*tau(h)` becomes `h`-responsive and `beta` acquires a
  `- h*c_t` correction; the boundary form `epsilon < gamma/beta` survives but
  `beta` is no longer a bare constant. This is the intended bridge to 1.2, not a
  defect.
- **The coupling sign is channel-dependent (§5).** The boundary magnitude is
  robust; the orientation under a damped update rule is not asserted as a single
  sign, consistent with the documented "sign isn't robust to universe size".
- **`lambda_q`, `gbar`, `psi` are state-dependent.** The boundary is stated at the
  probe reference state; the phase diagram (Priority 1.5) must report it as a band
  over the reference-state distribution, not a single number.
- **The cap (A4) is assumed slack.** Deep in the unstable regime
  `info_cap*tanh` engages and re-convexifies the problem; the analytic modulus is
  valid up to, and slightly past, the boundary — exactly the region of interest —
  but not arbitrarily far into divergence.
- **Single bond, zero skew (A1).** The scalar boundary is the base case; the
  factor-model lift to `gamma_joint` (Priority 1.5) and the multi-dealer
  `epsilon < gamma/(N*beta)` (Priority 1.3) build on it but are out of scope here.

---

## References

- J. Perdomo, T. Zrnic, C. Mendler-Dünner, M. Hardt. *Performative Prediction.*
  ICML 2020. — the `m = epsilon*beta/gamma` contraction criterion this work makes
  structural.
- M. Avellaneda, S. Stoikov. *High-frequency trading in a limit-order book.*
  Quant. Finance 2008; O. Guéant, C.-A. Lehalle, J. Fernández-Tapia (GLFT),
  2013. — the exponential fill intensity `lambda(delta) = A*exp(-k*delta)`
  underlying `gamma`.
- See [`../references.bib`](../references.bib) and
  `literature/literature-raghav/README.md` for the full citation map (Barzykin
  et al. 2025 for the `d tau / d h` adverse-selection grounding;
  Cont–Kukanov–Stoikov for the informed-flow slope used in the triangulation
  extension of 1.1).
