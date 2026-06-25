# 1.1 — Analytic Stability Boundary for a Performative Market Maker

**Status: derived and complete (Priority 1).** This document discharges
methodology target **1.1** in
[`new-methodology/README.md`](../README.md): it derives, in closed form, the
performative-prediction contraction modulus `m` of the REFLEX
policy↔distribution loop directly from the microstructure primitives of the
`endo_market_v2` simulator, and obtains the stability boundary

```
ε  <  γ / β          ⇔          m = εβ/γ  <  1
```

with **`γ`, `β`, and `ε` each given as a closed-form function of the model
parameters** — none swept, none tuned. The result is then expressed as a
falsifiable cross-check against the existing model-free estimator in
[`analysis/response_modulus.py`](../../endo_market_v2/endo_market/analysis/response_modulus.py).

> **Why this is the headline contribution.** Prior work on performative
> prediction (Perdomo et al., ICML 2020) proves that the repeated-retraining
> map contracts iff `ε < γ/β`, but treats `γ` (strong convexity), `β` (joint
> smoothness) and `ε` (distribution sensitivity) as *abstract Lipschitz
> constants of an unspecified loss*. For a structural market-making model these
> constants are not free: they are pinned by the fill-intensity curve, the
> adverse-selection channel, and the quoting-cost friction. We compute them.
> The deliverable is an **a-priori stability boundary** — you can predict
> whether a given bond/dealer configuration converges *before running a single
> RRM loop*, then verify the prediction against the model-free BR-slope probe.
> That predict-then-verify loop is the ICAIF-grade evidentiary bar.

Notation follows the surrounding code: ASCII names in prose where the code uses
them, Unicode (`φ, ε, γ, β, τ`) in display math. Every symbol is mapped to its
config field in §6.

---

## 1. Setup and assumptions

We analyse the loop at the level of the **central half-spread** `h` — the bias
coordinate `b_h` of the dealer policy
([`policy/dealer_policy.py`](../../endo_market_v2/endo_market/policy/dealer_policy.py)),
which the convergence study identifies as the dominant coordinate of the
policy→distribution iterate map. The model-free probe
[`response_modulus.py`](../../endo_market_v2/endo_market/analysis/response_modulus.py)
deploys *constant-`h`, zero-skew* policies; we adopt the same restriction so the
analytic and empirical moduli measure the same object.

**A1 (single representative bond, zero skew).** One bond, `skew = 0`. The
cross-sectional factor structure (`bonds.py`) rescales but does not change the
scalar boundary; the multi-bond lift is Priority 1.5.

**A2 (reference state).** Curvature is evaluated at the probe's reference state
`s₀ = simulator.reset(...)` with inventory `q₀ ≈ 0`, mispricing `g = v − m`, and
liquidity ratio `ρ = liquidity / liq_mean`. Expectations below are over the
client-flow noise (arrival, imbalance, informed signal) at that state.

**A3 (frozen-environment best response).** This is the structural content of
"the operator is blind to `dD/dφ`" (CLAUDE.md; `operator/`). Within a single
deployment the learned operator `T_θ` conditions on a **frozen** `policy_summary`
of the deployed regime. Hence the dealer, when computing its best response,
sees the *toxic-flow environment* as fixed at the level induced by the
**deployed** spread `h_dep`, while the benign demand-elasticity channel still
responds to its candidate `h`. The performative feedback enters only across
deployments — exactly the cobweb the loop iterates. We make this precise in §3.

**A4 (saturation non-binding).** The informed-flow cap
`info_cap·tanh(·/info_cap)` (`clients.py`) is in its linear regime at the
operating point; it "rarely binds" by construction. We carry the uncapped form
and flag the saturated regime as a documented corollary (§5.3).

---

## 2. The expected single-step objective `J(h)`

From [`simulator.py`](../../endo_market_v2/endo_market/env/simulator.py) the
three P&L components at zero skew are

```
spread_capture        = (S + B) · h
inventory_pnl         = q_after · (v' − v)
adverse_selection_loss = (B − S) · (m − v)
```

where `S` (dealer sells / clients lift the ask) and `B` (dealer buys / clients
hit the bid) decompose into uninformed + informed parts. The dealer objective
([`objective/reward.py`](../../endo_market_v2/endo_market/objective/reward.py))
is

```
J = pnl_scale · [ spread_capture + inventory_pnl − adverse_selection_loss
                  − inv_risk_weight · q_after²  −  w · (h − h_ref)² ]
```

with `w = quote_anchor_weight`, `h_ref = quote_anchor_ref`. Take expectations
term by term, using the flow model in
[`clients.py`](../../endo_market_v2/endo_market/env/clients.py).

**Uninformed flow.** Benign notional has GLFT exponential fill intensity

$$
U(h) \;=\; A\,e^{-k h}\,\rho,\qquad A=\texttt{base\_arrival\_rate},\; k=\texttt{demand\_elasticity},
$$

split ≈50/50 buy/sell with zero-mean idiosyncratic imbalance, so
`E[S_uninf] = E[B_uninf] = ½U(h)` and `E[S_uninf − B_uninf] = 0`.

**Informed (toxic) flow.** With `gate = tanh(|signal|/σ_s)`,
`signal = g + σ_s η`, `σ_s = info_signal_noise`, the informed notional is

$$
\tau(h) \;=\; \rho\,\bar g\,\bigl(I_b + \alpha f I\,e^{-c_t h}\bigr),
\qquad
\bar g := \mathbb{E}_\eta[\tanh(|g+\sigma_s\eta|/\sigma_s)],
$$

with `I_b = info_base_intensity`, `I = info_intensity`, `α = alpha`,
`f = toxicity_feedback`, `c_t = info_spread_decay`. Informed traders pick the
dealer off, so `E[(B−S)_inf] = −τ(h)\,\mathrm{sign}(g)` (modulo signal noise) and
the **expected adverse-selection loss** is

$$
\mathbb{E}[\text{adv}] \;=\; \psi\,\tau(h),
\qquad
\psi := \frac{\mathbb{E}_\eta[\tanh(|g+\sigma_s\eta|/\sigma_s)\,\mathrm{sign}(g+\sigma_s\eta)]\,g}{\bar g}\;>\;0 .
$$

Both `\bar g` and `ψ` are elementary 1-D Gaussian integrals in `g/σ_s` (evaluate
by quadrature; closed forms in terms of the error function exist for the
half-normal moments).

**Inventory terms.** The fundamental shock `v' − v = σ_v·shock` is drawn
*after*, and independently of, `q_after`, so `E[inventory_pnl] = 0` at every
`h`. The quadratic risk `E[q_after²]` is `O(q₀²)` plus a flow-variance term that
is second order at the `q₀≈0` reference; we carry it as a small additive
curvature `λ_q ≥ 0` (its inclusion only *raises* `γ`, i.e. is stabilising, and
does not change the boundary's form).

**Result.** Collecting terms, and writing the toxic environment level as
`T := τ(h_dep)` per A3:

$$
\boxed{\;
J(h;T)\;=\;\texttt{pnl\_scale}\cdot\Bigl[\,
\underbrace{h\,A e^{-k h}\rho}_{\text{uninformed spread}}
\;+\;\underbrace{h\,T}_{\text{toxic spread}}
\;-\;\underbrace{\psi\,T}_{\text{adverse selection}}
\;-\;\underbrace{w\,(h-h_\text{ref})^2}_{\text{quoting cost}}
\;-\;\lambda_q
\,\Bigr]\;}
$$

The adverse term `−ψT` is **constant in `h`** (the toxic level is frozen within
the deployment); it sets the *level* of profit — the "echo-chamber gap" of
Priority 1.2 — but drops out of every `h`-derivative below. This is itself a
clean structural fact: *adverse selection does not enter the stability boundary,
only the optimal-vs-stable profit gap.*

---

## 3. The best-response map and its slope

The dealer best-responds by maximising `J(h;T)` over `h` at the frozen toxic
level `T = τ(h_dep)`:

$$
\mathrm{BR}(h_\text{dep}) \;=\; \arg\max_h\, J\bigl(h;\,\tau(h_\text{dep})\bigr).
$$

The RRM iteration is `h_{k+1} = BR(h_k)`; linearising at a fixed point `h*` gives
`h_{k+1}-h* ≈ BR'(h*)\,(h_k-h*)`, so the loop contracts iff the **modulus**
`m := |BR'(h*)| < 1` — precisely the object `response_modulus.py` estimates by
common-random-number finite differences.

By the implicit-function theorem applied to the first-order condition
`∂_h J(h;T)=0`,

$$
\mathrm{BR}'(h_\text{dep})
\;=\;
\frac{\partial_h\big[\partial_h J\big]}{-\,\partial_{hh} J}\bigg|^{-1}
\!\!\!\cdot(\cdots)
\;=\;
\frac{\overbrace{\partial^2_{hT} J}^{\;\beta\;}\;\cdot\;\overbrace{\big|\,d\tau/dh_\text{dep}\big|}^{\;\varepsilon\;}}
{\underbrace{-\,\partial^2_{hh} J}_{\;\gamma\;}} .
$$

This is the structural realisation of Perdomo et al.'s
`m = εβ/γ`. We read off each factor.

### 3.1 Distribution sensitivity `ε` (the performative gain)

$$
\varepsilon \;:=\; \Bigl|\frac{d\tau}{dh_\text{dep}}\Bigr|
\;=\; \rho\,\bar g\,\alpha\,f\,I\,c_t\,e^{-c_t h_\text{dep}}\,.
$$

`ε` is the slope of the toxic-flow response to the deployed spread: **tighter
quotes summon more toxic flow.** It is *linear in `α` and in `toxicity_feedback`
`f`* — confirming the design claim in `clients.py` that `f` is "the explicit
gain of toxicity-vs-spread feedback (~ε scaler)" and that the empirical modulus
`m = κ·(dτ/dh)` scales with `α`. Here `κ = β/γ`.

### 3.2 Joint smoothness `β`

`β` measures how strongly a unit shift in the toxic environment moves the
dealer's *marginal* objective:

$$
\beta \;:=\; \bigl|\partial^2_{hT} J\bigr|
\;=\; \texttt{pnl\_scale}\cdot\bigl|\partial_h(hT-\psi T)/\partial T\bigr|
\;=\; \texttt{pnl\_scale}.
$$

The toxic spread-capture term `h·T` contributes coefficient `1` to `∂_h J`; the
constant adverse term contributes nothing. So `β = pnl_scale` (≡ 1 in the
default config). This is the cleanest possible identification: the smoothness
constant *is* the global P&L scale.

### 3.3 Strong convexity `γ`

Differentiating `J` twice in `h` (the toxic term `h·T` is linear, hence drops):

$$
-\,\partial^2_{hh}J
\;=\;\texttt{pnl\_scale}\cdot\Bigl[\,
\underbrace{2w}_{\text{quoting cost}}
\;+\;\underbrace{A\rho\,k\,e^{-k h}\,(2-k h)}_{\text{GLFT fill-curve curvature}}
\;+\;\lambda_q
\,\Bigr]\;=:\;\gamma(h).
$$

Two structural readings:

- The **quoting-cost** floor `2w·pnl_scale` is the irreducible convexity that
  makes the best response finite and the modulus tunable (cf. the `reward.py`
  docstring: the quadratic quoting cost "pins the dealer's optimum and makes the
  best-response sensitivity finite").
- The **GLFT fill-curve curvature** `Aρk e^{-kh}(2-kh)` is the
  Avellaneda–Stoikov/GLFT contribution: it is *positive (stabilising) for
  `kh<2`* and *decays to zero* as `h` grows. This is the analytic origin of the
  documented "γ→0 defensive-widening" regime (§5.3).

### 3.4 The modulus and the boundary

Evaluating at the fixed point `h*`:

$$
\boxed{\;
m(h^*)\;=\;\frac{\varepsilon\,\beta}{\gamma}
\;=\;
\frac{\rho\,\bar g\,\alpha f I\,c_t\,e^{-c_t h^*}}
{\,2w + A\rho\,k\,e^{-k h^*}(2-kh^*)+\lambda_q\,}\;}
$$

(the common `pnl_scale` cancels between `β` and `γ`), and the loop is **stable
iff**

$$
\boxed{\;\varepsilon \;<\; \gamma/\beta
\quad\Longleftrightarrow\quad
\rho\,\bar g\,\alpha f I\,c_t\,e^{-c_t h^*}
\;<\;
2w + A\rho\,k\,e^{-k h^*}(2-kh^*)+\lambda_q\;}
$$

Because every quantity on both sides is a closed form in the config, this
inequality is an **a-priori predictor**: given `(config)` and the fixed-point
`h*` (itself the root of the one-dimensional FOC `∂_h J=0`, §4), one evaluates
the boundary without running the loop.

---

## 4. Locating the fixed point `h*`

The boundary is evaluated at the deployed fixed point, the root of the
first-order condition `∂_h J(h;τ(h)) = 0` *with the toxic level evaluated
self-consistently at the same `h`* (a true fixed point deploys and best-responds
to the same spread). Writing it out:

$$
A\rho\,e^{-k h^*}(1-k h^*)\;+\;\tau(h^*)\;-\;2w\,(h^*-h_\text{ref})\;=\;0 .
$$

This is a smooth scalar equation; solve by Newton or bisection on
`[0, max_half_spread]`. (`τ(h*)` is the *frozen* level entering the gradient as
a constant `+T`; its self-consistent value at the fixed point is `τ(h*)`.) The
existence of a unique interior root on the operating range follows from
`γ(h)>0` there (strict concavity of `J` in `h`), which is exactly the stability
condition holding with room to spare.

---

## 5. Corollaries (each a falsifiable prediction)

### 5.1 Linear-in-adversariality scaling
`m ∝ α·f` at fixed `h*`. The critical feedback gain at which `m=1` is

$$
(\alpha f)^* \;=\; \frac{\gamma}{\rho\,\bar g\,I\,c_t\,e^{-c_t h^*}} .
$$

This predicts the `toxicity_feedback` value at which `sweep_feedback.yaml`
crosses the boundary, *before the sweep is run*.

### 5.2 Lazy-deploy / RGD raises effective curvature
Under the default `update_rule="rgd"` with step `η`, one gradient step gives
`h_{k+1} = h_k + η\,∂_h J(h_k;τ(h_k))`, whose linearised modulus is

$$
m_\text{RGD} \;=\; \bigl|1 - \eta(\gamma - \varepsilon\beta)\bigr| .
$$

Contraction requires `γ_eff := γ − εβ > 0`, i.e. the **same** boundary
`ε < γ/β`. Taking `K` lazy steps before redeploying compounds the per-step
contraction, so larger `K` widens the stable region in `(η)` — the
`γ_eff = γ − εβ` identity quoted in the methodology README, now derived.

### 5.3 The modulus saturates rather than blows up (documented gotcha, explained)
As the dealer defends by widening, `h*` grows and the GLFT curvature term
`A\rho k e^{-kh^*}(2-kh^*) → 0`, so `γ → 2w·pnl_scale` (a *floor*, not zero) while
`ε = ρ\bar g\alpha f I c_t e^{-c_t h^*}` also decays. Their ratio approaches a
finite plateau rather than diverging — the analytic explanation of the
empirically observed "modulus saturates (~1.25) past the boundary" behaviour
noted in `CLAUDE.md`. The plateau value is `m_∞ ≈ ε(h*)/(2w)` with both
evaluated in the wide-spread regime.

### 5.4 Why `ε` (not `α`) is the clean control variable
Sweeping `α` moves *both* `ε` (numerator, ∝α) and `h*` (the dealer widens),
which feeds back into `e^{-c_t h*}` and the `γ` curvature term — so the net
`m(α)` is confounded and its sign is not robust to universe size. Sweeping
`toxicity_feedback` `f` at fixed structural regime moves `ε` *linearly* with
`h*` far less perturbed, isolating the performative channel. This re-derives,
from the closed form, the `CLAUDE.md` instruction to use `sweep_feedback.yaml`
for the headline result.

---

## 6. Symbol → config map

| Symbol | Meaning | Config field | Default |
|--------|---------|--------------|---------|
| `A` | uninformed arrival scale | `clients.base_arrival_rate` | 1.0 |
| `k` | demand elasticity (GLFT fill decay) | `clients.demand_elasticity` | 1.5 |
| `I_b` | toxic baseline level | `clients.info_base_intensity` | 0.60 |
| `I` | toxic slope scale | `clients.info_intensity` | 1.4 |
| `α` | adversariality | `clients.alpha` | 0.15 |
| `f` | performative feedback gain | `clients.toxicity_feedback` | 0.22 |
| `c_t` | toxic spread-decay | `clients.info_spread_decay` | 1.5 |
| `σ_s` | informed signal noise | `clients.info_signal_noise` | 0.6 |
| `w` | quoting-cost convexity | `reward.quote_anchor_weight` | 0.25 |
| `h_ref` | quoting-cost anchor | `reward.quote_anchor_ref` | 1.0 |
| `pnl_scale` | global P&L scale (=`β`) | `reward.pnl_scale` | 1.0 |
| `λ_q` | inventory-risk curvature (∝`inv_risk_weight`) | `reward.inv_risk_weight` | 0.05 |
| `ρ` | liquidity ratio at reference | state (`liquidity/liq_mean`) | ≈1 |
| `ḡ, ψ` | gate mean / adverse severity | derived (Gaussian integrals in `g/σ_s`) | — |

---

## 7. Validation protocol (predict → verify)

The closed form is only a contribution if it agrees with the model-free probe.
The check requires **no new infrastructure**:

1. **Predict.** For each `toxicity_feedback` `f` in `sweep_feedback.yaml`,
   evaluate `h*` (§4 root-find), then `m_pred(f) = εβ/γ` (§3.4).
2. **Measure.** Run
   [`measure_response_modulus`](../../endo_market_v2/endo_market/analysis/response_modulus.py)
   at the same `h_ref ≈ h*` to obtain the common-random-number BR-slope estimate
   `m_meas(f)`.
3. **Compare.** Report `m_pred` vs `m_meas` across the sweep with cross-seed IQR
   bands (the median+IQR requirement of the methodology). The evidentiary
   claim is: the predicted boundary `f*` (§5.1) coincides with the measured
   crossing `m_meas = 1`, and `m_pred ≈ m_meas` in the contracting regime
   (`m<1`), diverging only in the saturated regime where §5.3 predicts the
   plateau.

A reference implementation of step 1–3 belongs in
`endo_market_v2/endo_market/analysis/` as `analytic_boundary.py` (a pure
function of `Config`), reusing `response_modulus.py` for step 2 — this is the
recommended next code task once the derivation here is accepted.

---

## 8. Honest caveats (for the paper's limitations paragraph)

- **`β = pnl_scale` is exact only under the frozen-environment FOC (A3).** If a
  future operator models `dD/dφ` (the PerfGD fix, Priority 1.2), the toxic
  spread-capture term `h·τ(h)` becomes `h`-responsive and `β` acquires a
  `−h·c_t` correction; the boundary form `ε<γ/β` survives but `β` is no longer a
  bare constant. This is the intended bridge to 1.2, not a defect.
- **`λ_q` and `ḡ, ψ` are state-dependent.** The boundary is stated at the probe
  reference state; the phase diagram (Priority 1.5) must report it as a band
  over the reference-state distribution, not a single number.
- **The cap (A4) is assumed slack.** Deep in the unstable regime
  `info_cap·tanh` engages and re-convexifies the problem; the analytic modulus
  is valid up to, and slightly past, the boundary — exactly the region of
  interest — but not arbitrarily far into divergence.
- **Single bond, zero skew (A1).** The scalar boundary is the base case; the
  factor-model lift to `γ_joint` (Priority 1.5) and the multi-dealer
  `ε < γ/(Nβ)` (Priority 1.3) build on it but are out of scope here.

---

## References

- J. Perdomo, T. Zrnic, C. Mendler-Dünner, M. Hardt. *Performative Prediction.*
  ICML 2020. — the `m = εβ/γ` contraction criterion this work makes structural.
- M. Avellaneda, S. Stoikov. *High-frequency trading in a limit-order book.*
  Quant. Finance 2008; O. Guéant, C.-A. Lehalle, J. Fernández-Tapia (GLFT),
  2013. — the exponential fill intensity `λ(δ)=A e^{-kδ}` underlying `γ`.
- See [`new-methodology/references.bib`](../references.bib) and
  `literature/literature-raghav/README.md` for the full citation map (Barzykin
  et al. 2025 for the `dτ/dh` adverse-selection grounding; Cont–Kukanov–Stoikov
  for the informed-flow slope used in the triangulation extension of 1.1).
