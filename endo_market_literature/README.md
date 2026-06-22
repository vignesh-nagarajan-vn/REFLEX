# Literature for `endo_market`
## Performative Prediction in an Endogenous OTC Bond Market

---

This repository collects **18 papers** that together supply (i) the mathematical scaffolding behind `endo_market` as it currently stands, and (ii) a concrete, ordered research agenda for advancing it to a publication-grade theoretical contribution. The original 10 papers are described in the main project README; this document expands each entry, adds **8 new papers** that open the next frontier, and closes with a **synthesis roadmap** that is more opinionated than the original — naming specific theorems to prove, experiments to run, and failure modes to rule out.

---

## Repository layout

```
endo_market_literature/
├── README.md               ← this document
├── references.bib          ← BibTeX for all 18 papers
├── download_pdfs.sh        ← fetches all 18 open-access PDFs into pdfs/
└── pdfs/                   ← populated by running the script
    ├── 01_perdomo_performative_prediction.pdf
    ├── 02_mendler_dunner_stochastic_pp.pdf
    ├── 03_miller_echo_chamber.pdf
    ├── 04_izzo_performative_gradient_descent.pdf
    ├── 05_drusvyatskiy_xiao_decision_dependent.pdf
    ├── 06_jagadeesan_regret_performative.pdf
    ├── 07_li_wai_state_dependent_pp.pdf
    ├── 08_gueant_lehalle_inventory_risk.pdf
    ├── 09_bergault_gueant_size_matters.pdf
    ├── 10_barzykin_bergault_adverse_selection_2025.pdf
    ├── 11_brown_sandholm_performative_games.pdf
    ├── 12_narang_faulkner_multiplayer_pp.pdf
    ├── 13_cao_shi_distributionally_robust_pp.pdf
    ├── 14_cuturi_peyre_computational_ot.pdf
    ├── 15_avellaneda_stoikov_hft_mm.pdf
    ├── 16_cartea_jaimungal_penalva_almgren.pdf
    ├── 17_lacker_mean_field_games_finance.pdf
    └── 18_cont_kukanov_stoikov_price_impact.pdf
```

To populate `pdfs/`:

```bash
chmod +x download_pdfs.sh
./download_pdfs.sh
```

All entries are open-access arXiv preprints. The script is idempotent — already-downloaded files are skipped.

---

## The project's core object (quick reference)

`endo_market` instantiates a **performative prediction loop** inside a structural OTC bond market:

| Symbol | Project meaning |
|--------|----------------|
| `φ` | Dealer's quoting policy (bid–ask half-spread `h`) |
| `D(φ)` | Order-flow distribution induced by quoting at `h` |
| `ε` | Sensitivity of `D` to `φ` — the "toxicity feedback gain" |
| `γ` | Strong-convexity of the loss in `φ` — "quoting-anchor curvature" |
| `β` | Smoothness of the loss in `φ` |
| `m ≈ εβ/γ` | Best-response contraction modulus |
| `ε*` | Critical gain where `m` crosses 1 — the stability boundary |

**The headline theorem** (Perdomo et al.): repeated risk minimisation (RRM) converges iff `ε < γ/β`, equivalently `m < 1`. The project demonstrates this crossing empirically; the agenda below makes it analytic.

---

## Reading map — all 18 papers

### ── CORE PERFORMATIVITY THEORY (Papers 1–6) ──────────────────────────────

---

#### 1. Performative Prediction
**Perdomo, Zrnic, Mendler-Dünner, Hardt — ICML 2020**
`arXiv:2002.06673` · https://arxiv.org/abs/2002.06673

**What it proves.** Formalises decision-dependent distributions `D(φ)`, introduces performatively stable (PS) and performatively optimal (PO) points, and shows RRM converges to PS with modulus `m = εβ/γ` whenever `ε < γ/β`. If any of the three conditions (β-smoothness, γ-strong-convexity, ε-sensitivity bound) fails, convergence is not guaranteed.

**Project connection.** This is the theorem `endo_market` instantiates. Every parameter in the simulator has a counterpart here.

**Critical reading notes.**
- Theorem 3 (contraction) requires *global* strong-convexity. In the simulator, γ → 0 as the dealer widens defensively into a low-curvature regime. This is not a flaw in the theorem — it is the theorem explaining the saturation.
- The stable ≠ optimal gap (Section 4) is the "echo chamber" the dealer is stuck in. It is bounded by `O(ε²)` under additional regularity; measuring this gap in the simulator is a concrete deliverable.
- The paper treats `ε` as a fixed scalar. Papers 6, 13 fix this.

**Extends the work by:** Converting the empirically measured `ε*` into a predicted one via plug-in estimates of `γ, β, ε`. The prediction is falsifiable — if measured `m` diverges from `εβ/γ`, something in the microstructure assumptions is violated.

---

#### 2. Stochastic Optimization for Performative Prediction
**Mendler-Dünner, Perdomo, Zrnic, Hardt — NeurIPS 2020**
`arXiv:2006.06887` · https://arxiv.org/abs/2006.06887

**What it proves.** Replaces full retraining with stochastic gradient updates. Key quantity: the *effective strong-convexity* `γ_eff = γ − εβ`. The algorithm converges when `γ_eff > 0`, i.e., exactly when `ε < γ/β`. Greedy deploy (publish after every SGD step) and lazy deploy (K steps before redeployment) are both analysed.

**Project connection.** Explains *why* `γ − εβ` is the right object: it is the curvature the optimiser actually sees once feedback is accounted for.

**Critical reading notes.**
- Proposition 2 gives the bias of greedy deploy relative to the PS point — this is the "frozen-summary bias" in the project.
- Lazy deploy with `K` steps is a stabilisation knob: larger `K` effectively reduces the sensitivity the optimiser sees per update, pushing `γ_eff` up.

**Extends the work by:** Building a `γ − εβ` estimator as a low-variance stability diagnostic to report alongside the empirical BR-slope `m`. Sweeping `K` in lazy deploy is a clean ablation: does stability persist longer with delayed redeployment?

---

#### 3. Outside the Echo Chamber: Optimizing the Performative Risk
**Miller, Perdomo, Zrnic — ICML 2021**
`arXiv:2102.08570` · https://arxiv.org/abs/2102.08570

**What it proves.** Analyses `PR(φ) = E_{z~D(φ)}[loss(z; φ)]` directly. Shows stable ≠ optimal in general; gives the location-scale condition under which `PR` is convex and the performative optimum is efficiently reachable. The "echo chamber" formalises the intuition that a learner trained only on its own induced distribution is biased.

**Project connection.** The operator `T_θ` conditions on `policy_summary` frozen at the deployed regime — the literal echo chamber. Defensive widening is the simulator's path to the echo-chamber pathology.

**Critical reading notes.**
- Theorem 3 (convexity of PR under location-scale) is the checkable condition. The toxic-flow term `α · toxicity_feedback · info_intensity · exp(−decay·h)` is close to a location-scale shift in `h`. Verify this.
- The stable-to-optimal gap is `O(ε²)` (Proposition 2). At `ε*` this is non-negligible — it should appear as spread inflation in the phase diagram.

**Extends the work by:** Building a PR solver for the dealer (not just a stability locator). Reporting the optimal-vs-stable spread gap as a second axis of the phase diagram gives the paper a second headline result.

---

#### 4. How to Learn when Data Reacts to Your Model: Performative Gradient Descent
**Izzo, Ying, Zou — ICML 2021**
`arXiv:2102.07698` · https://arxiv.org/abs/2102.07698

**What it proves.** Introduces PerfGD: estimates `dD/dφ` from a parametric model of the induced distribution across deployments and injects it into the gradient, converging to the PO point rather than the PS point.

**Project connection.** The project's single most important stated limitation — `T_θ` "cannot learn `dD/dφ`" — is exactly what PerfGD fixes.

**Critical reading notes.**
- Algorithm 1 requires fitting `dτ/dh` (the toxic-slope map) across deployments. The simulator already sweeps `toxicity_feedback`; logging `(h, τ)` pairs across runs is sufficient to fit this parametrically.
- The convergence guarantee requires the estimated `dD/dφ` to be accurate enough (Assumption 3). In the OTC setting, this means the toxic-slope estimator must be unbiased — a testable condition.

**Extends the work by:** Running the clean ablation: blind RRM vs PerfGD-corrected, showing the corrected loop stabilises past `ε*`. This is the paper's natural "Figure 2" in a journal submission.

---

#### 5. Stochastic Optimization with Decision-Dependent Distributions
**Drusvyatskiy & Xiao — Mathematics of Operations Research 2023**
`arXiv:2011.11173` · https://arxiv.org/abs/2011.11173

**What it proves.** ~60 pages of rigorous convergence theory. SGD/proximal/clipped algorithms on decision-dependent problems behave like the same algorithm on a static problem corrupted by a *vanishing bias*. Explicit rates under strong-monotonicity/convexity.

**Project connection.** The formal backbone for "why RRM converges." The vanishing-bias framing names the cobweb gap between `T_θ`'s equilibrium and the true performative equilibrium.

**Critical reading notes.**
- Assumption (A3) (Lipschitz distribution map) is exactly the `ε`-sensitivity condition. Checking it empirically requires measuring `W_1(D(h), D(h'))` for small perturbations `h' - h` — feasible from simulator logs.
- Section 5 handles non-convex losses with the monotone operator framework — relevant if the quoting loss is non-convex in spread width.

**Extends the work by:** Upgrading the empirical stability claim to a theorem with constants for the specific gradient estimators the project ships (REINFORCE/RGD/pathwise). The bias term becomes a bounded error quantifying how far the learned operator's equilibrium is from truth.

---

#### 6. Regret Minimization with Performative Feedback
**Jagadeesan, Zrnic, Mendler-Dünner — ICML 2022**
`arXiv:2202.00628` · https://arxiv.org/abs/2202.00628

**What it proves.** Exploits the fact that deploying `φ` yields *samples from `D(φ)`*, not just a scalar reward — a richer signal than a bandit. Confidence bounds on unexplored models are built from smoothness of the shift map. Regret scales with the complexity of distribution shifts, not the reward function. Crucially, **does not require convexity**.

**Project connection.** Reframes `ε` as something to explore and learn, not sweep by hand. The operator `T_θ` is exactly the `D(φ)` channel this paper uses.

**Critical reading notes.**
- The method is valid in the project's saturation regime (where Perdomo-style convexity breaks down) — this is its unique advantage.
- Algorithm 1's exploration bonus is over the space of policies, which in the OTC context means the space of quoting regimes `h`. The simulator already parameterises this.

**Extends the work by:** Replacing the 3-seed manual `ε` sweep with an adaptive online exploration that locates `ε*` with regret guarantees. The simulation cost is the same; the result is a learned `ε*` rather than a swept one.

---

### ── STATEFUL & DYNAMIC PERFORMATIVITY (Paper 7) ──────────────────────────

---

#### 7. State-Dependent Performative Prediction with Stochastic Approximation
**Li & Wai — AISTATS 2022**
`arXiv:2110.00800` · https://arxiv.org/abs/2110.00800

**What it proves.** Extends PP to a stateful setting: `D(φ, s)` depends on the current decision *and* a state `s` driven by a controlled Markov chain whose kernel depends on `φ`. Stochastic approximation analysis; `O(1/k)` convergence to the state-dependent PS point.

**Project connection.** Inventory `q_after = q0 + B − S` carries across steps. The current RRM under-models this — the project is genuinely state-dependent.

**Critical reading notes.**
- Assumption (B2) requires the Markov chain to mix fast enough relative to the learning rate. In the simulator, the inventory process mixes at the timescale of round-trip trades — check this against the learning-rate schedule.
- The state-dependent PS point differs from the static PS point by an `O(mixing time)` term. If inventory dynamics are slow, this correction is non-trivial.

**Extends the work by:** Providing the correct convergence theory once inventory is folded into the loop. The Markov-chain formulation matches the simulator's state transition exactly.

---

### ── MARKET MICROSTRUCTURE CONTROL THEORY (Papers 8–10) ──────────────────

---

#### 8. Dealing with the Inventory Risk
**Guéant, Lehalle, Fernández-Tapia — Mathematics and Financial Economics 2013**
`arXiv:1105.3115` · https://arxiv.org/abs/1105.3115

**What it proves.** Canonical tractable market-making control. Exponential intensity `λ(δ) = A·exp(−k·δ)`, HJB collapses to linear ODEs via exponential-utility substitution, closed-form optimal quotes.

**Project connection.** The structural origin of `exp(−info_spread_decay·h)`, the inventory penalty, and the quoting anchor.

**Critical reading notes.**
- The curvature `k` of the exponential intensity *is* `β` (smoothness) in the performativity framing. This gives a closed-form `β = k²·A·exp(−k·δ*)` at the optimal quote.
- The reservation-price drift `γq` (inventory adjustment) is the structural source of the strong-convexity `γ`. Deriving `γ` from the GLFT value function makes the boundary `ε < γ/β` fully analytic.

**Extends the work by:** Replacing heuristic anchor tuning with GLFT closed-form quotes as the baseline BR map. This validates the learned operator and makes `γ/β` a computable number, not a swept parameter.

---

#### 9. Size Matters for OTC Market Makers
**Bergault & Guéant — Mathematical Finance 2021**
`arXiv:1907.01225` · https://arxiv.org/abs/1907.01225

**What it proves.** Multi-asset OTC dealing with correlated inventory. Factor-model dimensionality reduction keeps optimal-quote computation tractable for large universes (100+ assets).

**Project connection.** The project's "scale to 100+ bonds" caveat. The 8-bond universe is a compute constraint, not a scientific one.

**Critical reading notes.**
- The factor reduction preserves the exponential-intensity structure — so the derivation of `γ` and `β` from paper #8 extends to the multi-asset case via the factor loadings.
- Proposition 4 gives the error from the dimensionality reduction as a function of residual variance. This should be reported as an approximation bound when scaling up.

**Extends the work by:** The concrete method to reach 100+ correlated bonds and generate tight `median+IQR` bands on `m` and `ε*` — the "publication-grade phase diagram."

---

#### 10. Optimal Quoting under Adverse Selection and Price Reading
**Barzykin, Bergault, Guéant, Lemmel — arXiv 2025**
`arXiv:2508.20225` · https://arxiv.org/abs/2508.20225

**What it proves.** Stationary infinite-horizon MM model with adverse selection and price reading. Perturbation expansion gives closed-form first-order corrections to optimal quotes and, critically, an analytic `dτ/dh` — the feedback slope.

**Project connection.** The principled derivation of the toxic-flow channel the project currently gates heuristically.

**Critical reading notes.**
- The perturbation expansion is in the informed-trader intensity (small adverse selection). Verify this is in the simulator's operating regime.
- Price reading (the dealer's quotes revealing inventory direction) is *not* in the current model. Adding it could shift `ε*` significantly — this is a prediction, not just a caveat.

**Extends the work by:** Replacing `toxicity_feedback` (a tuning knob) with the analytic `dτ/dh` from this paper. The PerfGD correction in paper #4 then has a closed-form distribution-response term — no estimation required.

---

### ── NEW: EXTENSION PAPERS (Papers 11–18) ─────────────────────────────────

These eight papers are **not in the original literature folder**. Each opens a distinct and concrete research direction that advances `endo_market` beyond its current scope.

---

#### 11. Performative Games
**Brown, Hod, Kalemaj — 2021**
`arXiv:2106.09784` · https://arxiv.org/abs/2106.09784

**What it proves.** Extends performative prediction to a **multi-agent setting**: multiple learners each deploy policies, and the induced distribution depends on *all* agents' policies simultaneously. Introduces performatively stable Nash equilibria (PSNE) and shows they exist under conditions analogous to Perdomo's single-agent case. The contraction modulus generalises: the joint map must be a contraction in the product space.

**Why it matters for `endo_market`.** The current model has a single dealer. Real OTC bond markets have 5–15 primary dealers quoting simultaneously, and each dealer's quotes affect the toxic flow that hits *all* dealers. The single-dealer `ε < γ/β` boundary is a special case of the multi-agent condition. When dealers are symmetric, the effective sensitivity seen by each dealer is `N·ε_bilateral` where `N` is the number of dealers — the stability boundary tightens dramatically.

**Concrete research direction.**
- Add a second dealer to the simulator. Let both run RRM simultaneously.
- Measure whether the joint system's BR-slope modulus crosses 1 at a lower `ε` than the single-dealer case.
- Prove that for `N` symmetric dealers the boundary becomes `ε < γ/(Nβ)` — a falsifiable prediction from the paper's Theorem 2.
- This yields a new axis for the phase diagram: number of dealers vs feedback gain. It also connects to systemic risk: the market-wide instability emerges from individually rational quoting.

**Reading priority:** High. This is the most direct extension of the headline theorem.

---

#### 12. Multiplayer Performative Prediction: Learning in Decision-Dependent Games
**Narang, Faulkner, Drusvyatskiy, Fazel, Ratliff — 2022**
`arXiv:2207.05630` · https://arxiv.org/abs/2207.05630

**What it proves.** Convergence theory for performative Nash equilibria under stochastic gradient dynamics. Whereas paper #11 establishes existence, this paper gives *rates* — O(1/k) convergence under a joint strong-monotonicity condition. Also handles the case where the joint game is not strongly monotone but only satisfies a weaker "variational stability" condition.

**Why it matters for `endo_market`.** Once the multi-dealer extension is built (paper #11), the natural question is whether the joint RRM loop converges and at what rate. This paper provides that theory. The variational-stability condition is weaker than strong monotonicity — it may hold even when individual curvatures `γ` are small (the defensive-widening regime).

**Concrete research direction.**
- Pair with paper #11: use #11 for existence, #12 for convergence rates.
- The joint strong-monotonicity constant is approximately `γ_joint = γ − N·ε·β`. This generalises `γ_eff = γ − εβ` from paper #2. Report `γ_joint` as a function of `N` in the phase diagram.
- The variational-stability path suggests the system may still converge in the saturation regime, but to a weaker equilibrium concept — an important caveat for systemic-risk interpretations.

---

#### 13. Distributionally Robust Performative Prediction
**Cao & Shi — 2022**
`arXiv:2206.01844` · https://arxiv.org/abs/2206.01844

**What it proves.** Introduces a minimax stable point: the policy that is stable under the *worst-case* distribution shift within an ambiguity set around the nominal sensitivity `ε`. Shows existence and gives a primal-dual algorithm. The robust stable point lies inside the nominal stable region and is more conservative — it widens the margin from the instability boundary.

**Why it matters for `endo_market`.** The project's `ε` is *estimated* from a finite simulation, not known exactly. Reporting a point estimate of `ε*` without uncertainty quantification is misleading for a publication. The distributionally robust formulation provides a formal framework for the uncertainty: instead of "the system is stable at `ε = 0.47`," report "the system is robustly stable for all `ε` in an ambiguity ball of radius `δ` around the estimated value."

**Concrete research direction.**
- Fit an ambiguity set from the variance of `ε` estimates across seeds.
- Compute the robust `ε*` and report both nominal and robust boundaries on the phase diagram.
- The gap between nominal and robust boundaries is a measure of estimation uncertainty — it should shrink with more simulation data, giving an explicit sample-complexity curve.
- This is also the framework for honest sensitivity analysis: "if our toxic-slope estimate is off by 20%, does the stability conclusion change?"

---

#### 14. Computational Optimal Transport
**Cuturi & Peyré — 2019 (textbook)**
`arXiv:1803.00567` · https://arxiv.org/abs/1803.00567

**What it provides.** The technical toolkit for working with Wasserstein distances computationally: entropy-regularised OT, Sinkhorn algorithm, gradient flows, sliced Wasserstein. The `ε`-sensitivity condition in Perdomo et al. is a Wasserstein-1 Lipschitz constant — this book supplies the tools to compute it.

**Why it matters for `endo_market`.** The project currently estimates `ε` via a finite-difference BR-slope — a heuristic proxy for the Wasserstein sensitivity. A principled estimate requires computing `W_1(D(h), D(h'))` directly from simulator samples. Sinkhorn distances are the practical tool for this at the sample sizes the simulator generates.

**Concrete research direction.**
- Replace the BR-slope modulus estimator with a Sinkhorn-distance-based `ε` estimator.
- Report both; compare their values at `ε*`.
- The Sinkhorn estimator has known sample-complexity bounds (from the book's Chapter 7) — this gives formal confidence intervals on `ε*` for the first time.
- The gradient of the Sinkhorn loss with respect to `φ` is the performative gradient — this connects directly to PerfGD (paper #4), where `dD/dφ` is estimated. The Sinkhorn gradient is the correct estimator.

**Reading priority:** Medium-high for methodology; essential if the paper is submitted to a theory-oriented venue.

---

#### 15. High-Frequency Trading in a Limit Order Book
**Avellaneda & Stoikov — Quantitative Finance 2008**

**What it proves.** The foundational Avellaneda-Stoikov model: a dealer maximises expected utility of terminal wealth, with Poisson order arrivals at intensity `λ = A·exp(−k·δ)`. The optimal reservation price is `r = s − q·γ·σ²·T` and the optimal spread is `δ* = γ·σ²·T + (2/k)·log(1 + k/γ)`.

**Why it matters for `endo_market`.** Paper #8 (GLFT) is the multi-asset generalisation of this model. For the single-bond baseline, the AS closed form gives `γ` and `β` analytically from `(k, σ, T)` — three quantities the simulator has access to. Every subsequent extension (papers #8–10) builds on this foundation.

**Concrete research direction.**
- Implement the AS closed-form as the baseline quoting policy.
- Use the AS value function to derive the theoretical `γ` and `β`.
- Report `ε_predicted = εβ/γ` (AS-derived) alongside `m` (empirical).
- The gap between them is a measure of model misspecification — it tells you how much of the stability behaviour is explained by the AS microstructure vs the performative feedback alone.

---

#### 16. Algorithmic and High-Frequency Trading
**Cartea, Jaimungal & Penalva — Cambridge University Press 2015**
`arXiv:1204.4051`

**What it provides.** The most comprehensive stochastic-control treatment of market making as an optimisation problem, including: alpha signals (directional views), adverse selection, inventory constraints, and multi-asset extensions. Chapter 10 specifically addresses adverse selection in the spirit of paper #10 (Barzykin et al.).

**Why it matters for `endo_market`.** The project's quoting policy is currently a learned object (the operator `T_θ`). The Cartea-Jaimungal-Penalva framework provides the *optimal* policy against which to compare. More importantly, their formulation of adverse selection (Chapter 10, Proposition 10.1) gives a closed-form `dτ/dh` — the feedback slope — derived from the trader's information advantage, not from a toxicity knob.

**Concrete research direction.**
- Use Cartea-Jaimungal-Penalva's adverse-selection model as the principled toxic-flow generator inside the simulator.
- The resulting `dτ/dh` is a derived quantity, not a parameter — this directly resolves the "toxicity_feedback is a tuning knob" limitation.
- Pair with paper #10 (Barzykin 2025): both give `dτ/dh`, but via different methods (dynamic programming vs perturbation theory). Comparing them is itself a contribution.

---

#### 17. A General Characterization of the Mean Field Limit for Stochastic Differential Games
**Lacker — Probability Theory and Related Fields 2016**
`arXiv:1510.01408`

**What it proves.** Rigorously establishes the mean-field game (MFG) limit for a large population of strategically interacting agents. As `N → ∞`, the `N`-player Nash equilibrium converges to a mean-field equilibrium where each agent optimises against the *aggregate distribution* of the population.

**Why it matters for `endo_market`.** Papers #11 and #12 extend `endo_market` to `N` dealers. As `N` grows, the discrete multi-dealer game becomes intractable. The MFG limit provides a tractable continuum approximation where each dealer optimises against the aggregate quoting distribution. The performative feedback loop in the MFG limit is: each dealer's quotes → aggregate induced distribution → each dealer's optimal response — a fixed-point problem that generalises Perdomo's single-agent loop.

**Concrete research direction.**
- Derive the MFG version of the performativity stability condition.
- The effective sensitivity in the MFG limit is `ε_MFG = ε · (population density function at the current policy)`.
- Show that the phase diagram `(ε, γ/β)` has a well-defined MFG limit as `N → ∞`.
- This provides the theoretical foundation for scaling the phase diagram from 8 bonds and 1 dealer to a realistic market with 100+ bonds and 10+ dealers simultaneously.

---

#### 18. The Price Impact of Order Book Events
**Cont, Kukanov & Stoikov — Journal of Financial Econometrics 2014**
`arXiv:1011.6402`

**What it proves.** Empirical and structural analysis of how limit order book events (market orders, limit orders, cancellations) generate price impact. Derives a linear price-impact model from order-book mechanics, empirically validated on equity data.

**Why it matters for `endo_market`.** The project's toxic channel is currently: tight quotes → informed flow → inventory loss. The Cont-Kukanov-Stoikov model provides the structural map from quote tightness to information leakage via the order book: tighter quotes reduce the bid-ask spread, increasing the informed trader's profit from picking the dealer off, thereby increasing informed order arrival. This is the microstructural source of the performative feedback that is currently assumed rather than derived.

**Concrete research direction.**
- Use the CKS price-impact model to calibrate the `A·exp(−k·δ)` intensity with an informed-uninformed trader decomposition.
- The informed-trader intensity `λ_informed(δ)` is the structural source of `ε`: `ε ≈ dλ_informed/dδ` at the equilibrium quote.
- This gives a third method for estimating `ε` (alongside BR-slope and Sinkhorn distance), providing a triangulation of the stability boundary from three independent approaches.
- It also connects `endo_market` to the empirical market microstructure literature — making the paper readable and relevant to a financial economics audience, not just a machine learning one.

---

## Synthesis: A Research Roadmap for the Next Version

The original synthesis (five ordered upgrades) remains correct. This section is more opinionated about **what to prove**, **what to measure**, and **how to position the paper**.

### Priority 1 — Make the boundary analytic (not just empirical)

**Target theorem:** For an OTC bond market with GLFT microstructure (#8) and adverse selection (#10, #15, #16), the performativity stability boundary is:

```
ε_analytic < γ(k, σ², q) / β(k, A, δ*)
```

where `γ` and `β` are derived from the AS/GLFT value function (not tuned), and `ε` is derived from the Barzykin perturbation expansion (not swept).

**Required papers:** #1, #2, #5, #8, #10, #14, #15, #16.

**Deliverable:** A table comparing `ε_analytic` vs `ε_empirical` (BR-slope) vs `ε_Sinkhorn` (OT-based) across simulation regimes. Agreement across three methods is strong evidence for the theory.

---

### Priority 2 — Un-blind the operator, prove convergence

**Target experiment:** RRM (blind) vs PerfGD (#4) with `dD/dφ` from the Barzykin perturbation expansion (#10).

**Target theorem:** Under GLFT microstructure, the PerfGD-corrected loop converges to the PO point with rate `O(1/k)`, remaining stable for `ε` up to the true performative-optimal boundary (which is strictly larger than `ε*`).

**Required papers:** #3, #4, #5, #10.

**Deliverable:** A second phase diagram showing the corrected vs uncorrected stability regions — the "echo chamber gap" as a function of `ε`.

---

### Priority 3 — Multi-dealer extension and systemic risk

**Target theorem:** For `N` symmetric dealers, the PSNE stability boundary is `ε < γ/(N·β)`. The market-wide instability (all dealers simultaneously crossing the boundary) emerges at `N·ε·β/γ = 1`.

**Required papers:** #11, #12, #17.

**Deliverable:** A 2D phase diagram `(N, ε)` showing the stability region. This is the paper's systemic-risk result — the finding that competitive quoting among dealers *destabilises* the market, not just a single dealer's loop.

---

### Priority 4 — Robust uncertainty quantification

**Target result:** Nominal `ε*` ± robust ambiguity radius `δ`, with sample-complexity curve showing the radius shrinks as `O(1/√n)`.

**Required papers:** #13, #14.

**Deliverable:** Error bars on the phase diagram. Without this, a reviewer at any theory-oriented venue will correctly object that the stability claim is not statistically valid.

---

### Priority 5 — Scale and calibration

**Target:** Phase diagram at 100+ correlated bonds, `γ` and `β` calibrated from real bond data (duration, DV01, spread volatility).

**Required papers:** #9, #18.

**Deliverable:** A publication-grade phase diagram with tight IQR bands. If bond data is unavailable, the Cont-Kukanov-Stoikov model (#18) provides a synthetic calibration target with known microstructure parameters.

---

## Paper positioning

The ideal submission venue for the completed paper is at the intersection of:

- **ML theory** (ICML, NeurIPS, COLT) — for the performativity angle + Priorities 1–3.
- **Operations Research / Mathematical Finance** (Mathematical Finance, Finance and Stochastics, Operations Research) — for the microstructure grounding + Priorities 4–5.
- **Market microstructure** (Journal of Finance, Review of Financial Studies) — if Priorities 4–5 dominate and the empirical calibration is tight.

The paper's novelty claim is: **the first structural market model where the performativity stability boundary is derived analytically from microstructure primitives, not assumed or tuned.** Papers #1–#7 know the boundary exists; papers #8–#10 know the microstructure; `endo_market` connects them. Papers #11–#18 make the connection multi-agent, robust, and scalable.

---

## What to read first

| If you are... | Start with |
|---------------|-----------|
| New to performativity | #1 → #2 → #3 |
| Debugging the RRM loop | #2 → #5 → #7 |
| Building the PerfGD correction | #4 → #10 → #16 |
| Adding a second dealer | #11 → #12 → #17 |
| Adding uncertainty quantification | #13 → #14 |
| Grounding the microstructure | #15 → #8 → #10 |
| Scaling to 100+ bonds | #9 → #18 |

---

*References: see `references.bib`. PDFs: run `./download_pdfs.sh`.*
