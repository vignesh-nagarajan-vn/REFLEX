# Literature for REFLEX (the `endo_market` lineage) — Performative Prediction in an Endogenous OTC Bond Market

> Prose below refers to the research program by its historical working name
> `endo_market`. The codebase generations are `endo_market_v1` (legacy) →
> `endo_market_v2` → `endo_market_v3` (current, package `reflex`).

This folder collects ten papers that together supply the mathematical scaffolding behind
the program and, more importantly, a concrete menu of formal tools for **extending its
scale, scope, and technical depth**. The project's headline object — a best-response
contraction modulus `m` that crosses 1 at a critical feedback gain `ε* ` , reproducing the
performative-prediction stability boundary `ε < γ/β` inside a structural market model — sits
at the intersection of two literatures that rarely talk to each other:

- **Performative prediction / decision-dependent stochastic optimization** — the theory of
  when a policy↔distribution retraining loop contracts or diverges.
- **Optimal market making (stochastic optimal control)** — the theory of how a dealer's
  quotes shape order-arrival intensities, inventory risk, and adverse selection.

The reading map below is organized so each paper points back at a specific component of the
codebase (the RRM loop, the operator `T_θ`, the BR-slope modulus estimator, the toxic-flow
gate, the inventory state, the scale-up caveats) and forward at a specific extension.

> **PDFs.** Run `./download_pdfs.sh` to populate `pdfs/`. Every entry is an open-access
> arXiv preprint, so the script needs only outbound access to `arxiv.org`. Full citations
> are in `references.bib`; direct links are in each section heading below.

---

## Reading map (how the ten map onto the project)

| # | Paper | Project component it speaks to |
|---|-------|--------------------------------|
| 1 | Perdomo et al. — *Performative Prediction* | The `ε < γ/β` boundary; RRM loop; stable vs optimal |
| 2 | Mendler-Dünner et al. — *Stochastic Optimization for PP* | The `γ − εβ` effective convexity; greedy/lazy deploy |
| 3 | Miller et al. — *Outside the Echo Chamber* | The stable≠optimal gap; defensive widening; PR objective |
| 4 | Izzo et al. — *Performative Gradient Descent* | Fixing the operator that is "blind to `dD/dφ`" |
| 5 | Drusvyatskiy & Xiao — *Decision-Dependent Distributions* | Rigorous convergence theory; vanishing-bias view |
| 6 | Jagadeesan et al. — *Regret Minimization w/ Performative Feedback* | Making `ε` an explorable quantity, not a swept knob |
| 7 | Li & Wai — *State-Dependent Performative Prediction* | Inventory carried across steps (`q_after`) |
| 8 | Guéant, Lehalle, Fernández-Tapia — *Inventory Risk* | The `exp(−decay·h)` intensity; deriving `γ` from microstructure |
| 9 | Bergault & Guéant — *Size Matters for OTC MMs* | The "scale to 100+ bonds" caveat; factor reduction |
| 10 | Barzykin, Bergault, Guéant, Lemmel — *Adverse Selection & Price Reading* | The toxic-flow channel, derived rather than gated |

Themes: **#1–#6** are the performative core, **#7** bridges to stateful dynamics, **#8–#10**
are the market-microstructure control theory. Novelty-of-formulation highlights: #5 (math.OC
bias-corruption framing), #6 (feedback-as-exploration), #7 (controlled-Markov-chain SA), and
#10 (a 2025 perturbation theory of adverse selection + price reading).

---

## 1. Performative Prediction
**Perdomo, Zrnic, Mendler-Dünner, Hardt — ICML 2020.** arXiv:2002.06673 ·
<https://arxiv.org/abs/2002.06673>

**Core findings.** This is the paper `endo_market` instantiates. It formalizes a setting where
deploying a model `φ` shifts the data distribution to `D(φ)` (decision-dependent data), and
introduces two solution concepts: a **performatively stable** point (a fixed point of repeated
retraining — optimal on the distribution it itself induces) and a **performatively optimal**
point (minimizer of the true performative risk). The central theorem: if the loss is
`β`-smooth and `γ`-strongly convex and the distribution map `D(·)` is `ε`-sensitive (Lipschitz
in the deployed parameter under the Wasserstein-1 metric), then **repeated risk minimization
(RRM) is a contraction with modulus bounded by `εβ/γ`**, converging linearly to the unique
stable point precisely when `εβ/γ < 1`. Equivalently, the loop is stable iff `ε < γ/β`. They
also show that if any of smoothness, strong convexity, or `ε`-sensitivity fails, retraining can
oscillate or diverge.

**Connection to the project.** The project's "contraction modulus `m` crosses 1" and its stated
boundary `ε < γ/β` *are* this theorem, realized structurally: `m ≈ εβ/γ`; the RRM loop is
literal repeated retraining; "the dealer re-tightens without anticipating the extra toxic flow"
is the stable-vs-optimal gap; and the toxicity-feedback knob is the `ε`-sensitivity of `D(φ)`.

**How it extends the work.** It converts the project's *measured* `ε*` into a *predicted* one:
estimate `γ` from the quoting-anchor curvature, `β` from the operator's smoothness, and `ε`
from the toxic slope, then compare `εβ/γ` against the empirical BR-slope `m`. The
"assumption-failure ⇒ divergence" clause is exactly the lens for the project's honest caveats:
when the best response defensively widens into a low-curvature region, `γ → 0`, the boundary
moves, and the modulus saturates instead of blowing up.

---

## 2. Stochastic Optimization for Performative Prediction
**Mendler-Dünner, Perdomo, Zrnic, Hardt — NeurIPS 2020.** arXiv:2006.06887 ·
<https://arxiv.org/abs/2006.06887>

**Core findings.** Moves from full retraining to **stochastic gradient** updates, separating
the act of *updating* parameters from *deploying* them. It analyzes **greedy deploy** (publish
after every stochastic step) and **lazy deploy** (several updates before redeploying), and
shows the convergence rate smoothly recovers the static `O(1/k)` optimum as performativity
vanishes. The key quantity: in the analysis the static strong-convexity constant `γ` is
**replaced by the effective constant `γ − εβ`**, which governs the contraction.

**Connection to the project.** `γ − εβ > 0 ⟺ ε < γ/β` — this is the same boundary, written
in the form that exposes *why* it is the boundary. The project's "refit from scratch each
iteration on one deployment's data" is greedy-deploy RRM; the BR-slope modulus is measuring the
contraction that `γ − εβ` controls.

**How it extends the work.** Near the critical point the project's finite-difference modulus is
noisy (its honest caveat). A `γ − εβ` **plug-in estimator** — built from operator/anchor
curvature — gives a lower-variance stability diagnostic to report alongside the BR-slope. Lazy
deployment (deploy every `K` retrains) becomes a *stabilization knob* the project could add and
sweep: a controlled way to trade convergence speed against robustness to the feedback gain.

---

## 3. Outside the Echo Chamber: Optimizing the Performative Risk
**Miller, Perdomo, Zrnic — ICML 2021.** arXiv:2102.08570 ·
<https://arxiv.org/abs/2102.08570>

**Core findings.** Studies the **performative risk** `PR(φ) = E_{z∼D(φ)}[loss]` directly,
rather than the stable point that RRM finds. Stable ≠ optimal in general (the "echo chamber":
a learner that only ever optimizes against the distribution it induced can be far from the
performative optimum). The paper gives structural conditions — notably a **location-scale**
assumption on how `φ` moves the distribution — under which `PR` is convex and the performative
optimum is efficiently findable.

**Connection to the project.** The project's mechanism *is* the echo chamber: the operator
conditions on a `policy_summary` frozen at the deployed regime, so the dealer optimizes inside
its own induced distribution and re-tightens without seeing the flow it summons. The
"defensive widening into a region where spread-capture curvature vanishes" is the pathology
this paper formalizes.

**How it extends the work.** The toxic-flow term is additive and spread-responsive
(`α · toxicity_feedback · info_intensity · exp(−decay·h)`), which is close to a location-scale
shift — so `PR` may be convex under a *checkable* condition. That opens a path beyond RRM:
build a performative-**optimum** solver for the dealer (not just a stability locator), and
report the optimal-vs-stable spread gap as a second axis of the phase diagram.

---

## 4. How to Learn when Data Reacts to Your Model: Performative Gradient Descent
**Izzo, Ying, Zou — ICML 2021.** arXiv:2102.07698 · <https://arxiv.org/abs/2102.07698>

**Core findings.** Introduces **PerfGD**, which estimates the *performative* gradient
`d/dφ E_{z∼D(φ)}[loss]`. That gradient splits into the usual static term **plus** a
distribution-response term that captures `dD/dφ`. By estimating `dD/dφ` from a parametric model
of the induced distribution across deployments, PerfGD descends the true performative risk and
converges toward the **optimum**, not merely the stable point — closing the echo-chamber gap.

**Connection to the project.** This addresses the project's single most important stated
limitation head-on: the learned operator `T_θ` "cannot learn `dD/dφ`" because the
`policy_summary` is constant within a deployment. PerfGD is the recipe to *un-freeze* it.

**How it extends the work.** Concretely: fit the `policy_summary → toxic-slope` map and inject
the resulting `dτ/dh` term into the policy gradient (the project already has pathwise /
REINFORCE / RGD paths to carry it). This turns the cobweb into a *corrected* descent and is a
clean, well-scoped next experiment: compare RRM (blind) vs PerfGD-corrected (anticipating) and
show the corrected loop staying stable past `ε*`.

---

## 5. Stochastic Optimization with Decision-Dependent Distributions
**Drusvyatskiy & Xiao — Mathematics of Operations Research 2023 (arXiv 2020).**
arXiv:2011.11173 · <https://arxiv.org/abs/2011.11173>

**Core findings.** The rigorous `math.OC` backbone (≈60 pages). It shows that a broad family of
standard stochastic algorithms — SGD, proximal, clipped, and others — applied *directly* to a
decision-dependent problem behave like the *same* algorithm on a fixed static problem corrupted
by a **vanishing bias**. Under a Lipschitz distribution map plus strong-monotonicity/convexity,
this yields convergence to the equilibrium with explicit rates, generalizing the Perdomo
contraction well beyond exact retraining.

**Connection to the project.** This is the formal language for "why repeated retraining
converges (or not)" and for the project's frozen-summary bias. The BR-slope contraction is a
special case; the "vanishing bias" precisely names the cobweb gap between what the operator
sees and the true performative gradient.

**How it extends the work.** It lets the project upgrade an *empirical* stability claim into a
*theorem with constants* for the specific algorithms it ships (pathwise / REINFORCE / RGD).
The bias-corruption framing also quantifies how far the learned operator's equilibrium sits
from the true performative equilibrium — turning a caveat into a bounded error term.

---

## 6. Regret Minimization with Performative Feedback
**Jagadeesan, Zrnic, Mendler-Dünner — ICML 2022.** arXiv:2202.00628 ·
<https://arxiv.org/abs/2202.00628>

**Core findings.** Observes that performativity gives a feedback structure **richer than a
bandit**: after deploying `φ` you observe *samples from* `D(φ)`, not just a scalar reward. The
paper exploits smoothness of the shift map to build confidence bounds on the risk of
*unexplored* models and derives regret that scales with the complexity of the **distribution
shifts**, not the reward function — and it does not require convexity.

**Connection to the project.** It reframes the project's feedback gain `ε` as something to be
**actively explored and learned**, rather than swept by hand over a fixed grid. The operator
`T_θ` is exactly the "samples from `D(φ)`" channel this paper relies on.

**How it extends the work.** Replace the 3-seed manual `ε` sweep with an **adaptive sequential
design**: an exploration policy over quote regimes that locates `ε*` online with regret
guarantees. Crucially, because the method does not assume convexity, it remains valid in the
project's **saturation regime** — precisely where Perdomo-style convexity arguments break down.

---

## 7. State-Dependent Performative Prediction with Stochastic Approximation
**Li & Wai — AISTATS 2022.** arXiv:2110.00800 · <https://arxiv.org/abs/2110.00800>

**Core findings.** Extends performative prediction to a **stateful** setting: the induced
distribution depends on both the current decision *and* previous states (memory / unforgetful
agents). The retraining loop is modeled as a **state-dependent stochastic approximation**
algorithm with biased gradients driven by a *controlled Markov chain* whose transition kernel
depends on the learner's state. They prove the iterates reach the performatively stable point
with expected squared error decaying as `O(1/k)`.

**Connection to the project.** The dealer is *not* memoryless: inventory `q_after = q0 + B − S`
carries across steps, the toxic channel gates on `edge` and state, and P&L includes
`inv_risk_weight·q²`. That makes the project a state-dependent performative problem — which the
current within-deployment RRM under-models.

**How it extends the work.** This is the correct convergence theory once inventory carryover is
folded into the loop. The controlled-Markov-chain formulation matches the project's simulator,
and it points toward a stability result that accounts for **inventory dynamics**, not just the
static quote→toxicity map — a genuine scope expansion of the headline theorem.

---

## 8. Dealing with the Inventory Risk: A Solution to the Market Making Problem
**Guéant, Lehalle, Fernández-Tapia — Mathematics and Financial Economics 2013 (arXiv 2011).**
arXiv:1105.3115 · <https://arxiv.org/abs/1105.3115>

**Core findings.** The canonical tractable market-making control problem (the
Avellaneda–Stoikov lineage). Order-arrival intensities decay exponentially in the quoted
distance, `λ(δ) = A·exp(−k·δ)`; the dealer maximizes expected utility of terminal P&L under
inventory risk. Via an exponential-utility change of variables the **HJB equation collapses to
a system of linear ODEs**, giving closed-form / asymptotic optimal bid–ask quotes that trade
spread capture against inventory variance.

**Connection to the project.** The project's `exp(−info_spread_decay·h)` toxic response, its
inventory penalty, and its quoting anchor are all in this lineage. The strong-convexity `γ`
that "pins the best response (no runaway)" has a concrete microstructural origin here.

**How it extends the work.** Swap the project's *heuristic* quoting-cost anchor for the
AS/GLFT closed-form value function, so `γ` is **derived rather than tuned** — which in turn
makes the predicted boundary `ε < γ/β` quantitative. It also supplies an analytic baseline BR
map to validate the learned operator against, and it explains the `α`-confound directly: the
exponential intensity decay `k` is the structural source of the curvature that vanishes at wide
spreads.

---

## 9. Size Matters for OTC Market Makers: General Results and Dimensionality Reduction
**Bergault & Guéant — Mathematical Finance 2021 (arXiv 2019).** arXiv:1907.01225 ·
<https://arxiv.org/abs/1907.01225>

**Core findings.** Tackles **multi-asset OTC** dealing, where one market maker quotes a long
list of correlated assets. The multi-asset HJB suffers the curse of dimensionality; the paper's
contribution is a **factor-model dimensionality-reduction** technique plus closed-form
approximations that keep the optimal-quote computation tractable for large universes.

**Connection to the project.** This is, almost verbatim, the project's own scale-up caveat:
"for a publication-grade phase diagram, scale up: larger bond universe (100+)." The current
8-bond market exists because of single-CPU compute, not because the science wants it small.

**How it extends the work.** The factor reduction is the concrete method to push the phase
diagram to 100+ correlated bonds without an exponential blow-up — letting the modulus and `ε*`
be estimated on a realistic universe with tight median+IQR bands, exactly the
"publication-grade" target the project names.

---

## 10. Optimal Quoting under Adverse Selection and Price Reading
**Barzykin, Bergault, Guéant, Lemmel — arXiv 2025.** arXiv:2508.20225 ·
<https://arxiv.org/abs/2508.20225>

**Core findings.** A 2025 stationary, infinite-horizon market-making control model that adds
two informational risks: **adverse selection** (informed traders systematically pick the dealer
off) and **price reading** (the dealer's own quotes leak the direction of its inventory and get
exploited). The authors use Taylor/perturbation expansions to capture the **first-order impact
of informational risk** on the optimal quotes, in a tractable infinite-horizon formulation.

**Connection to the project.** This is the rigorous control-theory version of the project's
toxic-flow channel. "Tighter quotes summon more informed flow that picks the dealer off" *is*
adverse selection plus price reading; the project's `gate(edge)` and `α · toxicity_feedback`
are heuristics for the informed-intensity this paper derives from first principles.

**How it extends the work.** Replace the project's hand-built toxic-flow gate with the
principled informed-intensity from this model. Better: its perturbation expansion yields an
**analytic `dτ/dh`** — a closed-form feedback slope — that can be compared head-to-head against
the swept `toxicity_feedback` knob, giving the project a theory-grounded `ε` instead of a
tuning parameter. This is the single most direct route to making the toxic channel principled.

---

## Synthesis: a mathematical roadmap for the next version

Reading the ten together suggests five concrete, ordered upgrades:

1. **Make the boundary predictive, not just measured.** Use Perdomo (#1) + Mendler-Dünner (#2)
   + Drusvyatskiy–Xiao (#5) to estimate `εβ/γ` (equivalently `γ − εβ`) from curvature and
   compare it against the BR-slope `m`. Report both. This also *explains* the saturation and
   `α`-non-monotonicity as assumption failure (`γ → 0` under defensive widening), upgrading two
   caveats into predictions.

2. **Un-blind the operator.** Use Izzo PerfGD (#4) and Miller (#3) to inject the `dD/dφ` term
   into the policy gradient and optimize the performative risk `PR(φ)` rather than only locating
   the stable point. Headline experiment: blind RRM vs performativity-aware descent, showing the
   latter stays stable past `ε*`.

3. **Make `ε` first-class.** Use Jagadeesan (#6) to replace the manual seed-grid sweep with an
   adaptive exploration of the feedback gain that finds `ε*` online with regret guarantees —
   and that remains valid in the non-convex saturation regime.

4. **Add inventory state to the loop.** Use Li & Wai (#7) to give the carried inventory its
   proper place, recasting the loop as a state-dependent stochastic approximation with a
   controlled Markov chain — and obtain a convergence result that respects inventory dynamics.

5. **Ground the market model and scale it.** Use Guéant–Lehalle–Fernández-Tapia (#8) and
   Barzykin et al. 2025 (#10) to *derive* `γ` and the toxic slope `dτ/dh` from microstructure
   instead of tuning them, then use Bergault–Guéant (#9)'s factor reduction to take the phase
   diagram to 100+ correlated bonds.

Taken end to end, this turns `endo_market` from a careful empirical demonstration of one
stability crossing into a model whose boundary, feedback gain, and market primitives are all
analytically grounded — and whose phase diagram is computed at publication scale.

---

### Files in this folder
- `README.md` — this document.
- `references.bib` — BibTeX for all ten papers (with published venues where available).
- `download_pdfs.sh` — fetches all ten open-access PDFs into `pdfs/`.
- `pdfs/` — destination for the downloaded PDFs (empty until you run the script).
