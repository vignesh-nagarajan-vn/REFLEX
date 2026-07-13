# CLAUDE.md

Orientation for AI coding agents working in **REFLEX**. Read this before making
changes; it captures context that is not obvious from the file tree alone.

## What this project is

**REFLEX** — *Reflexive Equilibrium Fixed-point Learning for endogenous
financial markets.* It is a research codebase, not a production system. The
scientific question: in an OTC corporate-bond market, a dealer's quoting policy
`φ` reshapes the order-flow distribution `D(φ)` (tighter quotes summon more
informed/"toxic" flow). Under **repeated retraining (RRM)**, when does the
policy↔distribution loop **converge** to a stable equilibrium versus
**diverge**?

This is an instance of **performative prediction** (Perdomo et al., ICML 2020)
realized inside a structural market-making model. The headline object is the
best-response contraction modulus `m ≈ εβ/γ`; the loop is stable iff `ε < γ/β`
(equivalently `m < 1`). See the root [README.md](README.md) for the full framing.

## Repository layout (what's authoritative vs. archived)

| Path | Status | Notes |
|------|--------|-------|
| `endo_market_v4/` | **CURRENT — work here** | The final, complete generation: package **`reflex`** v4 (un-blinded ML + theory 1.1–1.6 + real-data calibration + the structural PerfGD loop + the verification layer + all experiments). |
| `research/` | Active — the program folder | The research roadmap, the canonical math derivations (`math-theory/`, 1.1–1.6, LaTeX twins), the **canonical data pipeline** (`data_collection/`, `preprocessing/` — v4 ships copies of its outputs), the **executed paper-grade runs** (`results/`: v3's `07-10-2026`, v4's `07-12-2026`), their written **analyses** (`analysis/`), and the **ICAIF 2026 submission draft** (`paper/`: ACM sigconf `main.tex` + verified `references.bib` + figures; every number traced to the 07-12-2026 run). An extension/application of `endo_market_v4`, not a second implementation. |
| `literature/literature-vignesh/` | Active reference | 10 foundational papers + reading map; PDFs downloaded. |
| `literature/literature-raghav/` | Active reference | Superset: same 10 + 8 extension papers (18 total) + research roadmap. |
| `archive/` | **Frozen** | The four superseded generations: `edl_simulator_v1` (HTML/JS prototype), `endo_market_v1` (first Python build), `endo_market_v2` (identified `ε`, grew the analytic modules), `endo_market_v3` (the audited generation that produced the 07-10-2026 run). Don't extend anything here; see `archive/README.md`. |

When asked to change "the code" or "the experiment," default to
`endo_market_v4/` unless the user names an archived folder explicitly. v4 is
deliberately **self-contained**: derivation copies in `theory/`, Lean skeletons
in `lean/`, data copies in `data/`, experiments and tests inside the folder.

## `endo_market_v4` internals (package `reflex`)

Package root: `endo_market_v4/reflex/`. The strands and where they live:

- **Environment (`env/`)** — `simulator.py` (single-dealer `T_true`),
  `clients.py` (toxic/informed flow; the `toxicity_feedback` gain `ε`),
  `bonds.py`, `liquidity_field.py`, and **`multi_dealer.py`** (genuine
  `N`-dealer market sharing one informed pool with spillover `κ`; reduces
  bit-for-bit to the single-dealer market at `N = 1`).
- **Policies (`policy/`)** — `dealer_policy.py` (linear/MLP) and
  `glft_baseline.py` (non-learned closed-form baseline quoting the analytic
  `h_SP` or `h_PO`).
- **Operator (`operator/`)** — `response_operator.py`: `T_θ` with the
  un-blinding: `distribution_response()` / `toxic_slope()` read the *learned*
  `dD/dφ` out by autograd w.r.t. the policy summary. Whether it can learn that
  depends on **windowed fitting** (`operator.context_window` deployments per
  fit); frozen-summary optimisation keeps the blind v2 baseline available.
- **Theory (`theory/`)** — the six closed-form modules, numpy-only:
  `analytic_boundary` (1.1), `perfgd` (1.2), `multi_dealer` (1.3), `robust`
  (1.4, incl. the v4 `calibrate_radius`), `factor_scaling` (1.5),
  `lazy_deploy` (1.6: the K-step map `mu(K) = -m + c^K(1+m)` and the
  two-branch `gamma_eff`). Derivation documents ship in
  `endo_market_v4/theory/` with a code map.
- **Loops (`equilibrium/`)** — `loops.py` `run_loop(mode=...)`, four modes:
  `rrm` (blind baseline) | `perfgd_analytic` (closed-form correction
  `Δ = −β(h−ψ)ε(h)` as a surrogate gradient) | `perfgd_learned` (free-form
  live summary — kept as the documented negative result) |
  **`perfgd_structural`** (v4: every 1.2 ingredient *fitted* from the loop's
  own deployment history by `structural_response.py` — the GLFT-anchored
  families `τ̂ = C0 + C1·e^{−ch}`, `û = A_u·e^{−k_u h}`, realized `ψ̂` — with
  a trust region and an anti-echo freeze; the mode that stabilises beyond the
  boundary). Every iteration logs the learned, structural and analytic toxic
  slopes (the three-way ML↔math seam). `joint_loop.py` runs the simulated
  `N`-dealer cobweb + CRN joint-modulus probes. `rrm_loop.py` is the frozen
  v2-compatible baseline.
- **Estimators (`estimators/`)** — the three-way `ε` triangulation:
  `br_slope` (CRN best-response probe + the signed K-step probe
  `measure_rgd_response` for 1.6), `sinkhorn` (exact 1-D quantile W1 +
  debiased log-domain Sinkhorn with the tuned scale-relative blur
  `TUNED_REL_REG = 0.02`, `reg="auto"`), `cks` (fitted informed-flow-curve
  slope), `triangulate` (all three vs the closed form).
- **Calibration (`calibration/`)** — `loader.py` (shipped data CSVs),
  `mapping.py` (`(rating, regime) → Config`; the package's single
  unit-conversion point), `regimes.py` (VIX regimes).
- **Analysis (`analysis/`)** — `fragility.py` (the daily 1990–2026
  market-fragility index from real data), `phase.py` (analytic prediction
  curves, `(N, ε)` surface), plus `convergence`, `metrics`, `sweep`, `plots`.
- **Verification (`verification/`)** — `certificates.py`: 66 numerical proof
  checks re-deriving every load-bearing identity/inequality/dynamical claim of
  1.1–1.6 against the implementations (finite-difference slopes, eigensolves,
  Monte-Carlo rates), run on raw *and* calibrated configs. The Lean 4 formal
  skeletons live in `endo_market_v4/lean/` (reviewed statements; **not yet
  compiled** — no Lean toolchain on the dev machine; the certificates are the
  verification of record).

Experiments (`experiments/`): `run_single`, `run_sweep` (predict-then-verify +
robust bands), `run_perfgd` (`--ml` for the four-mode loops), `run_dealers`,
`run_universe`, `run_triangulation`, `run_fragility`, `run_calibrated`,
`run_lazy_deploy` (1.6), `run_tuning` (Sinkhorn blur + robust radius),
`run_certificates` (the proof checks), and `run_all --profile smoke|full`.
Configs in `configs/`; artifacts in `outputs/`.

## Conventions & gotchas

- **CPU-only, reproducible, no GPU.** Keep it that way. Every run must be
  deterministic from `(config, seed)`; route randomness through `utils/seeding.py`.
- **`ε` (`clients.toxicity_feedback`) is the clean control variable, not `α`
  (adversariality).** Sweeping `α` is confounded: high `α` drives the dealer to
  wide spreads where the toxic response `exp(−decay·h)` has decayed to ~0, so the
  modulus flattens or reverses (its sign isn't robust to universe size). Don't
  "fix" this — it's a documented feature. Use `sweep_feedback.yaml` for the
  headline result.
- **The modulus saturates past the boundary** rather than blowing up —
  defensive widening into a low-curvature region (derived in closed form:
  theory 1.1 §6.3; at default-like constants the self-consistent fixed point
  never crosses `m = 1` at all). Beyond the boundary the *measured* probe
  readings scatter widely (seed-level bifurcation) — they are
  finite-difference diagnostics there, not local slopes.
- **The structural loop optimises the *realized* market, not the A2 closed
  form.** In high-intensity regimes (e.g. the beyond-boundary demo) the
  realized response includes the `info_cap` saturation, the
  liquidity-inflation feedback and severity drift, which the frozen-reference
  closed forms deliberately omit (1.1 §9). Benchmark `perfgd_structural`
  against *independent structural fits*, never against the A2 `h_PO`; and do
  NOT de-saturate `info_cap` to force agreement — without it the
  `liq_flow_boost` feedback blows realized flows to 7–20× the closed forms.
- **Exponential response fits need a wide spread range.** Fitted on a narrow
  window they are line-degenerate and extrapolate arbitrarily badly (measured
  failure: the local-window loop marched to `h_SP` instead of `h_PO`). Keep
  `rrm.structural_window` long (the loop's own transient is the exploration),
  respect the trust region `rrm.structural_max_rel_step`, and never steer on
  an unidentified fit (the anti-echo freeze handles it).
- **The 1-D frozen-gradient helpers (`best_response`, `Phi'`) omit the
  inventory-risk curvature**: their measured slopes are governed by
  `γ − P·λ_q`, while the full-pipeline modulus uses the `λ_q`-inclusive `γ`
  (1.1 §7). Compare like with like (`lambda_q=0` for the 1-D map) — a
  certificate-discovered convention, encoded in
  `verification/certificates.py`.
- **Calibrated configs use real units** (per-$100-par, h ~ 0.4–3.5 per 100 par;
  one step ~ one trading day). Never hard-code absolute spread constants —
  probe widths, tolerances, caps AND the Sinkhorn blur must be *relative* to
  the configured scale. `reflex/calibration/mapping.py` is the only
  unit-conversion point. (This is also why the beyond-boundary demo dynamics
  certificate auto-excludes itself on calibrated configs: its demo constants
  are absolute raw-unit values.)
- **Only `(A, k, σ, h)` are data-identified**; the toxic channel is
  structurally scaled (documented ratios in `mapping.py`). The crisis-regime
  intensity fit is degenerate (`k = 0`) — crisis boundaries sit on the anchor
  floor and are flagged. State all of this plainly in any write-up.
- **`summary_mode` is the blind/un-blinded switch.** Frozen summary = blind RRM
  (v2 convention); live summary + `operator.context_window ≥ 2` = the learned
  `dD/dφ` enters the gradient. `perfgd_learned` with window 1 is noise (the
  loop warns); `perfgd_structural` with `collection_jitter = 0` loses its
  within-deployment identification (the loop warns).
- **Multi-dealer runs can saturate `info_cap`**: the combined gross flow of `N`
  dealers inflates the shared liquidity field. Scale `liq_flow_boost` down or
  `info_cap` up for flow-allocation studies (see `env/multi_dealer.py`).
- **Smoke vs full profiles:** `run_all --profile smoke` proves the pipeline
  (tiny settings, ~minutes); scientific claims need `--profile full`
  (~25 min measured on the v4 configs, 11 experiments).
- **Console prints must stay ASCII** — the Windows console (cp1252) crashes on
  `λ`, `ε`, `m̂` etc. in `print()`. Unicode in matplotlib labels/docstrings is
  fine.
- **Math notation:** code/READMEs sometimes use ASCII (`phi`, `epsilon`, `gamma`,
  `dtau/dh`) and sometimes Unicode (`φ`, `ε`, `γ`). Match the surrounding file.
- This is a **Windows** environment with **PowerShell** as the primary shell; a
  Bash tool is also available. The repo path itself contains a space
  (`GitHub Projects`) — quote paths. Python 3.9 is the system interpreter
  (venv at `.venv/`); the code targets ≥ 3.9.

## Build / test / run (`endo_market_v4/`)

```
../.venv/Scripts/python -m pip install -e .        # or: pip install -e .  (in the venv)
../.venv/Scripts/python -m pytest -q -m "not slow" # 142 fast tests
../.venv/Scripts/python -m pytest -q -m slow       # +10 slow end-to-end tests

../.venv/Scripts/python -m experiments.run_all --profile smoke   # everything, ~minutes
../.venv/Scripts/python -m experiments.run_all --profile full    # paper-grade, ~15-25 min
../.venv/Scripts/python -m experiments.run_certificates          # 66 proof checks, seconds
../.venv/Scripts/python -m experiments.run_fragility             # real-data index, seconds
```

152 tests total across 20 files (the 15 inherited from v3 — simulator, policy,
operator, rrm-convergence, the five analytic modules, glft-baseline,
calibration, fragility, estimators, unblinded-operator, perfgd-ml,
multi-dealer-env — plus `test_lazy_deploy`, `test_tuning`,
`test_structural_perfgd`, `test_certificates`). Run from inside
`endo_market_v4/`, using the repo venv (`.venv/` at the repo root — system
Python has no torch). Always run the fast tests after changing the library.
See `endo_market_v4/README.md` for methodology, layout, and honest caveats.

## Working with the literature

The two `literature/*/README.md` files are the maps from each paper to a specific
code component and a concrete extension. When implementing a literature-driven
feature (e.g., a PerfGD correction, a `γ−εβ` plug-in estimator, a multi-dealer
extension), consult `literature-raghav/README.md` first — it names the target
theorem, required papers, and deliverable for each roadmap priority. PDFs are
fetched per-collection via `download_pdfs.sh` (open-access arXiv preprints).

## The research program & dataset (`research/`)

The novelty claim: derive the performativity stability boundary analytically
from microstructure primitives instead of sweeping it by hand. Structure:

- `math-theory/` — the six canonical derivations (1.1 analytic boundary
  `m = εβ/γ`, 1.2 PerfGD un-blinding, 1.3 multi-dealer `ε < γ/(N_eff·β)`,
  1.4 robust boundary `O(1/√n)`, 1.5 factor scaling, 1.6 lazy deployment),
  each `.md` with a compilable `.tex` twin (PDFs under `latex-papers/`; the
  1.6 PDF is pending). **Authoritative implementations live in
  `endo_market_v4/reflex/theory/`** (copies of the documents ship in
  `endo_market_v4/theory/`); the older `endo_market_v2` modules are the
  frozen originals (under `archive/`).
- `data_collection/` + `preprocessing/` — the **canonical real-data pipeline**:
  ~36 yrs daily / 70 yrs monthly of *public, verified* macro + bond-factor
  series (CBOE VIX, EIA WTI, Fed H.15 10y, Shiller, gold/CPI,
  Dickerson–Mueller–Robotti TRACE bond factors, 212 real-CUSIP bond returns)
  joined into `REFLEX_MASTER_DATASET.csv`, cleaned/enriched, with fitted
  intensity params per (rating × regime). v4 ships copies of the artifacts it
  consumes (`endo_market_v4/data/`). **Provenance gotcha (state this plainly
  in the paper):** it is *not* trade-level TRACE — `h`, per-dealer `q`, and
  real per-bond `A`/`k` need WRDS TRACE Enhanced (pending); the pipeline uses
  the closest free proxies. See `data_collection/docs/REJECTED_SOURCES.md`.
- `results/` — the executed paper-grade runs (one dated folder per
  `run_all --profile full` execution: raw per-experiment artifacts + logs,
  each recording the producing commit). `07-10-2026` is the v3 run (8/8);
  `07-12-2026` is the v4 run (11/11, includes the new experiments). Raw
  copies only; no post-processing.
- `analysis/` — the written analyses of those runs (derived tables, figures,
  predicted-vs-measured breakdowns, honest caveats) plus the pre-run
  measurement-layer audit. The live code/experiments themselves are inside
  `endo_market_v4/` — this folder is the *application* of that package, not a
  second implementation.
- `paper/` — the ICAIF 2026 submission draft: `main.tex` (ACM `sigconf`,
  `anonymous,review` for double-blind), `references.bib` (arXiv IDs verified;
  two wrong IDs inherited from `literature/*/references.bib` corrected —
  don't copy entries from those bibs without checking), `figures/` (copies
  from `results/07-12-2026/`). No local TeX toolchain — compile on Overleaf;
  build/trim/camera-ready notes in `paper/README.md`. Keep every claim no
  stronger than its counterpart in `results/07-12-2026/REPORT.md`.

## Current phase & next steps

**The repo side of the program is complete (v4, July 2026).** Theory 1.1–1.6
derived and coded; the ML un-blinded; the v3 loop-level gap **closed** by the
structural mode (`perfgd_structural` settles at the realized performative
optimum beyond the boundary while blind RRM diverges — verified against
independent structural fits); the estimators tuned (Sinkhorn blur, robust
radius); the verification layer in place (66 numerical certificates passing on
raw + calibrated configs; Lean 4 skeletons reviewed, compile pending a
toolchain); the measurement layer audited (July 2026, six probe/protocol
defects fixed — see `research/analysis/pre-run-audit-2026-07.md`); paper-grade
full-profile suites executed and curated (`research/results/07-10-2026/` for
v3, `research/results/07-12-2026/` for v4). The **live to-do**:

1. **Finish and submit the ICAIF 2026 paper** (deadline Aug 2 2026). The
   submission draft is written: `research/paper/` (ACM `sigconf`, 8 pages,
   double-blind; scoped as settled in the analysis — closed forms +
   real-data fragility + probe-level verifications are the headline; the v4
   structural-loop stabilisation is reported against the *realized-market*
   benchmark with the A2-gap channels named). The working copy is
   **de-anonymized** (real author block — co-first authors, Vignesh
   corresponding — and the public GitHub footnote). Compiled on Overleaf
   2026-07-12 at **exactly 8 pages** (`research/paper/REFLEX_Research_Paper.pdf`;
   zero slack — re-check the count after any edit, trim order in
   `paper/README.md`). Remaining: before CMT submission flip the
   double-blind toggle documented in the `main.tex` header (class option +
   anonymized-mirror footnote); submit.
   Checklist in `research/README.md` (§ To-Do → ICAIF-specific requirements).
2. Compile the Lean skeletons once a toolchain is available (`lean/README.md`)
   and build the 1.6 PDF via Overleaf.

Gotcha for future measurement work: probe protocols matter at first order —
keep `rrm.collection_jitter = 0.05` (0.2 inflates the CRN BR-slope probe
~3x), probe at the operating spread (not the analytic fixed point), and
compare against the *realized-state* closed form (1.1 §9).
