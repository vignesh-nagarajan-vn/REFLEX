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

## Repository layout (what's authoritative vs. legacy)

| Path | Status | Notes |
|------|--------|-------|
| `archive/endo_market_v3/` | **CURRENT — work here** | The self-contained third generation: package **`reflex`** (un-blinded ML + theory 1.1–1.5 + real-data calibration + all experiments). |
| `research/` | Active — the program folder | The research roadmap, the canonical math derivations (`math-theory/`, with LaTeX/PDF), the **canonical data pipeline** (`data_collection/`, `preprocessing/` — v3 ships copies of its outputs), the **executed paper-grade runs** (`results/`), and their written **analyses** (`analysis/`). An extension/application of `endo_market_v3`, not a second implementation. |
| `literature/literature-vignesh/` | Active reference | 10 foundational papers + reading map; PDFs downloaded. |
| `literature/literature-raghav/` | Active reference | Superset: same 10 + 8 extension papers (18 total) + research roadmap. |
| `archive/endo_market_v2/` | **Superseded** | Second generation; its result (the ε* crossing) and modules were absorbed into v3. Keep frozen — don't extend it. |
| `archive/endo_market_v1/` | **Legacy** | Earliest Python iteration (formerly `endo_market/`). Don't extend it. |
| `archive/edl_simulator_v1/` | **Legacy prototype** | HTML/JS mockup; the earliest analytical version, superseded. |

When asked to change "the code" or "the experiment," default to
`archive/endo_market_v3/` unless the user names an older folder explicitly. v3 is
deliberately **self-contained**: derivation copies in `theory/`, data copies in
`data/`, experiments and tests inside the folder.

## `endo_market_v3` internals (package `reflex`)

Package root: `archive/endo_market_v3/reflex/`. The three strands and where they live:

- **Environment (`env/`)** — `simulator.py` (single-dealer `T_true`),
  `clients.py` (toxic/informed flow; the `toxicity_feedback` gain `ε`),
  `bonds.py`, `liquidity_field.py`, and **`multi_dealer.py`** (genuine
  `N`-dealer market sharing one informed pool with spillover `κ`; reduces
  bit-for-bit to the single-dealer market at `N = 1`).
- **Policies (`policy/`)** — `dealer_policy.py` (linear/MLP) and
  `glft_baseline.py` (non-learned closed-form baseline quoting the analytic
  `h_SP` or `h_PO`).
- **Operator (`operator/`)** — `response_operator.py`: `T_θ` with the v3
  un-blinding: `distribution_response()` / `toxic_slope()` read the *learned*
  `dD/dφ` out by autograd w.r.t. the policy summary. Whether it can learn that
  depends on **windowed fitting** (`operator.context_window` deployments per
  fit); frozen-summary optimisation keeps the blind v2 baseline available.
- **Theory (`theory/`)** — the five closed-form modules, numpy-only:
  `analytic_boundary` (1.1), `perfgd` (1.2), `multi_dealer` (1.3), `robust`
  (1.4), `factor_scaling` (1.5). Derivation documents ship in
  `archive/endo_market_v3/theory/` with a code map.
- **Loops (`equilibrium/`)** — `loops.py` `run_loop(mode=...)`:
  `rrm` (blind baseline) | `perfgd_analytic` (closed-form correction
  `Δ = −β(h−ψ)ε(h)` as a surrogate gradient) | `perfgd_learned` (live summary —
  the learned `dD/dφ` enters the gradient). Every iteration logs the learned
  toxic slope next to the analytic one (the ML↔math seam). `joint_loop.py`
  runs the simulated `N`-dealer cobweb + CRN joint-modulus probes.
  `rrm_loop.py` is the frozen v2-compatible baseline.
- **Estimators (`estimators/`)** — the three-way `ε` triangulation:
  `br_slope` (CRN best-response probe), `sinkhorn` (exact 1-D quantile W1 +
  debiased log-domain Sinkhorn), `cks` (fitted informed-flow-curve slope),
  `triangulate` (all three vs the closed form).
- **Calibration (`calibration/`)** — `loader.py` (shipped data CSVs),
  `mapping.py` (`(rating, regime) → Config`; the package's single
  unit-conversion point), `regimes.py` (VIX regimes).
- **Analysis (`analysis/`)** — `fragility.py` (the daily 1990–2026
  market-fragility index from real data), `phase.py` (analytic prediction
  curves, `(N, ε)` surface), plus `convergence`, `metrics`, `sweep`, `plots`.

Experiments (`experiments/`): `run_single`, `run_sweep` (predict-then-verify +
robust bands), `run_perfgd` (`--ml` for the three-mode loops), `run_dealers`,
`run_universe`, `run_triangulation`, `run_fragility`, `run_calibrated`, and
`run_all --profile smoke|full`. Configs in `configs/`; artifacts in `outputs/`.

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
  finite-difference diagnostics there, not local slopes. (The historical
  "saturates at ~1.25" figure was measured under the inflated-jitter
  protocol; see `research/analysis/pre-run-audit-2026-07.md` §1.6.)
- **Calibrated configs use real units** (per-$100-par, h ~ 0.4–3.5 per 100 par;
  one step ~ one trading day). Never hard-code absolute spread constants —
  probe widths, tolerances and caps must be *relative* to the configured spread
  scale. `reflex/calibration/mapping.py` is the only unit-conversion point.
- **Only `(A, k, σ, h)` are data-identified**; the toxic channel is
  structurally scaled (documented ratios in `mapping.py`). The crisis-regime
  intensity fit is degenerate (`k = 0`) — crisis boundaries sit on the anchor
  floor and are flagged. State all of this plainly in any write-up.
- **`summary_mode` is the blind/un-blinded switch.** Frozen summary = blind RRM
  (v2 convention); live summary + `operator.context_window ≥ 2` = the learned
  `dD/dφ` enters the gradient. `perfgd_learned` with window 1 is noise (the
  loop warns).
- **Multi-dealer runs can saturate `info_cap`**: the combined gross flow of `N`
  dealers inflates the shared liquidity field. Scale `liq_flow_boost` down or
  `info_cap` up for flow-allocation studies (see `env/multi_dealer.py`).
- **Smoke vs full profiles:** `run_all --profile smoke` proves the pipeline
  (tiny settings, ~minutes); scientific claims need `--profile full` (~10-15 min measured on current configs).
- **Console prints must stay ASCII** — the Windows console (cp1252) crashes on
  `λ`, `ε`, `m̂` etc. in `print()`. Unicode in matplotlib labels/docstrings is
  fine.
- **Math notation:** code/READMEs sometimes use ASCII (`phi`, `epsilon`, `gamma`,
  `dtau/dh`) and sometimes Unicode (`φ`, `ε`, `γ`). Match the surrounding file.
- This is a **Windows** environment with **PowerShell** as the primary shell; a
  Bash tool is also available. The repo path itself contains a space
  (`GitHub Projects`) — quote paths. Python 3.9 is the system interpreter
  (venv at `.venv/`); the code targets ≥ 3.9.

## Build / test / run (`archive/endo_market_v3/`)

```
../.venv/Scripts/python -m pip install -e .        # or: pip install -e .  (in the venv)
../.venv/Scripts/python -m pytest -q -m "not slow" # 103 fast tests
../.venv/Scripts/python -m pytest -q -m slow       # +7 slow end-to-end tests

../.venv/Scripts/python -m experiments.run_all --profile smoke   # everything, ~minutes
../.venv/Scripts/python -m experiments.run_all --profile full    # paper-grade, ~10-15 min
../.venv/Scripts/python -m experiments.run_fragility             # real-data index, seconds
```

110 tests total across 15 files (the 9 inherited from v2 — simulator, policy,
operator, rrm-convergence, the five analytic modules — plus `test_glft_baseline`,
`test_calibration`, `test_fragility`, `test_estimators`, `test_unblinded_operator`,
`test_perfgd_ml`, `test_multi_dealer_env`). Run from inside `archive/endo_market_v3/`,
using the repo venv (`.venv/` at the repo root — system Python has no torch).
Always run the fast tests after changing the library. See
`archive/endo_market_v3/README.md` for methodology, layout, and honest caveats.

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

- `math-theory/` — the five canonical derivations (1.1 analytic boundary
  `m = εβ/γ`, 1.2 PerfGD un-blinding, 1.3 multi-dealer `ε < γ/(N_eff·β)`,
  1.4 robust boundary `O(1/√n)`, 1.5 factor scaling), each `.md` with a
  compilable `.tex` twin (PDFs under `latex-papers/`). **Authoritative
  implementations now live in `archive/endo_market_v3/reflex/theory/`** (copies of the
  documents ship in `archive/endo_market_v3/theory/`); the older
  `archive/endo_market_v2/endo_market/analysis/` modules are the frozen originals.
- `data_collection/` + `preprocessing/` — the **canonical real-data pipeline**:
  ~36 yrs daily / 70 yrs monthly of *public, verified* macro + bond-factor
  series (CBOE VIX, EIA WTI, Fed H.15 10y, Shiller, gold/CPI,
  Dickerson–Mueller–Robotti TRACE bond factors, 212 real-CUSIP bond returns)
  joined into `REFLEX_MASTER_DATASET.csv`, cleaned/enriched, with fitted
  intensity params per (rating × regime). v3 ships copies of the artifacts it
  consumes (`archive/endo_market_v3/data/`). **Provenance gotcha (state this plainly
  in the paper):** it is *not* trade-level TRACE — `h`, per-dealer `q`, and
  real per-bond `A`/`k` need WRDS TRACE Enhanced (pending); the pipeline uses
  the closest free proxies. See `data_collection/docs/REJECTED_SOURCES.md`.
- `results/` — the executed paper-grade runs (one dated folder per
  `run_all --profile full` execution: raw per-experiment artifacts + logs,
  each recording the producing commit). Raw copies only; no post-processing.
- `analysis/` — the written analyses of those runs (derived tables, figures,
  predicted-vs-measured breakdowns, honest caveats) plus the pre-run
  measurement-layer audit. The live code/experiments themselves are inside
  `archive/endo_market_v3/` — this folder is the *application* of that package, not a
  second implementation.

## Current phase & next steps

Theory (1.1–1.5) derived and coded; the ML un-blinded and integrated; the
measurement layer **audited against the theory** (July 2026: six
probe/protocol defects found and fixed — sum-vs-mean dealer coupling, railed
probes, wrong probe points, mismatched prediction spreads, a dispersion
no-op, and inflated collection jitter; see
`research/analysis/pre-run-audit-2026-07.md`); the **paper-grade
full-profile suite executed** (8/8, ~10 min CPU — the "hours" estimate was
conservative) and curated into `research/results/07-10-2026/` with the
per-experiment analysis in `research/analysis/ANALYSIS-full-2026-07.md`.
The **live to-do**, in order:

1. **Write the ICAIF 2026 paper** (conference-ready; ACM `sigconf`, 8 pages,
   double-blind, deadline Aug 2 2026). Checklist in `research/README.md`
   (§ To-Do → ICAIF-specific requirements). Scoping is settled in the
   analysis: closed forms + real-data fragility + probe-level verifications
   are the headline; loop-level PerfGD stabilisation is reported as a
   diagnosed open gap (the ML artifacts are seam diagnostics).
2. **(Stretch)** close the loop-level gap: make the corrected *learned* loop
   find `h_PO` (operator `dJ/dh` anchored to the GLFT structural form,
   bias-compensating corrections, stability-penalty regularisation against
   the echo-chamber collapse).

Gotcha for future measurement work: probe protocols matter at first order —
keep `rrm.collection_jitter = 0.05` (0.2 inflates the CRN BR-slope probe
~3x), probe at the operating spread (not the analytic fixed point), and
compare against the *realized-state* closed form (1.1 §9).
