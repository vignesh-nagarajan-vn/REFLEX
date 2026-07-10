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
| `endo_market_v2/` | **CURRENT — work here** | The live experiment and library (simulator + all five analytic modules). |
| `new-methodology/` | **Active — research program** | The forward roadmap and its deliverables: `math-theory/` (closed-form derivations 1.1–1.5, all **derived + coded**), the real-data pipeline (`data_collection/`, `preprocessing/`), plus still-placeholder `simulator/`, `experiments/`, `results/`. See "The research program" below. |
| `literature/literature-vignesh/` | Active reference | 10 foundational papers + reading map; PDFs downloaded. |
| `literature/literature-raghav/` | Active reference | Superset: same 10 + 8 extension papers (18 total) + research roadmap. |
| `endo_market_v1/` | **Legacy** | Earlier iteration (formerly `endo_market/`) superseded by `endo_market_v2`. Don't extend it. |
| `edl_simulator_v1/` | **Legacy prototype** | HTML/JS mockup; the earliest analytical version, superseded. |

When asked to change "the code" or "the experiment," default to
`endo_market_v2/` unless the user names a legacy folder explicitly. The
analytic-theory *code* also lives in `endo_market_v2/` (in `analysis/` and
`equilibrium/`); `new-methodology/` holds its written derivations, the dataset,
and the roadmap.

## `endo_market_v2` internals

Package root: `endo_market_v2/endo_market/`. Pipeline mirrors the RRM loop:

- `env/` — the market simulator (`simulator.py`), `bonds.py`, `clients.py`
  (toxic/informed flow; the `toxicity_feedback` gain `ε` lives here, plus the
  multi-dealer `n_dealers` / `toxic_spillover` knobs for 1.3),
  `liquidity_field.py`.
- `policy/dealer_policy.py` — the dealer's quoting policy `φ` (bid–ask
  half-spread `h`).
- `operator/` — the learned market response operator `T_θ` (`response_operator.py`,
  `heads.py`). Note: `T_θ` conditions on a `policy_summary` frozen within a
  deployment, so it is **blind to `dD/dφ`** — this is the central modeled
  limitation (PerfGD, paper #4, is the intended fix, now implemented in
  `perfgd_loop.py`).
- `equilibrium/` — the RRM machinery: `data_collection.py` → `fit_operator.py`
  → `optimize_policy.py`, orchestrated by `rrm_loop.py`; **plus `perfgd_loop.py`**
  (the analytic PerfGD-corrected loop, math-theory 1.2).
- `objective/` — `reward.py` (P&L, inventory penalty) and `stability.py`.
- `analysis/` — diagnostics **and the closed-form analytic-theory modules**:
  `response_modulus.py` (the model-free BR-slope modulus `m`), plus the
  math-theory implementations `analytic_boundary.py` (1.1: closed-form `γ`, `β`,
  `ε`, `τ`, `ψ`, `h*`, `m` — the shared foundation), `multi_dealer_modulus.py`
  (1.3: `ε < γ/(N_eff·β)`, joint Jacobian/spectrum, mean-field limit),
  `robust_boundary.py` (1.4: ambiguity radius, robust certificate, `O(1/√n)`
  rate), `factor_reduction.py` (1.5: `d×d` modulus matrix, Woodbury reduction,
  truncation bound); and `convergence.py`, `metrics.py`, `sweep.py`, `plots.py`.
- `utils/` — `seeding.py` (everything is reproducible from `(config, seed)`),
  `logging.py`.

Experiments: `experiments/run_single.py` and `experiments/run_sweep.py`.
Configs: `configs/{default,sweep_feedback,sweep_adversariality}.yaml`.
Outputs (phase-diagram PNG + sweep CSV) are written to `outputs/`.

## Conventions & gotchas

- **CPU-only, reproducible, no GPU.** Keep it that way. Every run must be
  deterministic from `(config, seed)`; route randomness through `utils/seeding.py`.
- **`ε` (`clients.toxicity_feedback`) is the clean control variable, not `α`
  (adversariality).** Sweeping `α` is confounded: high `α` drives the dealer to
  wide spreads where the toxic response `exp(−decay·h)` has decayed to ~0, so the
  modulus flattens or reverses (its sign isn't robust to universe size). Don't
  "fix" this — it's a documented feature. Use `sweep_feedback.yaml` for the
  headline result.
- **The modulus saturates (~1.25) past the boundary** rather than blowing up —
  defensive widening into a low-curvature (`γ → 0`) region. Expected behavior.
- **Math notation:** code/READMEs sometimes use ASCII (`phi`, `epsilon`, `gamma`,
  `dtau/dh`) and sometimes Unicode (`φ`, `ε`, `γ`). Match the surrounding file.
- This is a **Windows** environment with **PowerShell** as the primary shell; a
  Bash tool is also available. The repo path itself contains a space
  (`GitHub Projects`) — quote paths.

## Build / test / run (`endo_market_v2/`)

```
pip install -e .
pytest -q -m "not slow"        # 59 fast tests
pytest -q -m slow              # +4 slow end-to-end / boundary tests (~min each)

python -m experiments.run_single --config configs/default.yaml --seed 0
python -m experiments.run_sweep  --config configs/sweep_feedback.yaml
```

63 tests total across 9 files (`test_simulator`, `test_policy`, `test_operator`,
`test_rrm_convergence`, plus one per analytic module: `test_analytic_boundary`,
`test_perfgd_loop`, `test_multi_dealer`, `test_robust_boundary`,
`test_factor_reduction`). Run these from inside `endo_market_v2/`. Always run the
fast tests after changing the library. **Note:** the default environment here has
no `torch` — the run-the-model phase needs `pip install -e .` first. See
`endo_market_v2/README.md` for full methodology, the locked P&L identity, and
honest caveats.

## Working with the literature

The two `literature/*/README.md` files are the maps from each paper to a specific
code component and a concrete extension. When implementing a literature-driven
feature (e.g., a PerfGD correction, a `γ−εβ` plug-in estimator, a multi-dealer
extension), consult `literature-raghav/README.md` first — it names the target
theorem, required papers, and deliverable for each roadmap priority. PDFs are
fetched per-collection via `download_pdfs.sh` (open-access arXiv preprints).

## The research program & dataset (`new-methodology/`)

The novelty claim: derive the performativity stability boundary analytically
from microstructure primitives instead of sweeping it by hand. Structure:

- `math-theory/` — five derivations, **all DONE (derived + coded)**:
  1.1 analytic boundary `m = εβ/γ`, 1.2 PerfGD un-blinding, 1.3 multi-dealer
  systemic risk `ε < γ/(N_eff·β)`, 1.4 robust boundary `O(1/√n)`, 1.5
  factor-model scaling to 100+ bonds. Each `.md` has a compilable `.tex` twin
  (PDFs under `math-theory/latex-papers/`) and maps to a module in
  `endo_market_v2/endo_market/analysis/` (see the code map in
  `math-theory/README.md`). Remaining math sub-items: the Sinkhorn and
  CKS-implied `ε` estimators and the three-way triangulation.
- `data_collection/` + `preprocessing/` — the **real-data calibration dataset**
  (the "newly uploaded dataset"). ~36 yrs daily / 70 yrs monthly of *public,
  verified* macro + bond-factor series (CBOE VIX, EIA WTI, Fed H.15 10y,
  Shiller, gold/CPI, Dickerson–Mueller–Robotti TRACE bond factors, 212 real-CUSIP
  bond returns) joined into `REFLEX_MASTER_DATASET.csv`, then cleaned/enriched
  into calibration + held-out episode splits. **Provenance gotcha (state this
  plainly in the paper):** it is *not* trade-level TRACE — `h`, per-dealer `q`,
  and real per-bond `A`/`k` need WRDS TRACE Enhanced (pending); the pipeline uses
  the closest free proxies (VIX-implied spreads, Dickerson liquidity/credit
  factors). See `data_collection/docs/REJECTED_SOURCES.md`.
- `simulator/`, `experiments/`, `results/` — still placeholders; they fill in
  during the run-the-model phase below.

## Current phase & next steps

Theory (1.1–1.5) is derived and coded; the real-data calibration pipeline is
built. The **live to-do**, in order:

1. **Run the model applying the math theory + the dataset** — calibrate the
   simulator/analytic modules from the `new-methodology/` dataset, run the RRM
   and PerfGD loops and the `ε`/`N`/universe-size sweeps, and generate output
   data (phase diagrams, echo-chamber gap, multi-dealer boundary) with
   median + IQR bands over seeds.
2. **Analyze** the results (`new-methodology/results/` is the home for figures +
   raw data).
3. **Write the ICAIF 2026 paper** (conference-ready; ACM `sigconf`, 8 pages,
   double-blind, deadline Aug 2 2026). The submission checklist is in
   `new-methodology/README.md` (§ To-Do → ICAIF-specific requirements).

Most unchecked To-Do items now live under "Training and tuning" (run the loops,
sweeps, phase diagrams) and the paper — not the math. See
`new-methodology/README.md#to-do` for the authoritative checklist.
