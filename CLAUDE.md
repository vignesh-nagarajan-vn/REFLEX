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
| `endo_market_v2/` | **CURRENT — work here** | The live experiment and library. |
| `literature/literature-vignesh/` | Active reference | 10 foundational papers + reading map; PDFs downloaded. |
| `literature/literature-raghav/` | Active reference | Superset: same 10 + 8 extension papers (18 total) + research roadmap. |
| `endo_market/` | **Legacy** | Earlier iteration superseded by `endo_market_v2`. Don't extend it. |
| `edl_simulator_v1/` | **Legacy prototype** | HTML/JS mockup; the earliest analytical version, superseded. |

When asked to change "the code" or "the experiment," default to
`endo_market_v2/` unless the user names a legacy folder explicitly.

## `endo_market_v2` internals

Package root: `endo_market_v2/endo_market/`. Pipeline mirrors the RRM loop:

- `env/` — the market simulator (`simulator.py`), `bonds.py`, `clients.py`
  (toxic/informed flow; the `toxicity_feedback` gain `ε` lives here),
  `liquidity_field.py`.
- `policy/dealer_policy.py` — the dealer's quoting policy `φ` (bid–ask
  half-spread `h`).
- `operator/` — the learned market response operator `T_θ` (`response_operator.py`,
  `heads.py`). Note: `T_θ` conditions on a `policy_summary` frozen within a
  deployment, so it is **blind to `dD/dφ`** — this is the central modeled
  limitation (PerfGD, paper #4, is the intended fix).
- `equilibrium/` — the RRM machinery: `data_collection.py` → `fit_operator.py`
  → `optimize_policy.py`, orchestrated by `rrm_loop.py`.
- `objective/` — `reward.py` (P&L, inventory penalty) and `stability.py`.
- `analysis/` — diagnostics: `response_modulus.py` (the BR-slope modulus `m`),
  `convergence.py`, `metrics.py`, `sweep.py`, `plots.py`.
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
pytest -q -m "not slow"        # 20 fast tests
pytest -q -m slow              # +1 end-to-end stability-boundary test (~2 min)

python -m experiments.run_single --config configs/default.yaml --seed 0
python -m experiments.run_sweep  --config configs/sweep_feedback.yaml
```

Run these from inside `endo_market_v2/`. Always run the fast tests after
changing the library. See `endo_market_v2/README.md` for full methodology, the
locked P&L identity, and honest caveats.

## Working with the literature

The two `literature/*/README.md` files are the maps from each paper to a specific
code component and a concrete extension. When implementing a literature-driven
feature (e.g., a PerfGD correction, a `γ−εβ` plug-in estimator, a multi-dealer
extension), consult `literature-raghav/README.md` first — it names the target
theorem, required papers, and deliverable for each roadmap priority. PDFs are
fetched per-collection via `download_pdfs.sh` (open-access arXiv preprints).

## Project goals (context for prioritization)

Sharpen the research question with novel ML depth; integrate the literature to
extend `endo_market_v2` (predictive `εβ/γ` boundary, PerfGD-corrected loop,
adaptive `ε` exploration, inventory-stateful convergence, 100+-bond phase
diagram); target a submission to ICAIF 2026 or a comparable venue.
