# 07-10-2026 - paper-grade run of the complete experiment suite

| | |
|---|---|
| **Producing code** | commit `815425d` (post-audit measurement layer) for the main suite; the two sweeps are the **clean-protocol rerun** after the jitter fix (audit §1.6), whose code lands in the same commit as these artifacts |
| **Commands** | `python -u -m experiments.run_all --profile full`; then `python -u -m experiments.run_sweep --config configs/sweep_feedback.yaml` and `... --config configs/sweep_adversariality.yaml` rerun with `collection_jitter = 0.05` (from `endo_market_v3/`, repo venv) |
| **Result** | suite **8/8 passed in 9.1 min**; clean sweeps ~4 + ~3 min |
| **Environment** | Python 3.9.13, torch 2.8.0+cpu, numpy 2.0.2, pandas 2.3.3; Windows 11, CPU only |
| **Determinism** | every experiment seeded from its config; sweep seeds 0–7 (feedback) / 100–104 (alpha) |
| **Preconditions** | 110/110 tests green after the [measurement-layer audit](../../analysis/pre-run-audit-2026-07.md), re-verified after the jitter fix |

**The complete illustrated report (all figures + findings): [`REPORT.md`](REPORT.md).**
Extended per-experiment interpretation:
[`../../analysis/ANALYSIS-full-2026-07.md`](../../analysis/ANALYSIS-full-2026-07.md).

Contents (artifacts copied verbatim from `endo_market_v3/outputs/`):

| Folder | Experiment | Artifacts |
|--------|-----------|-----------|
| `fragility/` | `run_fragility` - daily 1990–2026 boundary on real data | headline PNG, daily + by-regime CSVs (IG, HY) |
| `calibrated/` | `run_calibrated --measure --seeds 3` - a-priori boundary per (rating x regime) | table CSV + bar PNG |
| `universe/` | `run_universe` - rho(M) to 128 bonds, truncation bound | scaling + truncation CSVs, PNG |
| `perfgd/` | `run_perfgd --ml --iters 10` - gap scan, beyond-boundary demo, 3-mode ML loops | gap CSV, demo PNG, ML CSV + PNG |
| `dealers/` | `run_dealers --probe --episodes 8` - (N, f) surface + genuine-market probes | grid CSV, surface PNG, probe CSV |
| `triangulation/` | `run_triangulation --episodes 8` - three-way epsilon | CSV |
| `sweep/` | `run_sweep` - feedback phase diagram (8 seeds, robust bands) + alpha appendix | 2 CSVs + 2 PNGs |
| `single/` | `run_single --mode perfgd_analytic` - one instrumented loop | trajectory PNG |
| `logs/` | full console logs | `run_all_full.log`, `run_sweep_alpha.log` |
