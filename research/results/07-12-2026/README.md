# 07-12-2026 - paper-grade run of the complete v4 experiment suite

| | |
|---|---|
| **Producing code** | `endo_market_v4` at commit `2718012` (the archive-move commit; all v4 feature commits `b9f67d1`..`3bd4bc9` included) plus the documentation-only working-tree delta that lands in the same commit as these artifacts (three files, docstrings + the `__version__` string; zero experiment-code changes - verified by `git diff --stat`) |
| **Command** | `python -u -m experiments.run_all --profile full` (from `endo_market_v4/`, repo venv) |
| **Result** | suite **11/11 passed in 25.1 min** (certificates 0s, fragility 1s, calibrated 106s, universe 1s, perfgd 477s, dealers 5s, triangulation 7s, sweep 319s, lazy_deploy 105s, tuning 398s, single 89s) |
| **Environment** | Python 3.9.13, torch 2.8.0+cpu, numpy 2.0.2, pandas 2.3.3; Windows 11, CPU only |
| **Determinism** | every experiment seeded from its config; sweep seeds 0-7; lazy-deploy seeds 0-2; tuning probe seeds 0-5 |
| **Preconditions** | 152/152 tests green (142 fast + 10 slow) before the run; all 66 proof certificates passing on raw + calibrated configs |

**The illustrated report of findings: [`REPORT.md`](REPORT.md).** The
per-experiment deep-dive of the v3-era experiment set remains
[`../../analysis/ANALYSIS-full-2026-07.md`](../../analysis/ANALYSIS-full-2026-07.md)
(the v4 re-runs reproduce its numbers; see the report's consistency section).

Contents (artifacts copied verbatim from `endo_market_v4/outputs/`):

| Folder | Experiment | Artifacts |
|--------|-----------|-----------|
| `certificates/` | `run_certificates` - 66 numerical proof checks (raw + calibrated IG/normal) | checks table CSV |
| `fragility/` | `run_fragility` - daily 1990-2026 boundary on real data | headline PNG, daily + by-regime CSVs (IG, HY) |
| `calibrated/` | `run_calibrated --measure --seeds 3` - a-priori boundary per (rating x regime) | table CSV + bar PNG |
| `universe/` | `run_universe` - rho(M) to 128 bonds, truncation bound | scaling + truncation CSVs, PNG |
| `perfgd/` | `run_perfgd --ml --iters 10` - gap scan, beyond-boundary demo, **four-mode** ML loops incl. `perfgd_structural` | gap CSV, demo PNG, ML CSV + PNG (with the three-way seam) |
| `dealers/` | `run_dealers --probe --episodes 8` - (N, f) surface + genuine-market probes | grid CSV, surface PNG, probe CSV |
| `triangulation/` | `run_triangulation --episodes 8` - three-way epsilon | CSV |
| `sweep/` | `run_sweep` - feedback phase diagram (8 seeds, robust bands) | CSV + PNG |
| `lazy_deploy/` | `run_lazy_deploy` - the theory-1.6 K sweep (K in 1..20, 3 seeds, full-BR anchor, c-fit) | sweep + summary CSVs, two-panel PNG |
| `tuning/` | `run_tuning` - Sinkhorn blur bias curves + robust-radius coverage calibration | 2 CSVs + PNG |
| `single/` | `run_single --mode perfgd_analytic` - one instrumented loop | trajectory PNG |
| `logs/` | full console log | `run_all_full.log` |

Note: the alpha-confound appendix sweep (`sweep_adversariality.yaml`) was not
rerun - its v3 result stands unchanged in
[`../07-10-2026/sweep/`](../07-10-2026/sweep/) and nothing in v4 touches the
alpha channel.
