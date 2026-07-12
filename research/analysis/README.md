# analysis/ - written analyses of the executed runs

The interpretation layer over [`../results/`](../results/): derived tables,
comparison figures, predicted-vs-measured breakdowns, and honest caveats.
Raw artifacts stay in `results/`; nothing here is a primary measurement.

| Document | What it covers |
|----------|----------------|
| [`pre-run-audit-2026-07.md`](pre-run-audit-2026-07.md) | The measurement-layer audit performed *before* the paper-grade runs: three failing end-to-end guards, root causes (mean-vs-sum dealer coupling, railed/no-signal probes, mismatched prediction spreads, dispersion no-op), fixes, and the scientific reframings they forced |
| [`ANALYSIS-full-2026-07.md`](ANALYSIS-full-2026-07.md) | The master per-experiment analysis of the **v3 run** (`../results/07-10-2026/`): fragility index, calibrated boundaries, phase-diagram sweep with robust bands, PerfGD seam, dealer amplification, universe scaling, triangulation, and the alpha-confound appendix |
| `figures/` | Derived comparison figures generated from the result CSVs |

The **v4 run** (`../results/07-12-2026/`, 11 experiments) carries its findings
in its own [`REPORT.md`](../results/07-12-2026/REPORT.md) - the v4 additions
(the structural-loop gap closure, the 1.6 lazy-deploy sweep, the estimator
tuning, the proof certificates) plus re-executions of the v3 experiment set
by the v4 code. The v3 conclusions above were not re-litigated: the v4
re-runs are consistency checks, and the per-experiment deep-dive remains
`ANALYSIS-full-2026-07.md`.
