# analysis/ - written analyses of the executed runs

The interpretation layer over [`../results/`](../results/): derived tables,
comparison figures, predicted-vs-measured breakdowns, and honest caveats.
Raw artifacts stay in `results/`; nothing here is a primary measurement.

| Document | What it covers |
|----------|----------------|
| [`pre-run-audit-2026-07.md`](pre-run-audit-2026-07.md) | The measurement-layer audit performed *before* the paper-grade runs: three failing end-to-end guards, root causes (mean-vs-sum dealer coupling, railed/no-signal probes, mismatched prediction spreads, dispersion no-op), fixes, and the scientific reframings they forced |
| `ANALYSIS-full-2026-07.md` | (lands with the full-profile results) The master per-experiment analysis: fragility index, calibrated boundaries, phase-diagram sweep with robust bands, PerfGD seam, dealer amplification, universe scaling, triangulation, and the alpha-confound appendix |
| `figures/` | Derived comparison figures generated from the result CSVs |
