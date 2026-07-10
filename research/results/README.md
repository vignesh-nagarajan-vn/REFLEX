# results/ - executed paper-grade runs

Raw artifacts from the `endo_market_v3` experiment suite executed against the
shipped real-data calibrations. One dated folder per full-profile execution:

    results/
    |- MM-DD-YYYY/             one complete `run_all --profile full` execution (e.g. 07-10-2026/)
    |  |- README.md            what ran, at which commit, timings, environment
    |  |- REPORT.md            the complete illustrated report (all figures + findings)
    |  |- logs/                full console logs of the suite
    |  |- fragility/           per-experiment artifacts (CSVs + PNGs) + notes.md
    |  |- calibrated/          ...
    |  |- sweep/               (+ the alpha-confound appendix sweep)
    |  |- perfgd/
    |  |- dealers/
    |  |- triangulation/
    |  |- universe/
    |  \- single/

Conventions:

- Every run folder records the **git commit** of the code that produced it and
  the exact commands; runs are deterministic from `(config, seed)`.
- Raw artifacts are copied verbatim from `endo_market_v3/outputs/`; nothing is
  post-processed inside `results/`. Derived tables/figures and the written
  interpretation live in [`../analysis/`](../analysis/).
- The fragility index, calibrated a-priori boundaries and universe scaling are
  closed forms on real data (full-fidelity in any profile); the sweep / PerfGD
  / dealer / triangulation artifacts are only paper-grade when produced by the
  `full` profile.
