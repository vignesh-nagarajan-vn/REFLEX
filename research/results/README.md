# results/ - executed paper-grade runs

Raw artifacts from the `reflex` experiment suite executed against the shipped
real-data calibrations. One dated folder per full-profile execution:

    results/
    |- 07-10-2026/             the v3 run (8 experiments; produced by archive/endo_market_v3 @ 815425d)
    |- 07-12-2026/             the v4 run (11 experiments; produced by endo_market_v4 - the final generation)
    |
    |  each run folder:
    |  |- README.md            what ran, at which commit, timings, environment
    |  |- REPORT.md            the illustrated report (figures + findings)
    |  |- logs/                full console logs of the suite
    |  |- fragility/           per-experiment artifacts (CSVs + PNGs)
    |  |- calibrated/          ...
    |  |- sweep/               (+ the alpha-confound appendix sweep, when rerun)
    |  |- perfgd/
    |  |- dealers/
    |  |- triangulation/
    |  |- universe/
    |  |- single/
    |  |- certificates/        (v4 runs) the numerical proof-check table
    |  |- lazy_deploy/         (v4 runs) the theory-1.6 K sweep
    |  \- tuning/              (v4 runs) Sinkhorn blur + robust radius calibration

Conventions:

- Every run folder records the **git commit** of the code that produced it and
  the exact commands; runs are deterministic from `(config, seed)`.
- Raw artifacts are copied verbatim from the producing package's `outputs/`
  (v3 runs: `archive/endo_market_v3/outputs/`; v4 runs:
  `endo_market_v4/outputs/`); nothing is post-processed inside `results/`.
  Derived tables/figures and the written interpretation live in
  [`../analysis/`](../analysis/).
- The fragility index, calibrated a-priori boundaries, universe scaling and
  the proof certificates are closed forms on real data (full-fidelity in any
  profile); the sweep / PerfGD / dealer / triangulation / lazy-deploy
  artifacts are only paper-grade when produced by the `full` profile.
