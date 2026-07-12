# archive/ - the superseded REFLEX generations

Frozen prior generations of the REFLEX program, kept for provenance and for
the executed results that reference them. **Do not extend anything in this
folder** - the current, authoritative implementation is
[`../endo_market_v4/`](../endo_market_v4/) (package `reflex`).

| Folder | Generation | Status | One line |
|--------|-----------|--------|----------|
| [`edl_simulator_v1/`](edl_simulator_v1/) | 1st prototype | frozen | HTML/JS mockup of the analytical linear-quadratic model; proved the concept (one parameter flips a market between convergence and chaos). |
| [`endo_market_v1/`](endo_market_v1/) | 2nd | frozen | First Python/PyTorch build of the learned-operator RRM loop; scaffolding landed, the `alpha*` result did not reproduce. |
| [`endo_market_v2/`](endo_market_v2/) | 3rd | frozen | Identified `epsilon` as the clean control, reproduced the `eps < gamma/beta` boundary, and grew the five analytic modules later absorbed into v3/v4 (its measured crossing was later shown protocol-inflated; see its README). |
| [`endo_market_v3/`](endo_market_v3/) | 4th | frozen | The un-blinded self-contained `reflex` package: theory 1.1-1.5 + real-data calibration + the audited measurement layer. Executed paper-grade run curated in [`../research/results/07-10-2026/`](../research/results/07-10-2026/) (its artifacts cite v3 commits). `endo_market_v4` is a faithful superset. |

Provenance notes:

- The July-2026 paper-grade run in `research/results/07-10-2026/` was produced
  by `endo_market_v3` at commit `815425d`; the raw artifacts and analyses
  reference that code. v3 is kept intact (tests and configs included) so the
  run remains reproducible from history.
- The canonical math derivations live in `../research/math-theory/`; the
  frozen v2 implementations they originally shipped against live under
  `endo_market_v2/endo_market/analysis/`.
- Each folder's own README states its scope, mechanism and what superseded it.
