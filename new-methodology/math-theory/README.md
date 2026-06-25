# math-theory

All mathematical proofs and derivations for the REFLEX research program. Each
document discharges one of the mathematical-contribution targets in
[`../README.md`](../README.md) (§1, "Mathematical contributions"), separating
what is established in prior literature from what this project derives.

## Contents

| Priority | Document | Status |
|----------|----------|--------|
| **1.1** | [`01-analytic-stability-boundary.md`](01-analytic-stability-boundary.md) — closed-form `gamma`, `beta`, `epsilon` and the boundary `epsilon < gamma/beta` for the single-dealer market maker, with a predict-then-verify protocol against `analysis/response_modulus.py`. | **DONE** |
| **1.2** | [`02-perfgd-correction.md`](02-perfgd-correction.md) — closed-form PerfGD correction `Delta = dT_J * dtau/dh`, convergence to the performative optimum at `O(1/k)` (linear at rate `1 - eta*gamma_PO`), stability beyond the RRM boundary `epsilon*`, and the `O(epsilon)` decision / `O(epsilon^2)` value echo-chamber gap. | **DONE (derivation)**; loop code pending |
| 1.3 | Multi-dealer PSNE boundary `epsilon < gamma/(N*beta)`, mean-field `N -> inf` limit. | Not started |
| 1.4 | Distributionally robust `epsilon*` and `O(1/sqrt(n))` robust radius. | Not started |
| 1.5 | Factor-model dimensionality reduction and its error bound (100+ bonds). | Not started |

## Conventions

- Math is written to be liftable into the ICAIF submission (ACM `sigconf`,
  8-page double-blind). Keep each derivation self-contained — ICAIF accepts no
  appendices, so proofs that matter must fit the main text.
- Symbols are mapped to `endo_market_v2` config fields in each document's
  symbol table; the canonical map lives in
  [`01-analytic-stability-boundary.md`](01-analytic-stability-boundary.md) §6.
- Every analytic claim states its validation protocol against the existing
  simulator/estimators — a derivation is only a contribution once it is
  falsifiable against code.
