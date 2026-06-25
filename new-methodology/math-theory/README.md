# math-theory

All mathematical proofs and derivations for the REFLEX research program. Each
document discharges one of the mathematical-contribution targets in
[`../README.md`](../README.md) (§1, "Mathematical contributions"), separating
what is established in prior literature from what this project derives.

## Contents

| Priority | Document | Status |
|----------|----------|--------|
| **1.1** | [`01-analytic-stability-boundary.md`](01-analytic-stability-boundary.md) — closed-form `γ`, `β`, `ε` and the boundary `ε < γ/β` for the single-dealer market maker, with a predict-then-verify protocol against `analysis/response_modulus.py`. | ✅ **Complete** |
| 1.2 | PerfGD un-blinding: analytic `dD/dφ`, `O(1/k)` convergence, echo-chamber gap. | ☐ Not started |
| 1.3 | Multi-dealer PSNE boundary `ε < γ/(Nβ)`, mean-field `N→∞` limit. | ☐ Not started |
| 1.4 | Distributionally robust `ε*` and `O(1/√n)` robust radius. | ☐ Not started |
| 1.5 | Factor-model dimensionality reduction and its error bound (100+ bonds). | ☐ Not started |

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
