# math-theory

All mathematical proofs and derivations for the REFLEX research program. Each
document discharges one of the mathematical-contribution targets in
[`../README.md`](../README.md) (§1, "Mathematical contributions"), separating
what is established in prior literature from what this project derives.

## Contents

| Priority | Document | Status |
|----------|----------|--------|
| **1.1** | [`01-analytic-stability-boundary.md`](01-analytic-stability-boundary.md): closed-form `gamma`, `beta`, `epsilon` and the boundary `epsilon < gamma/beta` for the single-dealer market maker, with a predict-then-verify protocol against `analysis/response_modulus.py`. | **DONE** |
| **1.2** | [`02-perfgd-correction.md`](02-perfgd-correction.md): closed-form PerfGD correction `Delta = dT_J * dtau/dh`, convergence to the performative optimum at `O(1/k)` (linear at rate `1 - eta*gamma_PO`), stability beyond the RRM boundary `epsilon*`, and the `O(epsilon)` decision / `O(epsilon^2)` value echo-chamber gap. | **DONE** (derivation + code: [`equilibrium/perfgd_loop.py`](../../endo_market_v2/endo_market/equilibrium/perfgd_loop.py)) |
| **1.3** | [`03-multi-dealer-systemic-risk.md`](03-multi-dealer-systemic-risk.md): multi-dealer PSNE boundary `epsilon < gamma/(N_eff*beta)` (`N_eff = 1 + kappa*(N-1)`; headline `epsilon < gamma/(N*beta)` at full spillover), joint modulus `m_N = N_eff*m_1`, PSNE existence/uniqueness, joint convergence `gamma_joint = gamma - N_eff*epsilon*beta`, mean-field `N -> inf` limit, and the systemic critical dealer count `N_c = 1/m_1`. | **DONE** (derivation + code: [`analysis/multi_dealer_modulus.py`](../../endo_market_v2/endo_market/analysis/multi_dealer_modulus.py)) |
| **1.4** | [`04-robust-uncertainty.md`](04-robust-uncertainty.md): distributionally robust stability certificate `epsilon_hat_n + delta_n < gamma/beta` (robust boundary `epsilon*_rob = gamma/beta - delta_n`), the `O(1/sqrt(n))` ambiguity radius (parametric only via the CRN probe; naive difference is `O(n^{-1/3})`), sample complexity `n_req = (z*sigma/Delta)^2`, and the statistical-vs-structural uncertainty split. | **DONE** (derivation + code: [`analysis/robust_boundary.py`](../../endo_market_v2/endo_market/analysis/robust_boundary.py)) |
| **1.5** | [`05-factor-model-scaling.md`](05-factor-model-scaling.md): the `d x d` modulus matrix `M = beta*Gamma^{-1}*E` with boundary `rho(M) < 1`, the market-factor mode, `O(d*k^2)` factor reduction (Woodbury) of the diagonal-plus-low-rank `Gamma`, the truncation error bound linear in the residual factor variance `lambda_{k+1}(C)`, and the calibration map from bond statics (duration, DV01, spread vol) or synthetic CKS microstructure. | **DONE** (derivation + code: [`analysis/factor_reduction.py`](../../endo_market_v2/endo_market/analysis/factor_reduction.py)) |

## Code

The implementations of these derivations live in the **`endo_market_v2`** library
(the authoritative package, per the root `CLAUDE.md`), so every analytic claim runs
and is tested against the simulator:

| Priority | Module | Contents |
|----------|--------|----------|
| **1.1** | [`analysis/analytic_boundary.py`](../../endo_market_v2/endo_market/analysis/analytic_boundary.py) | Closed-form `gamma`, `beta`, `epsilon`, `tau`, `psi`, the best-response map, the fixed point `h*`, and the modulus `m`. Shared foundation for 1.2 and 1.3. |
| **1.2** | [`equilibrium/perfgd_loop.py`](../../endo_market_v2/endo_market/equilibrium/perfgd_loop.py) | The analytic PerfGD correction `Delta`, the performative optimum `h_PO`, `gamma_PO`, the echo-chamber gap, and the RRM-cobweb-vs-PerfGD demonstration. |
| **1.3** | [`analysis/multi_dealer_modulus.py`](../../endo_market_v2/endo_market/analysis/multi_dealer_modulus.py) | The `N`-dealer boundary, the joint Jacobian and its spectrum, the mean-field limits, the genuine `N`-dim joint cobweb, and the in-phase / anti-phase empirical probes (+ config fields `clients.n_dealers`, `clients.toxic_spillover`). |
| **1.4** | [`analysis/robust_boundary.py`](../../endo_market_v2/endo_market/analysis/robust_boundary.py) | The ambiguity radius, the robust certificate (stable / unstable / undecided), the `O(Δ^{-2})` sample-complexity curve, the `O(1/√n)` log-log rate check, the structural floor `eta_mod`, and a cross-seed empirical harness over the existing probe. |
| **1.5** | [`analysis/factor_reduction.py`](../../endo_market_v2/endo_market/analysis/factor_reduction.py) | The `d×d` modulus matrix `M = beta·Gamma^{-1}·E` and `rho(M)`, the `O(d·k^2)` Woodbury reduction, the `O(lambda_{k+1}(C))` truncation error bound, duration-calibrated `sigma_i`, and the 1.3×1.5 systemic composition. |

Tests: `endo_market_v2/tests/test_analytic_boundary.py`, `test_perfgd_loop.py`,
`test_multi_dealer.py`, `test_robust_boundary.py`, `test_factor_reduction.py` (run
`pytest -q -m "not slow"` from inside `endo_market_v2/`).

> **PDF build.** Priorities 1.1 and 1.2 have compiled PDFs in
> [`latex-papers/`](latex-papers/). The `.tex` companions for **1.3, 1.4, and 1.5**
> still need to be compiled into LaTeX PDFs via Overleaf and added to
> [`latex-papers/`](latex-papers/) (paste each `.tex` into an Overleaf project and
> download the PDF; the sources are self-contained and compile with `pdflatex`).

## Conventions

- Math is written to be liftable into the ICAIF submission (ACM `sigconf`,
  8-page double-blind). Keep each derivation self-contained, because ICAIF accepts
  no appendices, so proofs that matter must fit the main text.
- Symbols are mapped to `endo_market_v2` config fields in each document's
  symbol table; the canonical map lives in
  [`01-analytic-stability-boundary.md`](01-analytic-stability-boundary.md) §6.
- Every analytic claim states its validation protocol against the existing
  simulator/estimators; a derivation is only a contribution once it is
  falsifiable against code.
