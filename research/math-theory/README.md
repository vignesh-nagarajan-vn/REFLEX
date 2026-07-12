# math-theory

All mathematical proofs and derivations for the REFLEX research program. Each
document discharges one of the mathematical-contribution targets in
[`../README.md`](../README.md) (§1, "Mathematical contributions"), separating
what is established in prior literature from what this project derives.

## Contents

| Priority | Document | Status |
|----------|----------|--------|
| **1.1** | [`01-analytic-stability-boundary.md`](01-analytic-stability-boundary.md): closed-form `gamma`, `beta`, `epsilon` and the boundary `epsilon < gamma/beta` for the single-dealer market maker, with a predict-then-verify protocol against `analysis/response_modulus.py`. | **DONE** |
| **1.2** | [`02-perfgd-correction.md`](02-perfgd-correction.md): closed-form PerfGD correction `Delta = dT_J * dtau/dh`, convergence to the performative optimum at `O(1/k)` (linear at rate `1 - eta*gamma_PO`), stability beyond the RRM boundary `epsilon*`, and the `O(epsilon)` decision / `O(epsilon^2)` value echo-chamber gap. | **DONE** (derivation + code: [`equilibrium/perfgd_loop.py`](../../archive/endo_market_v2/endo_market/equilibrium/perfgd_loop.py)) |
| **1.3** | [`03-multi-dealer-systemic-risk.md`](03-multi-dealer-systemic-risk.md): multi-dealer PSNE boundary `epsilon < gamma/(N_eff*beta)` (`N_eff = 1 + kappa*(N-1)`; headline `epsilon < gamma/(N*beta)` at full spillover), joint modulus `m_N = N_eff*m_1`, PSNE existence/uniqueness, joint convergence `gamma_joint = gamma - N_eff*epsilon*beta`, mean-field `N -> inf` limit, and the systemic critical dealer count `N_c = 1/m_1`. | **DONE** (derivation + code: [`analysis/multi_dealer_modulus.py`](../../archive/endo_market_v2/endo_market/analysis/multi_dealer_modulus.py)) |
| **1.4** | [`04-robust-uncertainty.md`](04-robust-uncertainty.md): distributionally robust stability certificate `epsilon_hat_n + delta_n < gamma/beta` (robust boundary `epsilon*_rob = gamma/beta - delta_n`), the `O(1/sqrt(n))` ambiguity radius (parametric only via the CRN probe; naive difference is `O(n^{-1/3})`), sample complexity `n_req = (z*sigma/Delta)^2`, and the statistical-vs-structural uncertainty split. | **DONE** (derivation + code: [`analysis/robust_boundary.py`](../../archive/endo_market_v2/endo_market/analysis/robust_boundary.py)) |
| **1.5** | [`05-factor-model-scaling.md`](05-factor-model-scaling.md): the `d x d` modulus matrix `M = beta*Gamma^{-1}*E` with boundary `rho(M) < 1`, the market-factor mode, `O(d*k^2)` factor reduction (Woodbury) of the diagonal-plus-low-rank `Gamma`, the truncation error bound linear in the residual factor variance `lambda_{k+1}(C)`, and the calibration map from bond statics (duration, DV01, spread vol) or synthetic CKS microstructure. | **DONE** (derivation + code: [`analysis/factor_reduction.py`](../../archive/endo_market_v2/endo_market/analysis/factor_reduction.py)) |
| **1.6** | [`06-lazy-deployment.md`](06-lazy-deployment.md) (v4 addendum): the K-step outer map `mu(K) = -m + c^K(1+m)` (lazy-deploy interpolation of the exact cobweb), the deadbeat and max-stable step counts (laziness stabilises an `m > 1` market up to `K_max`), the two-branch effective curvature `gamma_eff(K) = gamma*m/\|mu(K)\|` split at the equal-modulus count, and the signed-CRN-probe + `c`-fit protocol. | **DONE** (derivation + code: [`reflex/theory/lazy_deploy.py`](../../endo_market_v4/reflex/theory/lazy_deploy.py)) |

## Code

The **authoritative implementations** live in the `reflex` package
(**`endo_market_v4/`**, per the root `CLAUDE.md`) as the numpy-only subpackage
`reflex.theory`; copies of these documents ship with the package under
`endo_market_v4/theory/`. The `endo_market_v3` copies and the original
`endo_market_v2` modules are kept frozen under `archive/`.

| Priority | Module (authoritative) | Contents |
|----------|------------------------|----------|
| **1.1** | [`reflex/theory/analytic_boundary.py`](../../endo_market_v4/reflex/theory/analytic_boundary.py) | Closed-form `gamma`, `beta`, `epsilon`, `tau`, `psi`, the best-response map, the fixed point `h*`, and the modulus `m`. Shared foundation for 1.2 and 1.3. |
| **1.2** | [`reflex/theory/perfgd.py`](../../endo_market_v4/reflex/theory/perfgd.py) | The analytic PerfGD correction `Delta`, the performative optimum `h_PO`, `gamma_PO`, the echo-chamber gap, and the RRM-cobweb-vs-PerfGD demonstration. The *training-loop* realisation (analytic + free-form learned + v4 structural correction modes) is `reflex/equilibrium/loops.py` + `reflex/equilibrium/structural_response.py`. |
| **1.3** | [`reflex/theory/multi_dealer.py`](../../endo_market_v4/reflex/theory/multi_dealer.py) | The `N`-dealer boundary, the joint Jacobian and its spectrum, the mean-field limits, and the empirical probes. The *genuine* simulated `N`-dealer market is `reflex/env/multi_dealer.py` + `reflex/equilibrium/joint_loop.py`. |
| **1.4** | [`reflex/theory/robust.py`](../../endo_market_v4/reflex/theory/robust.py) | The ambiguity radius, the robust certificate (stable / unstable / undecided), the `O(Δ^{-2})` sample-complexity curve, the `O(1/√n)` log-log rate check, the structural floor `eta_mod`, and the v4 coverage-calibrated radius (`calibrate_radius`); robust bands are wired into every `run_sweep`. |
| **1.5** | [`reflex/theory/factor_scaling.py`](../../endo_market_v4/reflex/theory/factor_scaling.py) | The `d×d` modulus matrix `M = beta·Gamma^{-1}·E` and `rho(M)`, the `O(d·k^2)` Woodbury reduction, the `O(lambda_{k+1}(C))` truncation error bound, data-calibrated `sigma_i`, and the 1.3×1.5 systemic composition. |
| **1.6** | [`reflex/theory/lazy_deploy.py`](../../endo_market_v4/reflex/theory/lazy_deploy.py) | `mu(K)`, `m_eff`, `lam_K`, the deadbeat / equal-modulus / max-stable step counts, the two-branch `gamma_eff`, and the one-parameter `c`-fit consumed by the signed CRN K-step probe (`reflex/estimators/br_slope.py: measure_rgd_response`). |

Tests: `endo_market_v4/tests/` (152 tests; run `pytest -q -m "not slow"` from
inside `endo_market_v4/` using the repo venv). The model-free measurement side
(the `epsilon` triangulation) is `reflex/estimators/`; the real-data fragility
index built on 1.1 is `reflex/analysis/fragility.py`. Every load-bearing
identity of 1.1-1.6 is additionally re-derived numerically by the v4
verification layer (`reflex/verification/certificates.py`, 66 checks) with the
logical skeletons formalised in Lean 4 (`endo_market_v4/lean/`; formal
statements reviewed, compile pending a toolchain - see its README).

> **PDF build.** Priorities 1.1 through 1.5 have compiled PDFs in
> [`latex-papers/`](latex-papers/) (created via Overleaf). 1.6 ships `.md` +
> `.tex` ([`06-lazy-deployment.tex`](06-lazy-deployment.tex)); its PDF is
> pending the next Overleaf pass.

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
