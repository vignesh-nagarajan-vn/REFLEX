# theory/ - the five closed-form derivations (shipped copies)

Self-contained copies of the REFLEX math-theory derivations (canonical originals
live in `../../../research/math-theory/`, including `.tex` sources and
compiled PDFs under `latex-papers/`). Each maps to a module in
[`../reflex/theory/`](../reflex/theory/):

| Priority | Derivation | Implementation | What it gives |
|----------|-----------|----------------|----------------|
| **1.1** | [`01-analytic-stability-boundary.md`](01-analytic-stability-boundary.md) | [`reflex/theory/analytic_boundary.py`](../reflex/theory/analytic_boundary.py) | Closed-form `gamma`, `beta`, `epsilon(h)`, `tau(h)`, `psi`, the fixed point `h*`, and the modulus `m = epsilon*beta/gamma` (stable iff `< 1`). |
| **1.2** | [`02-perfgd-correction.md`](02-perfgd-correction.md) | [`reflex/theory/perfgd.py`](../reflex/theory/perfgd.py) | The analytic PerfGD correction `Delta = -beta*(h-psi)*epsilon(h)`, the performative optimum `h_PO`, `gamma_PO`, the echo-chamber gap, and the cobweb-vs-corrected dynamics. |
| **1.3** | [`03-multi-dealer-systemic-risk.md`](03-multi-dealer-systemic-risk.md) | [`reflex/theory/multi_dealer.py`](../reflex/theory/multi_dealer.py) | The `N`-dealer boundary `epsilon < gamma/(N_eff*beta)` with `N_eff = 1 + kappa*(N-1)`, the joint Jacobian/spectrum, mean-field limits, and the critical dealer count `N_c = 1/m_1`. |
| **1.4** | [`04-robust-uncertainty.md`](04-robust-uncertainty.md) | [`reflex/theory/robust.py`](../reflex/theory/robust.py) | The ambiguity radius `delta_n`, the robust certificate `epsilon_hat + delta_n < gamma/beta` (stable / unstable / undecided), `O(1/sqrt(n))` rate machinery, and `n_req = O(Delta^-2)` sample complexity. |
| **1.5** | [`05-factor-model-scaling.md`](05-factor-model-scaling.md) | [`reflex/theory/factor_scaling.py`](../reflex/theory/factor_scaling.py) | The `d x d` modulus matrix `M = beta*Gamma^-1*E` with boundary `rho(M) < 1`, the `O(d*k^2)` Woodbury reduction, and the truncation bound `O(lambda_{k+1}(C))`. |

The model-free measurement side (the `epsilon` triangulation: BR-slope /
Sinkhorn / CKS) lives in [`../reflex/estimators/`](../reflex/estimators/); the
ML loops that consume the corrections live in
[`../reflex/equilibrium/`](../reflex/equilibrium/).

Convention: every analytic claim ships with a validation protocol against the
simulator - a derivation only counts once it is falsifiable against code.
