# lean/ - formal (Lean 4) skeletons of the REFLEX theory

Lean 4 + mathlib4 formalisations of the **logical and algebraic skeletons** of
theory 1.1-1.6. This is one half of the v4 verification layer; the other half
is the numerical certificate suite
([`reflex/verification/certificates.py`](../reflex/verification/certificates.py),
run via `python -m experiments.run_certificates`), which checks the
*model-specific* content (the GLFT curvature integrals, the gate means, fixed
point locations, Monte-Carlo rates) that a formal real-analysis development
would restate without adding assurance about the code.

## What is formalised

| File | Result | Statement |
|------|--------|-----------|
| `Reflex/Contraction.lean` | 1.1 (dynamics) | The linearised cobweb `x_{n+1} = a x_n + b` has closed form `x_n - x* = a^n (x0 - x*)`; it **converges for `\|a\| < 1`** and its distance to the fixed point **tends to infinity for `\|a\| > 1`** - the two sides of the `m < 1` boundary. |
| `Reflex/EchoChamber.lean` | 1.2 (separation) | A root of the blind gradient is **not** a stationary point of the corrected objective when the correction is nonzero there; under strict concavity the optimum lies strictly on the side the correction points to (`Delta(h_SP) < 0 => h_PO < h_SP`: the blind dealer over-defends). |
| `Reflex/MultiDealer.lean` | 1.3 (eigen-identity + algebra) | The all-ones vector is an eigenvector of the joint Jacobian (diag `-m1`, off-diag `-m1*kappa`) with eigenvalue `-m1*(1 + kappa(N-1)) = -m1*N_eff`; the boundary restatement `N_eff*(eps*beta/gamma) < 1 <-> eps < gamma/(N_eff*beta)`; monotonicity of `N_eff` in `kappa`. |
| `Reflex/Robust.lean` | 1.4 (Theorem 1 core) | Certificate soundness: `\|mhat - m\| <= delta` and `mhat + delta < 1` imply `m < 1` (and the unstable side); the undecided band is honest (both truths consistent); radius monotonicity. |
| `Reflex/LazyDeploy.lean` | 1.6 (complete algebra) | `mu(K) = -m + c^K(1+m)`: `mu(0) = 1`, the lazy-deploy interpolation identity, strict decrease in `K`, the `K -> infinity` limit `-m`, and the stability-window inequalities (`-1 < mu(K) <-> m - 1 < c^K(1+m)`; `mu(K) < 1` for `K >= 1`). |

**Deliberately not formalised** (numerically certified instead): the
derivations of `gamma`, `beta`, `epsilon`, `psi` from the market model (1.1
§2-3), `gamma_PO` and the gap magnitudes (1.2 §4-6), the mean-field limits
(1.3 §7), the `O(1/sqrt(n))` statistics (1.4 §2-3), and the entire
factor-scaling linear algebra (1.5 - mathlib's matrix analysis could carry the
Woodbury identity, but the *bound vs measured error* comparison is inherently
numerical). The split principle: formalise where the risk is a logic error,
certify numerically where the risk is a modelling/implementation error.

## Building

Requires [elan](https://github.com/leanprover/elan) (the Lean toolchain
manager). From this directory:

```
lake exe cache get   # fetch prebuilt mathlib oleans (several GB)
lake build
```

The toolchain is pinned by `lean-toolchain` (`leanprover/lean4:v4.11.0`) and
mathlib by `lakefile.toml` (`v4.11.0`).

## Compile status (honest)

**These files have NOT been compiled in this repository's development
environment**: no Lean toolchain is installed on the development machine, and
installing one (an external multi-GB download) was out of scope for the v4
build session. The proofs are written conservatively against mathlib4 v4.11.0
idioms (`tendsto_pow_atTop_nhds_zero_of_abs_lt_one`,
`pow_lt_pow_right_of_lt_one`, `Finset.sum_ite_eq`, `abs_le`, `linarith` /
`nlinarith` closures), but lemma-name drift or small tactic gaps are possible
until `lake build` has been run once. Treat the *numerical certificate suite*
as the verification of record; treat these files as reviewed formal statements
awaiting a toolchain. If a build surfaces breakage, fixes are expected to be
local (names/tactics), not structural.
