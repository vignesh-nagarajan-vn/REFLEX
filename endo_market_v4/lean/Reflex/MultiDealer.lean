/-
Theory 1.3, formal skeleton: the common-mode eigen-identity and the N_eff
boundary algebra.

The joint linearised Jacobian of the N-dealer cobweb with shared-pool
spillover `kappa` has diagonal `-m1` and off-diagonal `-m1 * kappa`
(derivation 1.3 §3; the sum-coupling convention of A2-multi).  Formalised
here: the all-ones vector is an eigenvector with eigenvalue
`-m1 * (1 + kappa * (N - 1)) = -m1 * N_eff` -- competition amplifies the
single-dealer modulus by exactly `N_eff` along the common mode -- and the
boundary-restatement algebra `N_eff * (eps*beta/gamma) < 1  <->
eps < gamma / (N_eff*beta)`.  The identification of `m1` with the market is
certified numerically.
-/
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Ring
import Mathlib.Tactic.Linarith

namespace Reflex

open Matrix BigOperators

/-- The linearised N-dealer joint Jacobian: `-m1` on the diagonal, `-m1*kappa`
off it (theory 1.3 §3). -/
def jointJacobian (m1 kappa : ℝ) (N : ℕ) : Matrix (Fin N) (Fin N) ℝ :=
  Matrix.of fun i j => if i = j then -m1 else -m1 * kappa

/-- **Common-mode eigen-identity.** The all-ones vector is an eigenvector of
the joint Jacobian with eigenvalue `-m1 * (1 + kappa*(N-1)) = -m1 * N_eff`. -/
theorem common_mode_eigen (m1 kappa : ℝ) (N : ℕ) :
    (jointJacobian m1 kappa N).mulVec (fun _ => (1 : ℝ)) =
      fun _ => -m1 * (1 + kappa * ((N : ℝ) - 1)) := by
  classical
  funext i
  have hsplit : ∀ j : Fin N,
      (if i = j then -m1 else -m1 * kappa)
        = -m1 * kappa + (if i = j then -m1 + m1 * kappa else 0) := by
    intro j
    by_cases h : i = j <;> simp [h] <;> ring
  simp only [jointJacobian, Matrix.mulVec, Matrix.dotProduct, Matrix.of_apply,
    mul_one]
  calc
    (∑ j : Fin N, if i = j then -m1 else -m1 * kappa)
        = ∑ j : Fin N,
            (-m1 * kappa + (if i = j then -m1 + m1 * kappa else 0)) := by
          exact Finset.sum_congr rfl fun j _ => hsplit j
    _ = (N : ℝ) * (-m1 * kappa) + (-m1 + m1 * kappa) := by
          rw [Finset.sum_add_distrib, Finset.sum_const, Finset.card_univ,
            Fintype.card_fin, Finset.sum_ite_eq, if_pos (Finset.mem_univ i)]
          push_cast
          ring
    _ = -m1 * (1 + kappa * ((N : ℝ) - 1)) := by ring

/-- **Boundary restatement (1.3 §4).** With `m1 = eps*beta/gamma`, the
common-mode stability condition `N_eff * m1 < 1` is exactly the PSNE boundary
`eps < gamma / (N_eff * beta)`. -/
theorem boundary_iff (eps beta gamma neff : ℝ)
    (hg : 0 < gamma) (hb : 0 < beta) (hn : 0 < neff) :
    neff * (eps * beta / gamma) < 1 ↔ eps < gamma / (neff * beta) := by
  rw [div_lt_div_iff (mul_pos hn hb), mul_div_assoc, div_lt_one hg] at *
  constructor <;> intro h <;> nlinarith

/-- `N_eff` is monotone in the spillover: more coupling, more amplification. -/
theorem n_eff_mono (N : ℕ) (hN : 1 ≤ N) {k1 k2 : ℝ} (hk : k1 ≤ k2) :
    1 + k1 * ((N : ℝ) - 1) ≤ 1 + k2 * ((N : ℝ) - 1) := by
  have hNr : (0 : ℝ) ≤ (N : ℝ) - 1 := by
    have : (1 : ℝ) ≤ (N : ℝ) := by exact_mod_cast hN
    linarith
  nlinarith

end Reflex
