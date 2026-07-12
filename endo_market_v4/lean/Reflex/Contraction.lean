/-
Theory 1.1, formal skeleton: the linearised RRM cobweb.

The market-specific content of 1.1 (that the best-response slope at the fixed
point equals `-epsilon*beta/gamma`, with the three constants given by the GLFT
integrals) is certified numerically in
`reflex/verification/certificates.py::certify_boundary`.  What is formalised
here is the *dynamical* half of the theorem: an affine iterate
`x_{n+1} = a*x_n + b` with fixed point `x*` satisfies
`x_n - x* = a^n (x0 - x*)`, hence converges geometrically iff `|a| < 1` and
its distance to the fixed point blows up when `|a| > 1` (from any
non-degenerate start) -- the "stable iff m < 1" boundary.
-/
import Mathlib.Analysis.SpecificLimits.Basic

namespace Reflex

/-- The linearised cobweb iterate `x_{n+1} = a * x_n + b`. -/
def cobweb (a b x0 : ℝ) : ℕ → ℝ
  | 0 => x0
  | n + 1 => a * cobweb a b x0 n + b

/-- Closed form of the linearised cobweb: `x_n - x* = a^n * (x0 - x*)`. -/
theorem cobweb_closed_form (a b x0 xs : ℝ) (hfix : a * xs + b = xs) :
    ∀ n, cobweb a b x0 n - xs = a ^ n * (x0 - xs) := by
  intro n
  induction n with
  | zero => simp [cobweb]
  | succ n ih =>
    have step : cobweb a b x0 (n + 1) - xs = a * (cobweb a b x0 n - xs) := by
      have : cobweb a b x0 (n + 1) = a * cobweb a b x0 n + b := rfl
      rw [this]
      nlinarith [hfix]
    rw [step, ih, pow_succ]
    ring

/-- **Stability (m < 1).** With `|a| < 1` the cobweb converges to the fixed
point from every start -- the contracting side of the 1.1 boundary. -/
theorem cobweb_converges (a b x0 xs : ℝ) (hfix : a * xs + b = xs)
    (hm : |a| < 1) :
    Filter.Tendsto (cobweb a b x0) Filter.atTop (nhds xs) := by
  have hpow : Filter.Tendsto (fun n => a ^ n * (x0 - xs)) Filter.atTop (nhds 0) := by
    simpa using
      (tendsto_pow_atTop_nhds_zero_of_abs_lt_one hm).mul_const (x0 - xs)
  have hfun : cobweb a b x0 = fun n => xs + a ^ n * (x0 - xs) := by
    funext n
    have h := cobweb_closed_form a b x0 xs hfix n
    linarith
  rw [hfun]
  simpa using tendsto_const_nhds.add hpow

/-- **Instability (m > 1).** With `|a| > 1` and a non-degenerate start the
distance to the fixed point tends to infinity -- the divergent side. -/
theorem cobweb_diverges (a b x0 xs : ℝ) (hfix : a * xs + b = xs)
    (hm : 1 < |a|) (hx : x0 ≠ xs) :
    Filter.Tendsto (fun n => |cobweb a b x0 n - xs|) Filter.atTop Filter.atTop := by
  have habs : (fun n => |cobweb a b x0 n - xs|)
      = fun n => |a| ^ n * |x0 - xs| := by
    funext n
    rw [cobweb_closed_form a b x0 xs hfix n, abs_mul, abs_pow]
  rw [habs]
  have hc : 0 < |x0 - xs| := abs_pos.mpr (sub_ne_zero.mpr hx)
  exact (tendsto_pow_atTop_atTop_of_one_lt hm).atTop_mul_const hc

end Reflex
