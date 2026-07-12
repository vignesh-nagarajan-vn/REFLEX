/-
Theory 1.6, formal skeleton: the K-step (lazy-deploy) outer map.

Everything here is the exact algebra of `reflex/theory/lazy_deploy.py`:
`mu(K) = -m + c^K (1+m)`, its boundary values, the lazy-deploy interpolation
identity, strict monotonicity in `K`, and the stability-window inequality for
an RRM-unstable market (`m > 1`).  The identification of `m` and `c` with the
market's constants is the numerical certificates' job.
-/
import Mathlib.Analysis.SpecificLimits.Basic

namespace Reflex

/-- The K-step deployment-map slope `mu(K) = -m + c^K * (1 + m)` (06 eq. 6.1). -/
def kStepSlope (m c : ℝ) (K : ℕ) : ℝ := -m + c ^ K * (1 + m)

/-- The lazy-deploy mixing weight `lam_K = 1 - c^K` (06 section 2.2). -/
def lazyWeight (c : ℝ) (K : ℕ) : ℝ := 1 - c ^ K

/-- `mu(0) = 1`: zero inner steps leave the deployment unchanged. -/
theorem kStepSlope_zero (m c : ℝ) : kStepSlope m c 0 = 1 := by
  simp [kStepSlope]

/-- **Interpolation (06 corollary 1).** One lazy deployment is the convex
combination `(1 - lam_K) * (do nothing) + lam_K * (exact cobweb)`. -/
theorem kStepSlope_interpolation (m c : ℝ) (K : ℕ) :
    kStepSlope m c K = (1 - lazyWeight c K) * 1 + lazyWeight c K * (-m) := by
  simp only [kStepSlope, lazyWeight]
  ring

/-- Each extra inner step strictly lowers the map slope (`0 < c < 1`, `m ≥ 0`). -/
theorem kStepSlope_succ_lt (m c : ℝ) (hm : 0 ≤ m) (hc0 : 0 < c) (hc1 : c < 1)
    (K : ℕ) : kStepSlope m c (K + 1) < kStepSlope m c K := by
  have hpow : c ^ (K + 1) < c ^ K := by
    have := pow_lt_pow_right_of_lt_one hc0 hc1 (Nat.lt_succ_self K)
    simpa using this
  have hpos : (0 : ℝ) < 1 + m := by linarith
  have := mul_lt_mul_of_pos_right hpow hpos
  simpa [kStepSlope] using add_lt_add_left this (-m)

/-- **Exact-RRM limit.** `mu(K) -> -m` as `K -> infinity` (`|c| < 1`). -/
theorem kStepSlope_tendsto (m c : ℝ) (hc : |c| < 1) :
    Filter.Tendsto (kStepSlope m c) Filter.atTop (nhds (-m)) := by
  have hpow : Filter.Tendsto (fun K => c ^ K * (1 + m)) Filter.atTop (nhds 0) := by
    simpa using
      (tendsto_pow_atTop_nhds_zero_of_abs_lt_one hc).mul_const (1 + m)
  have : kStepSlope m c = fun K => -m + c ^ K * (1 + m) := rfl
  rw [this]
  simpa using tendsto_const_nhds.add hpow

/-- **Stability window (06 eq. 6.3), lower side.** For any `m`, the lazy map
stays above `-1` exactly while `c^K (1+m) > m - 1` -- for `m > 1` this is the
window `K < K_max` inside which laziness stabilises an RRM-unstable market. -/
theorem neg_one_lt_kStepSlope_iff (m c : ℝ) (K : ℕ) :
    -1 < kStepSlope m c K ↔ m - 1 < c ^ K * (1 + m) := by
  constructor <;> (intro h; simp only [kStepSlope] at *; linarith)

/-- **Stability window, upper side.** For `m ≥ 0`, `0 < c < 1` and `K ≥ 1`
the lazy map sits strictly below `+1`: laziness never destabilises upward. -/
theorem kStepSlope_lt_one (m c : ℝ) (hm : 0 ≤ m) (hc0 : 0 < c) (hc1 : c < 1)
    (K : ℕ) (hK : 1 ≤ K) : kStepSlope m c K < 1 := by
  have hpow : c ^ K < 1 := pow_lt_one hc0.le hc1 (by omega)
  have hpos : (0 : ℝ) < 1 + m := by linarith
  have := mul_lt_mul_of_pos_right hpow hpos
  simp only [kStepSlope, one_mul] at *
  linarith

end Reflex
