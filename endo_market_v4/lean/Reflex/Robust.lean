/-
Theory 1.4, formal skeleton: soundness of the robust stability certificate.

The statistical content of 1.4 (the O(1/sqrt(n)) radius bought by the CRN
probe, the coverage of z*s) is certified numerically.  What is formalised is
the certificate's *logical core* (Theorem 1 of 1.4 §4): whenever the
estimation error is inside the ambiguity radius, the certificate's verdicts
are sound -- declaring "stable" (resp. "unstable") only when the truth is.
-/
import Mathlib.Data.Real.Basic
import Mathlib.Tactic.Linarith

namespace Reflex

/-- **Stable-side soundness.** If `|mhat - m| ≤ delta` and the whole ball is
below the boundary (`mhat + delta < 1`), the market is truly stable. -/
theorem certificate_sound_stable (m mhat delta : ℝ)
    (herr : |mhat - m| ≤ delta) (hcert : mhat + delta < 1) : m < 1 := by
  have h := abs_le.mp herr
  linarith [h.1, h.2]

/-- **Unstable-side soundness.** If `|mhat - m| ≤ delta` and the whole ball is
above the boundary (`1 < mhat - delta`), the market is truly unstable. -/
theorem certificate_sound_unstable (m mhat delta : ℝ)
    (herr : |mhat - m| ≤ delta) (hcert : 1 < mhat - delta) : 1 < m := by
  have h := abs_le.mp herr
  linarith [h.1, h.2]

/-- **The undecided band is honest.** Near the boundary -- specifically when
`|mhat - 1| ≤ delta` -- both a stable and an unstable truth are consistent
with the observation, so no sound certificate can decide: there exist
`m_lo < 1` and `1 < m_hi` both within `delta` of `mhat` (radius `delta > 0`). -/
theorem undecided_band_nonempty (mhat delta : ℝ) (hd : 0 < delta)
    (hband : |mhat - 1| ≤ delta) :
    ∃ mlo mhi : ℝ, |mhat - mlo| ≤ delta ∧ mlo < 1 ∧
      |mhat - mhi| ≤ delta ∧ 1 < mhi := by
  have h := abs_le.mp hband
  refine ⟨mhat - delta, mhat + delta, ?_, by linarith [h.1, h.2], ?_,
    by linarith [h.1, h.2]⟩
  · simp [abs_of_nonneg hd.le]
  · simp [abs_of_nonpos (by linarith : mhat - (mhat + delta) ≤ 0)]

/-- **Radius monotonicity.** A certificate that fires at radius `delta'` also
fires at any tighter error bound `delta ≤ delta'` -- shrinking the ambiguity
ball (more data, 1.4 §4) only strengthens the verdict. -/
theorem certificate_mono (m mhat delta delta' : ℝ) (hdd : delta ≤ delta')
    (herr : |mhat - m| ≤ delta) (hcert : mhat + delta' < 1) : m < 1 :=
  certificate_sound_stable m mhat delta herr (by linarith)

end Reflex
