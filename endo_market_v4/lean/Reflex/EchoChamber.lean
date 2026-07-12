/-
Theory 1.2, formal skeleton: the echo-chamber separation.

The market-specific content of 1.2 (the closed-form correction, gamma_PO, the
O(eps)/O(eps^2) gap sizes) is certified numerically.  Formalised here is the
*separation logic*: the stable point (root of the blind gradient `G`) is not
performatively optimal whenever the correction `Delta` is nonzero there, and
under strict concavity of the corrected objective the optimum sits strictly
on the side the correction points to.  In the market's operating regime
`h_SP > psi` gives `Delta(h_SP) < 0` (summoned flow is net profitable, worth
`h` against adverse cost `psi`), so `h_PO < h_SP`: the blind dealer
over-defends -- the echo-chamber direction of 1.2 §6 / §7.1.
-/
import Mathlib.Order.Monotone.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Tactic.Linarith

namespace Reflex

/-- **The stable point is not a stationary point of the true objective.**
If `G h_sp = 0` (RRM fixed point) and the performative correction does not
vanish there, the corrected gradient `Phi' = G + Delta` is nonzero at `h_sp`. -/
theorem stable_point_not_stationary (G Delta : ℝ → ℝ) (hsp : ℝ)
    (hG : G hsp = 0) (hD : Delta hsp ≠ 0) : G hsp + Delta hsp ≠ 0 := by
  simpa [hG] using hD

/-- Hence the stable point and the performative optimum are distinct. -/
theorem sp_ne_po (Phi' : ℝ → ℝ) (hsp hpo : ℝ)
    (hSP : Phi' hsp ≠ 0) (hPO : Phi' hpo = 0) : hsp ≠ hpo :=
  fun h => hSP (h ▸ hPO)

/-- **Gap direction.** For a strictly concave corrected objective (its
gradient `Phi'` strictly antitone), a negative correction at the stable point
(`Phi' h_sp = Delta h_sp < 0`) places the optimum strictly *tighter*:
`h_po < h_sp` -- the blind dealer over-defends (1.2 §6/§7.1). -/
theorem po_lt_sp_of_neg_correction (Phi' : ℝ → ℝ) (hanti : StrictAnti Phi')
    (hsp hpo : ℝ) (hPO : Phi' hpo = 0) (hneg : Phi' hsp < 0) : hpo < hsp := by
  by_contra hle
  push_neg at hle
  rcases eq_or_lt_of_le hle with heq | hlt
  · rw [← heq] at hPO
    linarith
  · have h := hanti hlt
    rw [hPO] at h
    linarith

/-- Symmetric direction: a positive correction at the stable point places the
optimum strictly wider. -/
theorem sp_lt_po_of_pos_correction (Phi' : ℝ → ℝ) (hanti : StrictAnti Phi')
    (hsp hpo : ℝ) (hPO : Phi' hpo = 0) (hpos : 0 < Phi' hsp) : hsp < hpo := by
  by_contra hle
  push_neg at hle
  rcases eq_or_lt_of_le hle with heq | hlt
  · rw [heq] at hPO
    linarith
  · have h := hanti hlt
    rw [hPO] at h
    linarith

end Reflex
