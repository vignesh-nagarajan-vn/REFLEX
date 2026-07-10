# sweep - the predict-then-verify phase diagram (+ alpha appendix)

CLEAN-PROTOCOL RERUN (collection_jitter = 0.05): the first execution used the
inherited jitter 0.2, which inflated every probe reading ~3x (m ~ 0.66 at
zero feedback where structure says 0) - audit section 1.6. These artifacts
supersede it.

Feedback sweep (7 gains x 8 seeds, h_ref = 1.0): measured medians track the
closed form at the probe spread through the contracting regime (0.067 / 0.159
/ 0.390 vs 0 / 0.213 / 0.426 at f = 0/1/2); measured crossing f* ~ 3.17 vs
a-priori predicted 4.70 - the gap matches the realized-state (liquidity
inflation) correction measured by the triangulation, which moves the
prediction to ~2.8-3.0. Robust certificates: stable through f = 2, unstable
at f = 4, undecided in the scattered beyond-boundary regime. IQRs widen
sharply past the boundary (the probe stops being a local slope there - the
theory's A4 caveat, visible in data).

Alpha appendix (8 points x 5 seeds, f = 5): the confound's full non-monotone
hump - medians 0.08 -> 1.83 (alpha = 0.45) -> 0.67 (alpha = 0.8): the
feedback-slope channel rises, then defensive widening reverses it. The
quantitative case for toxicity_feedback as the headline control variable.

Analysis: ANALYSIS-full-2026-07.md sections 4 and 9.
