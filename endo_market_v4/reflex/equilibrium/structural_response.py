"""Structurally-anchored learned market response (v4: closing the loop-level gap).

The v3 negative result (documented in ``research/analysis/ANALYSIS-full-2026-07.md``
section 8): the free-form learned loop (``perfgd_learned``) does not reproduce
the closed-form PerfGD stabilisation, because the *free-form* operator's implied
objective gradient ``dJ/dh`` -- an autograd readout of an MLP fit on one
regime's data -- diverges from the structural one as soon as the candidate
policy leaves the deployed neighbourhood.  The diagnosed fix (the v3 to-do's
stretch goal, realised here): **anchor the learned response to the GLFT
structural form**.  Instead of asking an MLP for ``dD/dphi``, fit the theory's
own parametric families to the loop's deployment data,

    tau_hat(h)  = C0 + C1 * exp(-c * h)          (informed / toxic flow, 1.1 §2.3)
    u_hat(h)    = A_u * exp(-k_u * h)            (uninformed GLFT fill curve, 1.1 §2.2)
    psi_hat     = sum(adverse_loss) / sum(informed_volume)   (adverse severity, 1.1 §2.5)

and assemble *estimated* counterparts of every 1.1/1.2 closed form -- the blind
gradient ``G_hat``, the correction ``Delta_hat = -beta*(h - psi_hat)*eps_hat(h)``,
the corrected ascent ``Phi_hat' = G_hat + Delta_hat``, the curvatures
``gamma_hat`` / ``gamma_PO_hat`` and the estimated optimum ``h_PO_hat``.  No
closed-form *market* constant is consumed: everything about the market is
measured from the window's transitions.  (``beta = pnl_scale`` and the quoting
anchor ``w, h_ref`` are the dealer's *own* objective bookkeeping -- reading
them from the config is not oracle access, cf. 1.2 §2.)

What the fits measure (state it plainly in any write-up).  The fitted response
is the **realized** market's: it includes the ``info_cap`` saturation (which
binds at tight spreads in high-intensity regimes -- e.g. the beyond-boundary
demo regime, where raw toxic notionals ~10 exceed the cap 8), the
liquidity-inflation channel (tight quoting boosts the liquidity field, which
multiplies all flows -- realized ``rho`` up to several times the A2 reference
``1.0``), and the realized severity drift.  The frozen-reference (A2) closed
forms of 1.1/1.2 deliberately omit all three (1.1 §9), so the structural
loop's optimum is the realized ``h_PO`` -- benchmark it against an independent
structural fit, not against the A2 number (see
``tests/test_structural_perfgd.py``).

Bias compensation.  Three estimation defects the v3 seam diagnostics identified
are compensated structurally:

* **regime anchoring** -- the MLP's slope readout was only valid at the
  deployed summary; the parametric family extrapolates correctly by
  construction (it is the true functional form of the flow model);
* **within-deployment identification** -- the fit consumes *per-step*
  ``(h, flow)`` pairs, so the collection jitter (``rrm.collection_jitter``)
  provides spread variation even after the outer loop has converged (the
  echo-chamber's identification collapse); and
* **degenerate-window freezing** -- when the realised spread range is still too
  narrow to identify the curve (:attr:`StructuralResponse.identified` False),
  the caller keeps the last identifiable fit instead of consuming noise (the
  freeze is the stability-penalty companion: never steer on an unidentified
  response).

The consuming loop mode is ``run_loop(mode="perfgd_structural")`` in
:mod:`reflex.equilibrium.loops`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from ..config import Config, RewardConfig
from .fit_operator import DeploymentRecord


@dataclass
class StructuralResponse:
    """The GLFT-anchored response fitted from a window of deployments."""

    # informed (toxic) flow curve tau_hat(h) = C0 + C1*exp(-c*h)
    C0: float
    C1: float
    c: float
    # uninformed GLFT fill curve u_hat(h) = A_u * exp(-k_u * h)
    A_u: float
    k_u: float
    # realized adverse severity per unit toxic notional
    psi_hat: float
    # identifiability record
    n_points: int
    h_lo: float
    h_hi: float
    identified: bool  # enough spread variation to trust the fitted slope
    tau_fit_ok: bool  # exponential fit converged (else finite-difference C1*c)
    residual_rms: float

    # ------------------------------------------------------------------ #
    # Fitted counterparts of the 1.1 closed forms                         #
    # ------------------------------------------------------------------ #
    def tau_hat(self, h: float) -> float:
        """Estimated toxic level ``C0 + C1*exp(-c*h)`` (fitted 1.1 §2.3)."""
        return self.C0 + self.C1 * math.exp(-self.c * float(h))

    def epsilon_hat(self, h: float) -> float:
        """Estimated distribution sensitivity ``|d tau_hat/dh| = c*C1*exp(-c*h)``."""
        return self.c * self.C1 * math.exp(-self.c * float(h))

    def uninformed_hat(self, h: float) -> float:
        """Estimated uninformed notional ``A_u*exp(-k_u*h)`` (fitted GLFT curve)."""
        return self.A_u * math.exp(-self.k_u * float(h))

    def toxic_slope_hat(self, h: float) -> float:
        """Estimated adverse-channel slope ``d(psi*tau)/dh = -psi_hat*eps_hat(h)``.

        Directly comparable to the operator's learned ``toxic_slope`` and the
        theory's ``-psi*epsilon(h)`` -- the third leg of the seam diagnostics.
        """
        return -self.psi_hat * self.epsilon_hat(h)

    # ------------------------------------------------------------------ #
    # Fitted counterparts of the 1.2 objective machinery                  #
    # ------------------------------------------------------------------ #
    def blind_gradient_hat(self, h: float, reward: RewardConfig) -> float:
        """Estimated blind gradient ``G_hat(h)`` (fitted 1.1 §3.2 form).

        ``G_hat(h) = P * [ u_hat(h)*(1 - k_u*h) + tau_hat(h) - 2*w*(h - h_ref) ]``
        -- the dealer's own first-order condition with the *measured* fill and
        toxic curves in place of the closed forms.
        """
        P = float(reward.pnl_scale)
        w = float(reward.quote_anchor_weight)
        h_ref = float(reward.quote_anchor_ref)
        h = float(h)
        return P * (
            self.uninformed_hat(h) * (1.0 - self.k_u * h)
            + self.tau_hat(h)
            - 2.0 * w * (h - h_ref)
        )

    def correction_hat(self, h: float, reward: RewardConfig) -> float:
        """Estimated PerfGD correction ``-beta*(h - psi_hat)*eps_hat(h)`` (1.2 §2)."""
        beta = float(reward.pnl_scale)
        return -beta * (float(h) - self.psi_hat) * self.epsilon_hat(h)

    def corrected_gradient_hat(self, h: float, reward: RewardConfig) -> float:
        """Estimated corrected ascent ``Phi_hat'(h) = G_hat(h) + Delta_hat(h)``."""
        return self.blind_gradient_hat(h, reward) + self.correction_hat(h, reward)

    def gamma_hat(self, h: float, reward: RewardConfig, lambda_q: Optional[float] = None) -> float:
        """Estimated strong convexity (fitted 1.1 §3.2 (*))."""
        P = float(reward.pnl_scale)
        w = float(reward.quote_anchor_weight)
        lq = float(reward.inv_risk_weight if lambda_q is None else lambda_q)
        h = float(h)
        glft = self.A_u * self.k_u * math.exp(-self.k_u * h) * (2.0 - self.k_u * h)
        return P * (2.0 * w + glft + lq)

    def gamma_po_hat(self, h: float, reward: RewardConfig, lambda_q: Optional[float] = None) -> float:
        """Estimated objective curvature at the optimum (fitted 1.2 §4.1)."""
        beta = float(reward.pnl_scale)
        return self.gamma_hat(h, reward, lambda_q=lambda_q) + beta * self.epsilon_hat(h) * (
            2.0 + self.c * self.psi_hat - self.c * float(h)
        )

    def solve_h_po_hat(self, reward: RewardConfig, h_max: float) -> float:
        """Estimated performative optimum: root of ``Phi_hat'`` on ``[0, h_max]``.

        This is the optimum of the *realized* market the fits measure --
        including the ``info_cap`` saturation and the liquidity/state feedback
        that the frozen-reference (A2) closed forms deliberately omit (1.1
        §9), so in strongly-fed-back regimes it sits away from the A2
        ``h_PO``.  The apples-to-apples benchmark for a loop that settles
        here is an *independent* structural fit, not the A2 number.
        """
        return _root_on_grid(
            lambda h: self.corrected_gradient_hat(h, reward), 1e-3, float(h_max)
        )

    def solve_h_sp_hat(self, reward: RewardConfig, h_max: float) -> float:
        """Estimated *stable point*: root of the blind gradient ``G_hat``.

        The realized-market counterpart of ``h_SP`` (1.2 §1) -- where a loop
        that ignores its own distribution response parks.  The estimated
        echo-chamber decision gap is ``solve_h_sp_hat() - solve_h_po_hat()``.
        """
        return _root_on_grid(
            lambda h: self.blind_gradient_hat(h, reward), 1e-3, float(h_max)
        )


def _root_on_grid(fn, lo: float, hi: float, n_grid: int = 256) -> float:
    """First sign-change root of ``fn`` on ``[lo, hi]`` (grid scan + bisection).

    Falls back to the grid point of least ``|fn|`` when no sign change exists
    (the root sits at the boundary of the operating range).
    """
    xs = np.linspace(float(lo), float(hi), int(n_grid))
    vals = np.array([fn(float(x)) for x in xs])
    sign_change = np.where(np.sign(vals[:-1]) * np.sign(vals[1:]) < 0)[0]
    if not sign_change.size:
        return float(xs[int(np.argmin(np.abs(vals)))])
    i = int(sign_change[0])
    a, b = float(xs[i]), float(xs[i + 1])
    fa = fn(a)
    for _ in range(80):
        mid = 0.5 * (a + b)
        fm = fn(mid)
        if fa * fm <= 0.0:
            b = mid
        else:
            a, fa = mid, fm
    return 0.5 * (a + b)


def _exp_curve_fit(
    h: np.ndarray, y: np.ndarray
) -> "tuple[float, float, float, bool, float]":
    """Fit ``y = C0 + C1*exp(-c*h)`` by nonlinear least squares.

    Returns ``(C0, C1, c, ok, residual_rms)``.  Mirrors the CKS estimator's fit
    (the same functional family) with bounds keeping the curve in the
    decreasing-response regime; falls back to a linearised slope
    representation when the optimiser fails.
    """
    from scipy.optimize import curve_fit

    h = np.asarray(h, dtype=float)
    y = np.asarray(y, dtype=float)
    span = float(y.max() - y.min())
    p0 = (max(float(y.min()), 1e-6), max(span, 1e-6), 1.0)
    try:
        popt, _ = curve_fit(
            lambda x, c0, c1, cc: c0 + c1 * np.exp(-cc * x),
            h, y, p0=p0,
            bounds=([0.0, 0.0, 0.02], [np.inf, np.inf, 20.0]),
            maxfev=20000,
        )
        c0, c1, cc = (float(v) for v in popt)
        resid = y - (c0 + c1 * np.exp(-cc * h))
        return c0, c1, cc, True, float(np.sqrt(np.mean(resid ** 2)))
    except Exception:
        # Finite-difference fallback: represent the local line y = a + b*h as
        # a shallow exponential (c -> small) with matching value and slope.
        A = np.vstack([h, np.ones_like(h)]).T
        slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
        cc = 0.05
        h_bar = float(h.mean())
        c1 = max(-float(slope), 0.0) / (cc * math.exp(-cc * h_bar))
        c0 = max(float(intercept + slope * h_bar) - c1 * math.exp(-cc * h_bar), 0.0)
        resid = y - (c0 + c1 * np.exp(-cc * h))
        return c0, c1, cc, False, float(np.sqrt(np.mean(resid ** 2)))


def fit_structural_response(
    deployments: Sequence[DeploymentRecord],
    cfg: Config,
    min_rel_range: Optional[float] = None,
) -> StructuralResponse:
    """Fit the GLFT-anchored response from a history of deployments.

    Consumes *per-bond, per-step* samples from every transition: the realised
    half-spread (which varies within a deployment through the per-bond
    collection jitter -- the identification the echo-chamber cannot destroy),
    the per-bond informed notional (the ``tau`` samples), and the per-bond
    uninformed notional (the GLFT fill-curve samples).  ``psi_hat`` is the
    volume-weighted realized severity ``sum(adverse_loss)/sum(informed_volume)``.

    Callers should pass a *long* history (``rrm.structural_window``
    deployments), not just the operator's short context window: exponential
    families fitted on a narrow spread range are line-degenerate and
    extrapolate arbitrarily badly, whereas the loop's own traversed trajectory
    covers the informative range (its transient is the exploration -- the
    deployment-history estimation of Izzo et al.'s PerfGD, made parametric).

    ``min_rel_range`` (default ``cfg.rrm.structural_min_rel_range``) sets the
    identifiability floor: the fit is flagged unidentified when the realised
    spread range is below ``min_rel_range * mean(h)``, and callers must then
    hold the previous fit rather than steer on noise.
    """
    if min_rel_range is None:
        min_rel_range = float(cfg.rrm.structural_min_rel_range)

    hs: list = []
    taus: list = []
    uninf: list = []
    adverse_sum = 0.0
    informed_sum = 0.0
    for dep in deployments:
        for tr in dep.transitions:
            # Per-bond samples: the collection jitter is *per bond*, so the
            # per-bond pairs carry twice the spread variation of the
            # cross-bond mean (and n_bonds times the count) -- and tau(h) is a
            # per-bond object in the theory (1.1 A1), so the units match.
            h_b = tr.quotes.half_spread.detach().cpu().numpy().astype(float)
            inf_b = tr.fills.informed_volume.detach().cpu().numpy().astype(float)
            gross_b = tr.fills.gross_volume.detach().cpu().numpy().astype(float)
            hs.append(h_b)
            taus.append(inf_b)
            uninf.append(np.clip(gross_b - inf_b, 0.0, None))
            adverse_sum += float(tr.pnl_components["adverse_selection_loss"].sum())
            informed_sum += float(tr.fills.informed_volume.sum())

    h_arr = np.concatenate(hs) if hs else np.empty(0)
    tau_arr = np.concatenate(taus) if taus else np.empty(0)
    u_arr = np.concatenate(uninf) if uninf else np.empty(0)
    n = int(h_arr.size)
    if n < 8:
        return StructuralResponse(
            C0=0.0, C1=0.0, c=1.0, A_u=0.0, k_u=1.0, psi_hat=0.0,
            n_points=n, h_lo=float("nan"), h_hi=float("nan"),
            identified=False, tau_fit_ok=False, residual_rms=float("nan"),
        )

    h_lo, h_hi = float(h_arr.min()), float(h_arr.max())
    h_bar = float(h_arr.mean())
    identified = (h_hi - h_lo) >= float(min_rel_range) * max(h_bar, 1e-9)

    # Toxic curve: the theory's own exponential family (1.1 §2.3).
    C0, C1, c, tau_ok, resid = _exp_curve_fit(h_arr, tau_arr)

    # Uninformed GLFT curve: log-linear regression (exact for the model family).
    logu = np.log(np.clip(u_arr, 1e-9, None))
    A = np.vstack([h_arr, np.ones_like(h_arr)]).T
    slope, intercept = np.linalg.lstsq(A, logu, rcond=None)[0]
    k_u = max(-float(slope), 1e-6)
    A_u = float(np.exp(intercept))

    psi_hat = adverse_sum / informed_sum if informed_sum > 1e-9 else 0.0

    return StructuralResponse(
        C0=C0, C1=C1, c=c, A_u=A_u, k_u=k_u, psi_hat=float(psi_hat),
        n_points=n, h_lo=h_lo, h_hi=h_hi,
        identified=bool(identified), tau_fit_ok=bool(tau_ok),
        residual_rms=resid,
    )


def retune_central_spread(policy, ref_state, h_target: float, tol: float = 1e-5) -> float:
    """Shift the policy's half-spread bias so its central spread hits ``h_target``.

    The structural mode's 1-D update acts on the central half-spread (the
    dominant coordinate of the iterate map, 1.1 §1); this realises the updated
    spread in the deployed policy by bisection on the scalar bias shared by
    both policy classes (``LinearPolicy.b_h`` / the MLP head's half-spread
    output bias).  Monotone in the bias by construction (softplus), so
    bisection is exact.  Returns the achieved central spread.
    """
    import torch

    from ..policy.dealer_policy import LinearPolicy, MLPPolicy

    if isinstance(policy, LinearPolicy):
        bias = policy.b_h
        idx = None
    elif isinstance(policy, MLPPolicy):
        bias = policy.net[-1].bias
        idx = 0
    else:  # pragma: no cover - future policy types must opt in explicitly
        raise TypeError(f"retune_central_spread: unsupported policy {type(policy).__name__}")

    def _central(shift: float) -> float:
        with torch.no_grad():
            if idx is None:
                bias.add_(shift)
            else:
                bias[idx] += shift
            h = float(policy.quote(ref_state).half_spread.mean())
            if idx is None:
                bias.add_(-shift)
            else:
                bias[idx] -= shift
        return h

    h_target = float(h_target)
    lo, hi = -20.0, 20.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if _central(mid) < h_target:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol * 1e-2:
            break
    shift = 0.5 * (lo + hi)
    with torch.no_grad():
        if idx is None:
            bias.add_(shift)
        else:
            bias[idx] += shift
        return float(policy.quote(ref_state).half_spread.mean())
