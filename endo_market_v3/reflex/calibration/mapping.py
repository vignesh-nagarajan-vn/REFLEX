"""Map fitted ``(rating, regime)`` microstructure to a simulator :class:`Config`.

Unit conventions
----------------
The simulator works in **per-$100-par** price units with mid ~ 100 and one step
~ one trading day.  The dataset works in decimal spread fractions.  Conversions
(the only place in the package where units cross):

* half-spread     ``h_100  = h_decimal * 100``
* arrival scale   ``A_100  = A_fit``                     (unit-free count/notional)
* demand decay    ``k_100  = k_decay / 100``             (so ``k*h`` is invariant)
* per-step vol    ``vol_100 = sigma_annual / sqrt(252) * 100``

Data-identified vs. structural
------------------------------
Only ``(A, k, sigma, h)`` are identified by the dataset (and even those are
VIX-proxy fits pending TRACE trade-level access -- see ``data/README.md``).
The **toxic/informed channel is structurally scaled, not data-identified**:
the ratios below tie its magnitude to the identified scales so that
*cross-regime comparisons isolate the data-identified variation*.  Each ratio
matches the corresponding endo_market_v2 default at that package's operating
point (A=1, h~0.8, vol=0.25), so the calibrated market is the same *mechanism*
at data-set scale.  State this plainly in any write-up: the boundary's regime
*ordering* is data-driven; the absolute critical feedback gain is not.

``epsilon`` (``clients.toxicity_feedback``) stays a free sweep knob -- it is
the control variable whose critical value the theory predicts per regime.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Optional, Union
from pathlib import Path

from ..config import Config
from .loader import RegimeMicrostructure, regime_microstructure

# ------------------------- unit constants ---------------------------------- #
PAR = 100.0  # simulator price level (per-$100-par units)
STEPS_PER_YEAR = 252.0  # one simulator step ~ one trading day

# ------------------- structural (non-identified) ratios --------------------- #
# Toxic channel: scales tied to the uninformed arrival scale A (v2 defaults at
# A=1: info_intensity=1.4, info_base_intensity=0.6, info_cap=8).
TOXIC_INTENSITY_RATIO = 1.4  # I   = ratio * A
TOXIC_BASE_RATIO = 0.6  # I_b = ratio * A
INFO_CAP_RATIO = 8.0  # cap = ratio * A
# Toxic spread-responsiveness: keep c_t * h at the v2 operating product
# (1.5 * 0.8 = 1.2) so the feedback channel is *active* at the calibrated
# operating point instead of decayed to zero by a unit change.
CT_TIMES_H = 1.2  # c_t = CT_TIMES_H / h_100
# Informed signal noise: v2 sigma_s=0.6 with per-step vol 0.25 -> 2.4 steps.
SIGNAL_NOISE_STEPS = 2.4  # sigma_s = ratio * vol_100
MISPRICING_RATIO = 2.0  # init_mispricing_vol = ratio * vol_100 (v2: 0.5/0.25)
# Mid dynamics: keep v2's ratios to the per-step vol / arrival scale.
MID_NOISE_RATIO = 0.08  # mid_noise = ratio * vol_100          (v2: 0.02/0.25)
MID_MOVE_CAP_RATIO = 16.0  # mid_move_cap = ratio * vol_100    (v2: 4.0/0.25)
IMPACT_RATIO = 1.2  # impact = ratio * vol_100 / A             (v2: 0.30/(0.25*1))
# Inventory-risk curvature: v2 0.05 at h~0.8, A=1 -> 0.0625 * h / A.
INV_RISK_RATIO = 0.0625  # inv_risk_weight = ratio * h_100 / A
# Policy range: cap the half-spread at a multiple of the calibrated level.
MAX_SPREAD_MULT = 8.0


@dataclass
class CalibrationInfo:
    """Record of what a calibration applied (attached to the run outputs)."""

    rating: str
    regime: str
    micro: RegimeMicrostructure
    h_100: float  # calibrated central half-spread (per-100-par)
    A_100: float
    k_100: float
    vol_100: float  # per-step fundamental vol (per-100-par)
    anchor_weight: float  # quoting-cost convexity w chosen by the stiffness rule
    degenerate: bool  # crisis fits have k pinned to 0 (documented)


def calibrated_config(
    base: Config,
    rating: Optional[str] = None,
    regime: Optional[str] = None,
    anchor_stiffness: Optional[float] = None,
    data_dir: Optional[Union[str, Path]] = None,
) -> "tuple[Config, CalibrationInfo]":
    """Return a copy of ``base`` recalibrated to a ``(rating, regime)`` cell.

    Reads defaults for ``rating``/``regime``/``anchor_stiffness``/``data_dir``
    from ``base.calibration`` when not given.  The quoting-anchor weight is set
    by the stiffness rule ``w = S * A / h0``: it pins the dealer's operating
    point near the *observed* spread level ``h0`` (the anchor stands in for the
    competition/inventory frictions the single-dealer model does not carry --
    with stiffness ``S`` the FOC displacement from ``h0`` is ~``1/(2S)``).
    """
    cal = base.calibration
    rating = cal.rating if rating is None else rating
    regime = cal.regime if regime is None else regime
    S = float(cal.anchor_stiffness if anchor_stiffness is None else anchor_stiffness)
    ddir = data_dir if data_dir is not None else (cal.data_dir or None)

    micro = regime_microstructure(rating, regime, data_dir=ddir)

    h0 = micro.h_mean_decimal * PAR
    A = micro.A
    k = micro.k_decay / PAR
    vol = micro.sigma_annual / math.sqrt(STEPS_PER_YEAR) * PAR
    w = S * A / max(h0, 1e-9)

    clients = replace(
        base.clients,
        base_arrival_rate=A,
        demand_elasticity=k,
        info_intensity=TOXIC_INTENSITY_RATIO * A,
        info_base_intensity=TOXIC_BASE_RATIO * A,
        info_cap=INFO_CAP_RATIO * A,
        info_spread_decay=CT_TIMES_H / max(h0, 1e-9),
        info_signal_noise=SIGNAL_NOISE_STEPS * vol,
        spread_ref=h0,
    )
    simulator = replace(
        base.simulator,
        fundamental_vol=vol,
        init_mispricing_vol=MISPRICING_RATIO * vol,
        mid_noise=MID_NOISE_RATIO * vol,
        mid_move_cap=MID_MOVE_CAP_RATIO * vol,
        impact=IMPACT_RATIO * vol / max(A, 1e-9),
    )
    policy = replace(
        base.policy,
        init_half_spread=h0,
        max_half_spread=MAX_SPREAD_MULT * h0,
    )
    reward = replace(
        base.reward,
        quote_anchor_ref=h0,
        quote_anchor_weight=w,
        inv_risk_weight=INV_RISK_RATIO * h0 / max(A, 1e-9),
    )
    cfg = replace(
        base,
        clients=clients,
        simulator=simulator,
        policy=policy,
        reward=reward,
        calibration=replace(cal, enabled=True, rating=rating, regime=regime),
    )
    info = CalibrationInfo(
        rating=rating,
        regime=regime,
        micro=micro,
        h_100=h0,
        A_100=A,
        k_100=k,
        vol_100=vol,
        anchor_weight=w,
        degenerate=micro.degenerate,
    )
    return cfg, info


def apply_calibration(cfg: Config) -> "tuple[Config, Optional[CalibrationInfo]]":
    """Idempotent entry-point hook: recalibrate iff ``cfg.calibration.enabled``."""
    if not cfg.calibration.enabled:
        return cfg, None
    return calibrated_config(cfg)
