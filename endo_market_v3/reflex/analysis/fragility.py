"""The REFLEX market-fragility index: the analytic boundary on real data.

For every trading day 1990-2026 in the shipped master panel, evaluate the
closed-form performative-stability quantities of theory 1.1 at that day's
data-identified microstructure:

* ``gamma(t) = 2*w + A(t)*k(t)*exp(-k*h(t))*(2 - k*h(t))`` -- the strong
  convexity (GLFT fill-curve curvature + a fixed anchor floor);
* ``eps_star(t) = gamma(t)/beta`` -- the **stability headroom**: the largest
  performative-feedback sensitivity the market can absorb before repeated
  retraining destabilises (the a-priori boundary of 1.1, per day);
* ``epsilon(t) = c_t * C1(t) * exp(-c_t*h(t))`` with a data-driven toxic pool
  ``C1(t) = tox_ratio * s(t) * A(t)`` where ``s(t)`` is the informed share
  proxy ``|credit_rf| / (|credit_rf| + |liquidity_rf|)`` (adverse-selection vs.
  pure-illiquidity signal decomposition, as in the preprocessing pipeline);
* ``m(t) = epsilon(t)*beta/gamma(t)`` -- the modulus at the observed operating
  point; and
* ``fragility(t) = median(eps_star) / eps_star(t)`` -- the headline index
  (relative tightening of the boundary; > 1 = less headroom than typical).

Inputs per day: ``A`` from the master's VIX-mapped arrival estimate, ``k`` from
the per-``(rating, regime)`` exponential-intensity fit (merged by regime; the
crisis fit is degenerate with ``k = 0``, so crisis days sit on the anchor floor
-- exactly the "curvature collapse" fragility mechanism), ``sigma`` from the
per-day vol estimate, ``h`` from the VIX-implied spread rescaled so its
per-regime mean matches the fitted ``h_mean_decimal`` for the chosen rating.

Everything is a closed form (no simulation), so the full 9,218-day index
computes in well under a second.  Structural constants (the anchor floor
fraction, ``c_t*h`` product, toxic ratio) are module-level and documented; the
*variation* of the index is entirely data-driven.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

from ..calibration.loader import load_intensity_params, load_master
from ..calibration.regimes import REGIME_ORDER

# Structural constants (documented; see module docstring).  These fix the
# *level* of the index; its time variation comes from the data.
CT_TIMES_H = 1.2  # c_t = CT_TIMES_H / mean(h): toxic decay active at the mean spread
TOX_RATIO = 1.0  # toxic pool scale: C1(t) = TOX_RATIO * s(t) * A(t)
GAMMA_FLOOR_FRAC = 0.5  # anchor floor 2w = frac * mean GLFT curvature
BETA = 1.0  # P&L scale (cancels in ratios; kept explicit)


@dataclass
class FragilityResult:
    """The daily fragility panel plus its per-regime summary."""

    daily: pd.DataFrame  # one row per trading day (see compute_fragility)
    by_regime: pd.DataFrame  # per-regime medians of the key columns
    rating: str
    c_t: float  # toxic decay used (per decimal-h)
    anchor_floor: float  # the 2w gamma floor used


def compute_fragility(
    rating: str = "IG",
    master: Optional[pd.DataFrame] = None,
    data_dir: Optional[Union[str, Path]] = None,
    ct_times_h: float = CT_TIMES_H,
    tox_ratio: float = TOX_RATIO,
    gamma_floor_frac: float = GAMMA_FLOOR_FRAC,
) -> FragilityResult:
    """Compute the daily fragility index for a rating bucket.

    Parameters
    ----------
    rating:
        ``"IG"`` or ``"HY"`` -- selects the per-day ``A``/``sigma`` estimates
        and the per-regime ``(k, h)`` fits.
    master:
        Optional pre-loaded master panel (loaded from ``data_dir`` if ``None``).
    data_dir:
        Override for the shipped data directory.
    ct_times_h, tox_ratio, gamma_floor_frac:
        Structural constants (see module docstring); exposed for sensitivity
        analysis, defaults are the documented paper values.
    """
    if rating not in ("IG", "HY"):
        raise ValueError(f"unknown rating {rating!r} (expected 'IG' or 'HY')")
    df = load_master(data_dir) if master is None else master.copy()
    fits = load_intensity_params(data_dir).loc[rating]  # indexed by regime

    out = pd.DataFrame({"date": df["date"], "regime": df["regime"], "vix_close": df["vix_close"]})

    # --- per-day data-identified inputs ------------------------------------ #
    A = df[f"A_{rating}_estimate"].astype(float).to_numpy()
    sigma = df[f"sigma_{rating}_estimate"].astype(float).to_numpy()
    # k merged by regime from the fitted table (crisis: k = 0, degenerate).
    k = df["regime"].map(fits["k_decay_fit"]).astype(float).to_numpy()
    # h: VIX-implied spread, rescaled so each regime's mean matches the fitted
    # h_mean_decimal of this rating bucket (preserves within-regime variation).
    h_implied = (df["implied_spread_bps"].astype(float) / 1e4)
    regime_mean = h_implied.groupby(df["regime"]).transform("mean")
    target = df["regime"].map(fits["h_mean_decimal"]).astype(float)
    h = (h_implied * target / regime_mean).to_numpy()
    # Informed-share proxy s(t); pre-2004 rows lack bond factors -> fill with
    # the post-2004 mean (macro context only, flagged via `informed_imputed`).
    credit = df["credit_rf"].astype(float).abs()
    liq = df["liquidity_rf"].astype(float).abs()
    s_raw = credit / (credit + liq + 1e-12)
    informed_imputed = s_raw.isna()
    s = s_raw.fillna(s_raw.mean()).to_numpy()

    # --- closed forms (theory 1.1, rho = 1, lambda_q = 0) ------------------- #
    kh = k * h
    glft = A * k * np.exp(-kh) * (2.0 - kh)  # GLFT fill-curve curvature term
    anchor_floor = gamma_floor_frac * float(np.nanmean(glft))  # the 2w floor
    gamma = anchor_floor + glft
    gamma = np.maximum(gamma, 1e-12)  # numerical guard (never binds in-sample)

    h_bar = float(np.nanmean(h))
    c_t = ct_times_h / h_bar
    c1 = tox_ratio * s * A
    epsilon = c_t * c1 * np.exp(-c_t * h)

    eps_star = gamma / BETA  # stability headroom (a-priori boundary)
    modulus = epsilon * BETA / gamma  # modulus at the observed operating point
    fragility = float(np.nanmedian(eps_star)) / eps_star

    out["h_decimal"] = h
    out["A"] = A
    out["k_decay"] = k
    out["sigma_annual"] = sigma
    out["informed_share"] = s
    out["informed_imputed"] = informed_imputed.to_numpy()
    out["gamma"] = gamma
    out["epsilon_at_h"] = epsilon
    out["modulus_at_h"] = modulus
    out["eps_star"] = eps_star
    out["fragility"] = fragility
    out["stable_at_h"] = modulus < 1.0
    out["regime_order"] = out["regime"].map(REGIME_ORDER)

    by_regime = (
        out.groupby("regime")[["h_decimal", "A", "gamma", "epsilon_at_h", "modulus_at_h", "eps_star", "fragility"]]
        .median()
        .reindex([r for r in REGIME_ORDER])
    )
    return FragilityResult(
        daily=out,
        by_regime=by_regime,
        rating=rating,
        c_t=c_t,
        anchor_floor=anchor_floor,
    )


def save_fragility(result: FragilityResult, outdir: Union[str, Path]) -> "tuple[Path, Path]":
    """Write the daily panel and regime summary as CSVs; return their paths."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    daily_path = outdir / f"fragility_index_{result.rating}_daily.csv"
    regime_path = outdir / f"fragility_index_{result.rating}_by_regime.csv"
    result.daily.to_csv(daily_path, index=False)
    result.by_regime.to_csv(regime_path)
    return daily_path, regime_path
