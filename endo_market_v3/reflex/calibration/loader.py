"""Load the shipped real-data calibration artifacts.

The package ships (under ``endo_market_v3/data/``) small, verified extracts of
the REFLEX data pipeline (``new-methodology/{data_collection,preprocessing}``):

* ``calibration/03_fitted_intensity_params.csv`` -- the exponential-intensity
  fits ``lambda(h) = A * exp(-k*h)`` per ``(rating_bucket, regime)`` from the
  2004-2021 TRACE window (VIX-proxied observations; see the provenance notes in
  ``data/README.md`` -- the fits are *proxy-level*, not trade-level TRACE).
* ``calibration/reflex_I_calibration_params.csv`` -- per-regime macro summary
  (VIX / yields / spread / sigma ranges), used for cross-checks.
* ``calibration/reflex_G2_bond_xsection_sigma.csv`` -- monthly cross-sectional
  dispersion of 212 real-CUSIP bond returns (per-bond sigma calibration, 1.5).
* ``calibration/scaler_stats.csv`` -- feature mu/sigma fit on the 2004-2019
  calibration window (lookahead-safe normalisation).
* ``master/REFLEX_MASTER_DATASET.csv`` -- the daily 1990-2026 joined panel
  (9,218 rows) used by the fragility index.

All loaders take an optional ``data_dir`` override; by default they resolve the
directory shipped with this repository so the package is self-contained.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import pandas as pd

#: Canonical regime ordering (calm -> crisis), as used across the dataset.
REGIMES = ("calm", "normal", "elevated", "stress", "crisis")
#: Rating buckets with fitted intensity parameters.
RATINGS = ("IG", "HY")


def default_data_dir() -> Path:
    """The ``data/`` directory shipped inside ``endo_market_v3/``."""
    return Path(__file__).resolve().parents[2] / "data"


def _resolve(data_dir: Optional[Union[str, Path]]) -> Path:
    d = Path(data_dir) if data_dir else default_data_dir()
    if not d.exists():
        raise FileNotFoundError(
            f"calibration data directory not found: {d} -- pass data_dir explicitly "
            "or run from a checkout that ships endo_market_v3/data/"
        )
    return d


# --------------------------------------------------------------------------- #
# Fitted intensity parameters (the core calibration table)                    #
# --------------------------------------------------------------------------- #
@dataclass
class RegimeMicrostructure:
    """Fitted microstructure primitives for one ``(rating, regime)`` cell.

    Units are the *dataset's*: ``h`` in decimal fraction of price (50 bps =
    0.005), ``k_decay`` per unit decimal-``h`` (so ``k*h`` is dimensionless),
    ``A`` in arrivals/notional per step, ``sigma_annual`` an annualised return
    vol.  :mod:`reflex.calibration.mapping` converts to simulator units.
    """

    rating: str
    regime: str
    A: float  # fitted arrival scale  (lambda(h) = A * exp(-k*h))
    k_decay: float  # fitted exponential decay of arrivals in decimal h
    sigma_annual: float  # annualised return volatility for the bucket
    h_mean_decimal: float  # mean observed half-spread (decimal)
    n_days: int  # observations behind the fit
    resid_stationary: bool  # ADF-stationary fit residuals (fails in stress/crisis)

    @property
    def degenerate(self) -> bool:
        """True when the fit is degenerate (crisis cells: k pinned to 0, n=74)."""
        return not (self.k_decay > 0.0) or not math.isfinite(self.k_decay)


def load_intensity_params(data_dir: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """Load ``03_fitted_intensity_params.csv`` indexed by ``(rating, regime)``."""
    d = _resolve(data_dir)
    df = pd.read_csv(d / "calibration" / "03_fitted_intensity_params.csv")
    df = df.set_index(["rating_bucket", "regime"]).sort_index()
    return df


def regime_microstructure(
    rating: str = "IG",
    regime: str = "normal",
    data_dir: Optional[Union[str, Path]] = None,
) -> RegimeMicrostructure:
    """Return the fitted :class:`RegimeMicrostructure` for one cell."""
    if rating not in RATINGS:
        raise ValueError(f"unknown rating {rating!r} (expected one of {RATINGS})")
    if regime not in REGIMES:
        raise ValueError(f"unknown regime {regime!r} (expected one of {REGIMES})")
    df = load_intensity_params(data_dir)
    row = df.loc[(rating, regime)]
    return RegimeMicrostructure(
        rating=rating,
        regime=regime,
        A=float(row["A_fit"]),
        k_decay=float(row["k_decay_fit"]),
        sigma_annual=float(row["sigma_mean"]),
        h_mean_decimal=float(row["h_mean_decimal"]),
        n_days=int(row["n_days"]),
        resid_stationary=bool(row["resid_stationary"]),
    )


# --------------------------------------------------------------------------- #
# The other shipped artifacts                                                 #
# --------------------------------------------------------------------------- #
def load_regime_summary(data_dir: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """Per-regime macro summary (``reflex_I_calibration_params.csv``)."""
    d = _resolve(data_dir)
    return pd.read_csv(d / "calibration" / "reflex_I_calibration_params.csv").set_index("regime")


def load_xsection_sigma(data_dir: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """Monthly cross-sectional bond-return dispersion (``reflex_G2``)."""
    d = _resolve(data_dir)
    df = pd.read_csv(d / "calibration" / "reflex_G2_bond_xsection_sigma.csv")
    return df


def load_bond_returns(data_dir: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """Per-bond monthly log returns for the 212 real CUSIPs (``reflex_G``)."""
    d = _resolve(data_dir)
    df = pd.read_csv(d / "calibration" / "reflex_G_bond_returns_monthly.csv",
                     parse_dates=["date"])
    return df


def bond_vol_dispersion(
    data_dir: Optional[Union[str, Path]] = None,
    min_months: int = 12,
) -> float:
    """Relative cross-sectional dispersion of per-bond volatility.

    Computes each bond's time-series return volatility from the shipped
    per-CUSIP monthly returns (bonds with at least ``min_months`` observations)
    and returns the coefficient of variation ``std(sigma_i) / mean(sigma_i)``
    across bonds -- the data-identified heterogeneity of per-bond sigmas that
    theory 1.5 scales the universe's idiosyncratic vols by.  The aggregated G2
    panel cannot identify this (a single cross-sectional sigma per month mixes
    vol heterogeneity with factor exposure), hence the per-bond file.
    """
    df = load_bond_returns(data_dir)
    counts = df.groupby("cusip")["log_ret_monthly"].count()
    keep = counts[counts >= int(min_months)].index
    sigmas = (
        df[df["cusip"].isin(keep)]
        .groupby("cusip")["log_ret_monthly"]
        .std(ddof=1)
        .dropna()
    )
    # Zero-vol bonds are stale marks (flat price series), not zero-risk bonds;
    # drop them, then winsorise the sigma distribution at p05/p95 (the pipeline's
    # own tail-trimming convention) before taking the coefficient of variation.
    sigmas = sigmas[sigmas > 0.0]
    if len(sigmas) < 10:
        raise ValueError(
            f"too few bonds with >= {min_months} months of non-stale returns "
            f"({len(sigmas)}) to estimate vol dispersion"
        )
    lo, hi = sigmas.quantile(0.05), sigmas.quantile(0.95)
    w = sigmas.clip(lo, hi)
    return float(w.std(ddof=1) / w.mean())


def load_scaler_stats(data_dir: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """Calibration-window feature mu/sigma (``scaler_stats.csv``)."""
    d = _resolve(data_dir)
    return pd.read_csv(d / "calibration" / "scaler_stats.csv")


def load_master(data_dir: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """The daily 1990-2026 master panel with parsed dates."""
    d = _resolve(data_dir)
    df = pd.read_csv(d / "master" / "REFLEX_MASTER_DATASET.csv", parse_dates=["date"])
    return df
