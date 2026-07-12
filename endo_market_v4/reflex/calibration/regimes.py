"""VIX-regime utilities shared by calibration and the fragility index.

The dataset classifies each day into five volatility regimes.  The thresholds
below reproduce the dataset's ``regime`` column convention (consistent with the
per-regime VIX means in ``reflex_I_calibration_params.csv``: calm ~12.9,
normal ~17.4, elevated ~23.9, stress ~35.0, crisis ~58.5).
"""

from __future__ import annotations

from typing import Dict

from .loader import REGIMES

#: VIX upper cutoffs per regime (a day is the first regime whose cutoff exceeds it).
VIX_CUTOFFS: Dict[str, float] = {
    "calm": 15.0,
    "normal": 20.0,
    "elevated": 30.0,
    "stress": 50.0,
    "crisis": float("inf"),
}

#: Plot colours per regime (calm -> crisis).
REGIME_COLORS: Dict[str, str] = {
    "calm": "#4daf4a",
    "normal": "#377eb8",
    "elevated": "#ff7f00",
    "stress": "#e41a1c",
    "crisis": "#67001f",
}

#: Numeric severity ordering (calm=0 ... crisis=4).
REGIME_ORDER: Dict[str, int] = {r: i for i, r in enumerate(REGIMES)}


def classify_vix(vix: float) -> str:
    """Map a VIX level to its regime label (dataset convention)."""
    for regime in REGIMES:
        if vix < VIX_CUTOFFS[regime]:
            return regime
    return "crisis"
