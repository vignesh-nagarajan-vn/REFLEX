"""Real-data calibration: map the verified market dataset to simulator configs.

* :mod:`.loader` -- read the shipped calibration CSVs (fitted intensity params,
  regime summary, cross-sectional sigma, scaler stats, the daily master panel).
* :mod:`.mapping` -- the ``(rating, regime) -> Config`` parameter map with the
  package's single unit-conversion point and the documented structural ratios.
* :mod:`.regimes` -- VIX-regime thresholds, ordering and plot colours.
"""

from .loader import (
    RATINGS,
    REGIMES,
    RegimeMicrostructure,
    default_data_dir,
    load_intensity_params,
    load_master,
    load_regime_summary,
    load_scaler_stats,
    load_xsection_sigma,
    regime_microstructure,
)
from .mapping import (
    CalibrationInfo,
    apply_calibration,
    calibrated_config,
)
from .regimes import REGIME_COLORS, REGIME_ORDER, VIX_CUTOFFS, classify_vix

__all__ = [
    "RATINGS",
    "REGIMES",
    "RegimeMicrostructure",
    "default_data_dir",
    "load_intensity_params",
    "load_master",
    "load_regime_summary",
    "load_scaler_stats",
    "load_xsection_sigma",
    "regime_microstructure",
    "CalibrationInfo",
    "apply_calibration",
    "calibrated_config",
    "REGIME_COLORS",
    "REGIME_ORDER",
    "VIX_CUTOFFS",
    "classify_vix",
]
