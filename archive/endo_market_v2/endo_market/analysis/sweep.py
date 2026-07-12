"""Sweep a control knob and build the stability phase diagram.

For each value of the swept variable (the performative-feedback gain
``toxicity_feedback`` = epsilon for the primary experiment, or ``alpha`` for the
secondary one) we estimate the best-response contraction modulus
(:func:`measure_response_modulus`) across several seeds and summarise it by its
median and inter-quartile range.  The critical value where the **median** modulus
crosses 1 is the empirical stability boundary.

We sweep ``toxicity_feedback`` as the primary axis because it scales the
performative feedback directly; sweeping ``alpha`` is confounded by best-response
saturation (the dealer flees to wide spreads where the spread-capture curvature
vanishes), which flattens or even reverses the trend -- a finding worth reporting
but not the clean control.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import yaml

from ..config import Config, _from_dict
from .response_modulus import measure_response_modulus

# Map the short sweep-variable name to the (config-section, field) it sets.
_VARIABLE_PATHS = {
    "toxicity_feedback": ("clients", "toxicity_feedback"),
    "alpha": ("clients", "alpha"),
    "quote_anchor_weight": ("reward", "quote_anchor_weight"),
    "info_base_intensity": ("clients", "info_base_intensity"),
}


@dataclass
class SweepPoint:
    """Modulus statistics at a single swept value."""

    value: float
    moduli: List[float]
    median: float
    q25: float
    q75: float
    frac_unstable: float  # fraction of seeds with modulus > 1


@dataclass
class SweepResult:
    """Full phase-diagram sweep."""

    variable: str
    points: List[SweepPoint] = field(default_factory=list)
    critical_value: Optional[float] = None  # where the median modulus crosses 1

    @property
    def values(self) -> np.ndarray:
        return np.array([p.value for p in self.points], dtype=float)

    @property
    def medians(self) -> np.ndarray:
        return np.array([p.median for p in self.points], dtype=float)

    def to_rows(self) -> List[Dict[str, Any]]:
        """Flat rows for CSV export."""
        rows = []
        for p in self.points:
            rows.append(
                {
                    "variable": self.variable,
                    "value": p.value,
                    "median_modulus": p.median,
                    "q25": p.q25,
                    "q75": p.q75,
                    "frac_unstable": p.frac_unstable,
                    "moduli": ";".join(f"{m:.4f}" for m in p.moduli),
                }
            )
        return rows


def load_sweep_spec(path: str | Path) -> Dict[str, Any]:
    """Load a sweep YAML (with ``base`` and ``sweep`` blocks)."""
    with open(path, "r") as fh:
        return yaml.safe_load(fh) or {}


def _config_from_base(spec: Dict[str, Any]) -> Config:
    base = dict(spec.get("base", {}))
    for key in ("seed", "device", "dtype"):
        if key in spec:
            base.setdefault(key, spec[key])
    return _from_dict(Config, base)


def _set_variable(cfg: Config, variable: str, value: float) -> None:
    if variable not in _VARIABLE_PATHS:
        raise ValueError(f"unknown sweep variable {variable!r}; known: {list(_VARIABLE_PATHS)}")
    section, field_name = _VARIABLE_PATHS[variable]
    setattr(getattr(cfg, section), field_name, value)


def _locate_crossing(values: np.ndarray, medians: np.ndarray) -> Optional[float]:
    """Linear-interpolate the first up-crossing of median modulus through 1."""
    for i in range(len(values) - 1):
        a, b = medians[i], medians[i + 1]
        if a < 1.0 <= b:
            t = (1.0 - a) / (b - a)
            return float(values[i] + t * (values[i + 1] - values[i]))
    return None


def run_sweep(
    spec: Dict[str, Any],
    verbose: bool = True,
) -> SweepResult:
    """Run a modulus sweep from a loaded sweep spec.

    Parameters
    ----------
    spec:
        Parsed sweep YAML with ``base`` and ``sweep`` blocks.  ``sweep`` must
        provide ``variable``, ``grid``, ``n_seeds`` (and optionally ``seed_base``,
        ``h_ref``, ``delta``).
    verbose:
        Print a line per swept value.

    Returns
    -------
    SweepResult
    """
    sweep = spec["sweep"]
    variable = sweep["variable"]
    grid = [float(x) for x in sweep["grid"]]
    n_seeds = int(sweep.get("n_seeds", 3))
    seed_base = int(sweep.get("seed_base", 0))
    h_ref = float(sweep.get("h_ref", 1.0))
    delta = float(sweep.get("delta", 0.25))

    result = SweepResult(variable=variable)
    for value in grid:
        t0 = time.time()
        moduli: List[float] = []
        for s in range(n_seeds):
            cfg = _config_from_base(spec)
            _set_variable(cfg, variable, value)
            res = measure_response_modulus(cfg, seed=seed_base + s, h_ref=h_ref, delta=delta)
            moduli.append(res.modulus)
        arr = np.array(moduli)
        point = SweepPoint(
            value=value,
            moduli=moduli,
            median=float(np.median(arr)),
            q25=float(np.quantile(arr, 0.25)),
            q75=float(np.quantile(arr, 0.75)),
            frac_unstable=float(np.mean(arr > 1.0)),
        )
        result.points.append(point)
        if verbose:
            print(
                f"  {variable}={value:6.2f}: median m={point.median:.3f} "
                f"[{point.q25:.2f},{point.q75:.2f}] frac_unstable={point.frac_unstable:.2f} "
                f"({time.time()-t0:.0f}s)"
            )

    result.critical_value = _locate_crossing(result.values, result.medians)
    return result
