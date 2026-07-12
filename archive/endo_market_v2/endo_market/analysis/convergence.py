"""Convergence diagnostics for the policy<->distribution iteration.

The Repeated-Risk-Minimization loop produces a sequence of policy iterates
``phi_0, phi_1, ...``.  We never know the fixed point ``phi*`` in advance, so the
key estimator is the **empirical contraction modulus** built from *consecutive
iterate differences* rather than distances to ``phi*``:

For a (locally) linear map ``e_{k+1} = -m e_k`` (the cobweb that arises here), the
successive differences ``d_k = phi_{k+1} - phi_k`` satisfy ``|d_{k+1} / d_k| = m``.
So the median of the ratios of successive step sizes is a robust estimate of the
spectral radius ``m`` of the iteration -- and the loop converges iff ``m < 1``.

These functions accept either a 1-D sequence (a scalar iterate coordinate, e.g.
the central half-spread, which is the dominant cobweb coordinate) or a 2-D array
of flat iterates ``[K+1, P]`` (reduced to successive L2 step sizes).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np

RunClass = Literal["converged", "oscillating", "divergent"]


def _step_sizes(iterates: np.ndarray) -> np.ndarray:
    """Return successive step sizes ``||phi_{k+1} - phi_k||`` from an iterate array.

    A 1-D input is treated as a scalar coordinate (absolute successive
    differences); a 2-D ``[K+1, P]`` input gives L2 norms of successive rows.
    """
    arr = np.asarray(iterates, dtype=float)
    if arr.ndim == 1:
        return np.abs(np.diff(arr))
    return np.linalg.norm(np.diff(arr, axis=0), axis=1)


def empirical_lipschitz(
    iterates: Sequence[float] | np.ndarray,
    eps: float = 1e-8,
    burn_in: int = 0,
) -> float:
    """Estimate the contraction modulus from successive iterate differences.

    Parameters
    ----------
    iterates:
        1-D scalar coordinate sequence or 2-D ``[K+1, P]`` flat-iterate array.
    eps:
        Floor on the denominator step size (avoids divide-by-zero when the
        iteration has essentially stopped moving).
    burn_in:
        Number of initial steps to drop before estimating (the first re-optimise
        from the cold initial policy is often an outlier).

    Returns
    -------
    float
        Median ratio of successive step sizes (the empirical spectral radius).
        Returns ``0.0`` if there are too few steps.
    """
    steps = _step_sizes(iterates)
    if burn_in > 0:
        steps = steps[burn_in:]
    if steps.size < 2:
        return 0.0
    num = steps[1:]
    den = steps[:-1]
    mask = den > eps
    if not mask.any():
        return 0.0
    ratios = num[mask] / den[mask]
    return float(np.median(ratios))


def fixed_point_residual(iterates: Sequence[float] | np.ndarray) -> float:
    """Return the final successive step size (a proxy for the fixed-point residual)."""
    steps = _step_sizes(iterates)
    if steps.size == 0:
        return 0.0
    return float(steps[-1])


def is_oscillating(iterates: Sequence[float] | np.ndarray, min_flips: int = 2) -> bool:
    """Whether the scalar iterate sequence alternates direction (cobweb signature)."""
    arr = np.asarray(iterates, dtype=float)
    if arr.ndim != 1 or arr.size < 4:
        return False
    diffs = np.diff(arr)
    signs = np.sign(diffs)
    signs = signs[signs != 0]
    if signs.size < 3:
        return False
    flips = int(np.sum(signs[1:] != signs[:-1]))
    return flips >= min_flips


def classify_run(
    iterates: Sequence[float] | np.ndarray,
    tol: float = 1e-3,
    margin: float = 0.15,
    lipschitz: float | None = None,
) -> RunClass:
    """Classify an RRM run as converged / oscillating / divergent.

    Logic (in order):

    * if the iteration has essentially stopped (final step ``< tol``) **or** the
      modulus is clearly below 1, it is ``"converged"``;
    * else if the modulus is clearly above 1 (steps growing), it is
      ``"divergent"``;
    * otherwise (steps neither clearly shrinking nor growing, often sign-flipping
      near the stability boundary) it is ``"oscillating"``.

    Parameters
    ----------
    iterates:
        Scalar coordinate sequence or flat-iterate array.
    tol:
        Step-size threshold below which the run is treated as settled.
    margin:
        Dead-band around 1 for the modulus so near-critical runs are labelled
        ``"oscillating"`` rather than forced into converged/divergent.
    lipschitz:
        Pre-computed modulus; if ``None`` it is estimated here.
    """
    steps = _step_sizes(iterates)
    L = empirical_lipschitz(iterates) if lipschitz is None else lipschitz
    final_step = float(steps[-1]) if steps.size else 0.0

    if final_step < tol or L < 1.0 - margin:
        return "converged"
    if L > 1.0 + margin:
        return "divergent"
    return "oscillating"
