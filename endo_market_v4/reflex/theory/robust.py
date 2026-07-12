"""Distributionally robust stability certificate (math-theory 1.4).

The nominal boundary of 1.1 is the deterministic inequality ``m < 1`` (equivalently
``epsilon < gamma/beta``).  But ``m`` is never observed -- it is *estimated* by the
common-random-numbers (CRN) best-response probe
:func:`measure_response_modulus` from a finite number ``n`` of simulated episodes,
so the crossing ``m = 1`` is a random event.  This module turns the point boundary
into a **statistically defensible** one, following
``research/math-theory/04-robust-uncertainty.md``:

* the estimator concentrates at the **parametric rate** ``|m_hat_n - m| =
  O_p(1/sqrt(n))`` -- a property *bought by the CRN construction* in the probe (a
  naive independent-noise difference gives only ``n^{-1/3}``; §2);
* an **ambiguity ball** of radius ``delta_n = z * s`` is fit from the cross-seed
  standard deviation ``s`` of the estimate (§3.1);
* the **robust certificate** (Cao & Shi 2022) declares stability only when the whole
  ball is stable-side, ``m_bar + delta_n < 1`` -- a robust boundary
  ``m*_rob = 1 - delta_n`` that closes on the nominal one at ``O(1/sqrt(n))``
  (Theorem 1, §4);
* the **sample complexity** ``n_req = (z*sigma/Delta)^2`` to resolve a market a
  distance ``Delta`` from the boundary diverges as ``Delta -> 0`` (§5); and
* **statistical** uncertainty (shrinks with data) is separated from **structural**
  model uncertainty ``eta_mod`` (a fixed floor; §6.2).

Working space.  We phrase everything in the directly measured **modulus** ``m``
with boundary ``1`` (1.4 §1: "all statements hold verbatim for ``m_hat`` with
boundary ``m = 1``").  The equivalent ``epsilon``-space statement follows by
``epsilon = m * gamma/beta`` with ``gamma``, ``beta`` from :mod:`analytic_boundary`;
:func:`robust_boundary` reports both.

The pure-statistics helpers (:func:`empirical_radius`, :func:`robust_certificate`,
:func:`sample_complexity`, :func:`loglog_rate`) are dependency-light and fast; the
empirical harness (:func:`measure_modulus_estimates`, :func:`robust_boundary`,
:func:`rate_check`) drives the full deploy->collect->fit->optimize probe over seeds.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
from scipy.stats import norm

from ..config import Config
from .analytic_boundary import analytic_boundary
# NOTE: the torch-based BR-slope estimator is imported lazily inside
# measure_modulus_estimates -- keeps the closed forms numpy-only and avoids
# the estimators -> equilibrium -> theory -> estimators import cycle.


def _z(confidence: float, two_sided: bool = False) -> float:
    """Normal quantile for a one- (default) or two-sided confidence level.

    The certificate is one-sided (1.4 §6.3: ``P(declare stable | truly unstable)
    <= a``), so the default is ``z = Phi^{-1}(confidence)`` (``1.645`` at 95%).
    """
    a = 1.0 - float(confidence)
    q = 1.0 - (a / 2.0 if two_sided else a)
    return float(norm.ppf(q))


# --------------------------------------------------------------------------- #
# Pure statistics (fast, no torch)                                            #
# --------------------------------------------------------------------------- #
def empirical_radius(
    estimates: Sequence[float],
    confidence: float = 0.95,
    two_sided: bool = False,
) -> tuple[float, float, float]:
    """Cross-seed mean, std and ambiguity radius ``delta_n = z * s`` (1.4 §3.1).

    ``estimates`` are per-seed modulus (or epsilon) estimates, each from ``n``
    episodes.  ``s`` (sample std, ``ddof=1``) already estimates the per-estimate
    std ``sigma/sqrt(n)``, so ``delta_n = z*s`` is the confidence half-width of a
    single ``n``-episode estimate (the ``1/sqrt(n)`` scaling is *tested* via
    :func:`rate_check`, not divided in here).
    """
    arr = np.asarray(list(estimates), dtype=float)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    return mean, std, _z(confidence, two_sided) * std


@dataclass
class RobustCertificate:
    """Distributionally robust stability verdict (1.4 §4)."""

    mean: float  # ebar: the point estimate (modulus space unless noted)
    radius: float  # delta_n: statistical ambiguity half-width
    boundary: float  # nominal boundary (1.0 in modulus space; gamma/beta in eps space)
    eta_mod: float  # structural (model) relative ambiguity floor (1.4 §6.2)
    upper: float  # ebar*(1+eta_mod) + delta_n  (worst case)
    lower: float  # ebar*(1-eta_mod) - delta_n  (best case)
    robust_boundary: float  # boundary - delta_n (statistical robust boundary)
    verdict: str  # "stable" | "unstable" | "undecided"
    confidence: float


def robust_certificate(
    mean: float,
    radius: float,
    boundary: float = 1.0,
    eta_mod: float = 0.0,
    confidence: float = 0.95,
) -> RobustCertificate:
    """Robust stability certificate (Theorem 1, 1.4 §4; structural floor §6.2).

    Declares ``"stable"`` iff the whole ambiguity ball is below ``boundary``
    (``mean*(1+eta_mod) + radius < boundary``), ``"unstable"`` iff it is entirely
    above (``mean*(1-eta_mod) - radius > boundary``), and ``"undecided"`` in the
    band between -- the region ``n`` episodes cannot yet call.  ``eta_mod`` is the
    *structural* relative ambiguity that does **not** shrink with data.
    """
    upper = mean * (1.0 + eta_mod) + radius
    lower = mean * (1.0 - eta_mod) - radius
    if upper < boundary:
        verdict = "stable"
    elif lower > boundary:
        verdict = "unstable"
    else:
        verdict = "undecided"
    return RobustCertificate(
        mean=mean,
        radius=radius,
        boundary=boundary,
        eta_mod=eta_mod,
        upper=upper,
        lower=lower,
        robust_boundary=boundary - radius,
        verdict=verdict,
        confidence=confidence,
    )


def sample_complexity(sigma: float, distance: float, confidence: float = 0.95) -> float:
    """Episodes needed to resolve a market a distance ``Delta`` from the boundary.

    ``n_req = (z * sigma / Delta)^2 = O(Delta^{-2})`` (1.4 §5); diverges as the
    truth approaches the boundary (``Delta -> 0``).  ``sigma`` is the per-episode
    estimator std (so that the per-``n`` half-width is ``z*sigma/sqrt(n)``).
    """
    if distance <= 0.0:
        return math.inf
    return (_z(confidence) * float(sigma) / float(distance)) ** 2


def finite_sample_radius(sigma_eps: float, n: int, alpha: float = 0.05) -> float:
    """Sub-Gaussian (Hoeffding) finite-sample radius ``sigma*sqrt(2 log(2/a)/n)`` (1.4 §2.4).

    The honest radius for a hard certificate; agrees in *rate* (``O(1/sqrt(n))``)
    with the asymptotic ``z*s`` of :func:`empirical_radius` but with a larger
    constant.
    """
    return float(sigma_eps) * math.sqrt(2.0 * math.log(2.0 / alpha) / max(int(n), 1))


# --------------------------------------------------------------------------- #
# Ambiguity-radius calibration (v4 tuning)                                     #
# --------------------------------------------------------------------------- #
@dataclass
class RadiusCalibration:
    """Calibration report for the ambiguity radius (v4; tunes 1.4 §3.1).

    The ``z*s`` radius assumes the per-seed estimates are ~normal.  This
    report puts a distribution-free companion next to it: the one-sided
    empirical-quantile radius of the observed deviations, the implied
    multiplier on ``z*s``, and the bootstrap coverage of both radii under the
    empirical distribution of the estimates.  The **calibrated radius** is
    ``max(z_radius, quantile_radius)`` -- never tighter than the parametric
    one, inflated exactly when the observed tail says the normal
    approximation is too optimistic.
    """

    n: int
    mean: float
    std: float
    z_radius: float  # z * s (1.4 section 3.1)
    quantile_radius: float  # one-sided empirical-quantile radius
    calibrated_radius: float  # max(z_radius, quantile_radius)
    multiplier: float  # quantile_radius / z_radius (1.0 = normal approx exact)
    coverage_normal: float  # bootstrap one-sided coverage of mean + z_radius
    coverage_calibrated: float  # ... of mean + calibrated_radius
    confidence: float


def calibrate_radius(
    estimates: Sequence[float],
    confidence: float = 0.95,
    n_boot: int = 4000,
    seed: int = 0,
) -> RadiusCalibration:
    """Calibrate the ambiguity radius on per-seed estimates (v4 tuning).

    Two ingredients, both distribution-free:

    * **quantile radius** -- the one-sided ``confidence`` quantile of the
      centered deviations ``m_i - mean`` (the empirical counterpart of the
      ``z*s`` half-width, which it matches exactly for normal data as
      ``n -> infinity``); and
    * **bootstrap coverage** -- resample the estimates with replacement
      ``n_boot`` times; for each replicate check whether the grand mean lies
      below ``mean_b + radius_b`` (the one-sided event the certificate needs,
      1.4 §6.3), with each radius recomputed on the replicate.

    The tuned recommendation implemented here: keep ``z*s`` when the measured
    multiplier is ~1 (the CRN probe's estimates are close to normal -- the 1.4
    §2 parametric-rate argument), switch to ``max(z*s, quantile)`` when the
    tail is heavier.  ``robust_boundary(..., radius_method="calibrated")``
    applies it end to end.
    """
    arr = np.asarray(list(estimates), dtype=float)
    n = int(arr.size)
    if n < 2:
        raise ValueError("calibrate_radius needs at least 2 estimates")
    mean = float(arr.mean())
    std = float(arr.std(ddof=1))
    z_rad = _z(confidence) * std
    dev = arr - mean
    q_rad = float(np.quantile(dev, confidence))
    q_rad = max(q_rad, 0.0)
    cal_rad = max(z_rad, q_rad)

    rng = np.random.default_rng(seed)
    covered_norm = 0
    covered_cal = 0
    for _ in range(int(n_boot)):
        b = arr[rng.integers(0, n, size=n)]
        mb = float(b.mean())
        sb = float(b.std(ddof=1))
        zb = _z(confidence) * sb
        qb = max(float(np.quantile(b - mb, confidence)), 0.0)
        if mean <= mb + zb:
            covered_norm += 1
        if mean <= mb + max(zb, qb):
            covered_cal += 1
    return RadiusCalibration(
        n=n,
        mean=mean,
        std=std,
        z_radius=z_rad,
        quantile_radius=q_rad,
        calibrated_radius=cal_rad,
        multiplier=(q_rad / z_rad if z_rad > 0.0 else float("inf")),
        coverage_normal=covered_norm / max(n_boot, 1),
        coverage_calibrated=covered_cal / max(n_boot, 1),
        confidence=float(confidence),
    )


def loglog_rate(ns: Sequence[float], stds: Sequence[float]) -> float:
    """Least-squares slope of ``log(std)`` vs ``log(n)`` (1.4 §2.3, §8 check 1).

    The parametric-rate prediction is a slope of ``-1/2``; a slope near ``-1/3``
    would reveal the CRN coupling is broken (§2.2).
    """
    x = np.log(np.asarray(list(ns), dtype=float))
    y = np.log(np.asarray(list(stds), dtype=float))
    A = np.vstack([x, np.ones_like(x)]).T
    slope, _ = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(slope)


# --------------------------------------------------------------------------- #
# Empirical harness (drives the full probe; slow)                             #
# --------------------------------------------------------------------------- #
def measure_modulus_estimates(
    cfg: Config,
    seeds: Sequence[int],
    n_episodes: Optional[int] = None,
    h_ref: float = 1.0,
    delta: float = 0.25,
) -> List[float]:
    """Measure the CRN BR-slope modulus once per seed (each on ``n`` episodes).

    ``n_episodes`` overrides ``cfg.rrm.n_episodes`` (the ``n`` of the ``O(1/sqrt(n))``
    rate); left ``None`` it uses the config's value.
    """
    from ..estimators.br_slope import measure_response_modulus as _measure

    cfg_local = copy.deepcopy(cfg)
    if n_episodes is not None:
        cfg_local.rrm.n_episodes = int(n_episodes)
    return [
        _measure(cfg_local, seed=int(s), h_ref=h_ref, delta=delta).modulus
        for s in seeds
    ]


@dataclass
class RobustBoundaryResult:
    """Full robust-boundary report in both modulus and epsilon space (1.4 §4)."""

    n_episodes: int
    n_seeds: int
    modulus: RobustCertificate  # certificate against m = 1
    epsilon: RobustCertificate  # certificate against gamma/beta
    epsilon_star: float  # nominal boundary gamma/beta (1.1)
    gamma_over_beta: float  # scale converting modulus to epsilon


def robust_boundary(
    cfg: Config,
    seeds: Sequence[int],
    n_episodes: Optional[int] = None,
    h_ref: float = 1.0,
    delta: float = 0.25,
    confidence: float = 0.95,
    eta_mod: float = 0.0,
    boundary: float = 1.0,
    radius_method: str = "normal",
) -> RobustBoundaryResult:
    """Cross-seed robust stability certificate for ``cfg`` (1.4 §3--§4, §6).

    Runs the CRN probe over ``seeds``, fits the ambiguity radius (``delta_n =
    z*s`` for ``radius_method="normal"``; the v4-calibrated
    ``max(z*s, empirical quantile)`` of :func:`calibrate_radius` for
    ``"calibrated"``), and returns the certificate in modulus space (against
    ``boundary = 1``) and its ``epsilon``-space image (against ``gamma/beta``),
    including the structural floor ``eta_mod``.
    """
    if radius_method not in ("normal", "calibrated"):
        raise ValueError(f"unknown radius_method {radius_method!r}")
    ms = measure_modulus_estimates(cfg, seeds, n_episodes=n_episodes, h_ref=h_ref, delta=delta)
    mean_m, _std_m, radius_m = empirical_radius(ms, confidence=confidence)
    if radius_method == "calibrated":
        radius_m = calibrate_radius(ms, confidence=confidence).calibrated_radius
    cert_m = robust_certificate(mean_m, radius_m, boundary=boundary, eta_mod=eta_mod, confidence=confidence)

    # epsilon-space image: epsilon = m * gamma/beta, boundary epsilon* = gamma/beta.
    ab = analytic_boundary(cfg, mispricing=None, liquidity_ratio=1.0)
    scale = ab.gamma / ab.beta if ab.beta != 0.0 else float("inf")
    cert_e = robust_certificate(
        mean_m * scale, radius_m * scale, boundary=ab.boundary_epsilon,
        eta_mod=eta_mod, confidence=confidence,
    )
    n_used = int(cfg.rrm.n_episodes if n_episodes is None else n_episodes)
    return RobustBoundaryResult(
        n_episodes=n_used,
        n_seeds=len(list(seeds)),
        modulus=cert_m,
        epsilon=cert_e,
        epsilon_star=ab.boundary_epsilon,
        gamma_over_beta=scale,
    )


def rate_check(
    cfg: Config,
    n_values: Sequence[int],
    seeds: Sequence[int],
    h_ref: float = 1.0,
    delta: float = 0.25,
) -> Dict[str, object]:
    """Measure the cross-seed std at several ``n`` and fit the log-log rate (1.4 §8).

    Returns ``{"n": [...], "std": [...], "slope": s}``; the parametric-rate
    prediction is ``slope ~ -0.5`` (a broken CRN coupling would give ``~ -1/3``).
    """
    ns: List[int] = []
    stds: List[float] = []
    for n in n_values:
        ms = measure_modulus_estimates(cfg, seeds, n_episodes=int(n), h_ref=h_ref, delta=delta)
        _mean, std, _rad = empirical_radius(ms)
        ns.append(int(n))
        stds.append(std)
    slope = loglog_rate(ns, stds) if len(ns) >= 2 and all(s > 0 for s in stds) else float("nan")
    return {"n": ns, "std": stds, "slope": slope}
