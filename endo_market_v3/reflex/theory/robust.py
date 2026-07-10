"""Distributionally robust stability certificate (math-theory 1.4).

The nominal boundary of 1.1 is the deterministic inequality ``m < 1`` (equivalently
``epsilon < gamma/beta``).  But ``m`` is never observed -- it is *estimated* by the
common-random-numbers (CRN) best-response probe
:func:`measure_response_modulus` from a finite number ``n`` of simulated episodes,
so the crossing ``m = 1`` is a random event.  This module turns the point boundary
into a **statistically defensible** one, following
``new-methodology/math-theory/04-robust-uncertainty.md``:

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
from ..estimators.br_slope import measure_response_modulus


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
    cfg_local = copy.deepcopy(cfg)
    if n_episodes is not None:
        cfg_local.rrm.n_episodes = int(n_episodes)
    return [
        measure_response_modulus(cfg_local, seed=int(s), h_ref=h_ref, delta=delta).modulus
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
) -> RobustBoundaryResult:
    """Cross-seed robust stability certificate for ``cfg`` (1.4 §3--§4, §6).

    Runs the CRN probe over ``seeds``, fits the ambiguity radius ``delta_n = z*s``,
    and returns the certificate in modulus space (against ``boundary = 1``) and its
    ``epsilon``-space image (against ``gamma/beta``), including the structural
    floor ``eta_mod``.
    """
    ms = measure_modulus_estimates(cfg, seeds, n_episodes=n_episodes, h_ref=h_ref, delta=delta)
    mean_m, _std_m, radius_m = empirical_radius(ms, confidence=confidence)
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
