"""Scaling to 100+ correlated bonds: the modulus matrix and factor reduction (1.5).

Lifts the scalar boundary of 1.1 from one bond to a universe of ``d`` correlated
bonds, following ``research/math-theory/05-factor-model-scaling.md``.  The
scalar modulus ``m = epsilon*beta/gamma`` becomes the ``d x d`` **modulus matrix**

    M = beta * Gamma^{-1} * E ,     Gamma = D_gamma + zeta' * G * Sigma * G ,
                                    E = diag(epsilon_i) ,

and the loop contracts iff ``rho(M) < 1``.  The cross-bond coupling lives entirely
in the inventory Hessian ``zeta' * G * Sigma * G``, where ``Sigma`` is the
`bonds.py` factor covariance.  Because ``Gamma`` is **diagonal-plus-low-rank**, both
``Gamma^{-1}`` and ``rho(M)`` are computable in ``O(d*k^2)`` via Woodbury
(Bergault--Guéant), and truncating to the top ``k`` factors perturbs the boundary by
an amount **linear in the residual factor variance** ``lambda_{k+1}(C)`` (Theorem 1,
§5).

Provided here:

* :func:`per_bond_constants` -- per-bond ``gamma0_i``, ``epsilon_i``, ``sigma_i``,
  ``g_i`` from 1.1's closed forms + the `bonds.py` universe;
* :func:`modulus_matrix` / :func:`factor_modulus` -- the dense ``M``, its spectral
  radius, the top (market-factor) eigenvector, and ``chi = rho(M)/max_i m_1^(i)``;
* :func:`factorize_corr` / :func:`gamma_inverse_woodbury` /
  :func:`spectral_radius_woodbury` -- the ``O(d*k^2)`` factor reduction;
* :func:`truncation_error_bound` -- the §5 error bound and ``lambda_{k+1}(C)``;
* :func:`sweep_global_factor` -- the diversification law (§7.1); and
* :func:`systemic_spectral_radius` -- the 1.3 x 1.5 composition ``N_eff*chi*m_1``.

Honest note on the correlation direction (1.5 §3.3).  In the *clean, priced-risk*
``M = beta*Gamma^{-1}*E`` derived here, correlation enters through the inventory
**curvature** ``Gamma``, so a more correlated (higher ``global_factor``) book is
actually *more* stable along the market factor: the fragile direction of ``M`` is
the **idiosyncratic** one, and ``rho(M)`` *decreases* as correlation rises.  The
"correlation is destabilising" reading of §3.3 requires the un-hedged / operator-
blind channel (A3''), which this priced-risk modulus does not capture; we compute
and report the clean ``M`` and let the numbers speak, per that caveat.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from ..config import Config
from ..env.bonds import BondUniverse
from .analytic_boundary import (
    ReferenceState,
    epsilon as epsilon_of,
    reference_state,
    solve_fixed_point,
)


def _zeta_prime(cfg: Config) -> float:
    """Inventory risk-aversion curvature scale ``zeta' = P*zeta`` (1.5 A4'').

    ``zeta`` is taken proportional to ``reward.inv_risk_weight`` (1.5 §0/§8 map).
    """
    return float(cfg.reward.pnl_scale) * float(cfg.reward.inv_risk_weight)


@dataclass
class PerBondConstants:
    """Per-bond closed-form constants over the universe (1.5 §0/§2)."""

    h: np.ndarray  # per-bond fixed-point half-spread
    gamma0: np.ndarray  # own-bond curvature gamma_i^0 (excludes the inventory Hessian)
    epsilon: np.ndarray  # per-bond distribution sensitivity
    sigma: np.ndarray  # per-bond fundamental vol (DV01/duration-calibrated)
    g: np.ndarray  # inventory-to-spread sensitivity diag(dq_i/dh_i)
    beta: float  # shared joint smoothness ( = pnl_scale )


def per_bond_constants(
    cfg: Config,
    universe: BondUniverse,
    h: Optional[np.ndarray] = None,
    mispricing: Optional[float] = None,
    liquidity_ratio: float = 1.0,
    sigma: Optional[np.ndarray] = None,
    g: Optional[np.ndarray] = None,
    inventory_gain: float = 1.0,
) -> PerBondConstants:
    """Assemble the per-bond constants for the modulus matrix (1.5 §2, §6.1).

    With the client micro-constants shared across bonds (per-bond calibration is a
    documented extension, 1.5 §8), ``gamma0_i`` and ``epsilon_i`` are evaluated at
    each bond's fixed point ``h_i`` (identical here).  The cross-sectional texture
    enters through ``sigma_i`` -- calibrated from bond duration as a DV01 proxy
    ``sigma_i = fundamental_vol * duration_i / mean(duration)`` (1.5 §6.1) -- and the
    inventory sensitivity ``g_i = dq_i/dh_i``, a first-order proxy for the inventory
    *accumulated over a deployment* ``g_i = horizon*A*exp(-k*h_i)`` (scaled by
    ``inventory_gain``).  ``g_i`` is a leading-order operating-point model (1.5 §10
    caveat); both ``sigma`` and ``g`` may be overridden.
    """
    d = universe.n_bonds
    ref = reference_state(cfg, mispricing=mispricing, liquidity_ratio=liquidity_ratio)
    P = float(cfg.reward.pnl_scale)
    w = float(cfg.reward.quote_anchor_weight)
    A = float(cfg.clients.base_arrival_rate)
    k = float(cfg.clients.demand_elasticity)

    if h is None:
        h_star = solve_fixed_point(cfg, ref)
        h = np.full(d, h_star, dtype=float)
    else:
        h = np.asarray(h, dtype=float)

    # own-bond curvature gamma_i^0 = P*[2w + A*rho*k*exp(-k*h)*(2 - k*h)]  (no lambda_q;
    # the inventory-risk curvature is now the explicit zeta'*G*Sigma*G term).
    glft = A * ref.rho * k * np.exp(-k * h) * (2.0 - k * h)
    gamma0 = P * (2.0 * w + glft)
    eps = np.array([epsilon_of(cfg, float(hi), ref) for hi in h], dtype=float)

    features = universe.features.numpy()  # [d, 4]: duration, rating, size, sector
    duration = features[:, 0].astype(float)
    if sigma is None:
        mean_dur = float(duration.mean()) if duration.mean() != 0 else 1.0
        sigma = float(cfg.simulator.fundamental_vol) * (duration / mean_dur)
    else:
        sigma = np.asarray(sigma, dtype=float)
    if g is None:
        horizon = float(cfg.simulator.horizon)
        g = float(inventory_gain) * horizon * A * np.exp(-k * h)
    else:
        g = np.asarray(g, dtype=float)

    return PerBondConstants(h=h, gamma0=gamma0, epsilon=eps, sigma=sigma, g=g, beta=P)


def covariance(universe: BondUniverse, sigma: np.ndarray) -> np.ndarray:
    """Cross-bond covariance ``Sigma = D_sigma * C * D_sigma`` (1.5 §0)."""
    C = universe.corr.numpy().astype(float)
    Ds = np.diag(np.asarray(sigma, dtype=float))
    return Ds @ C @ Ds


def curvature_matrix(cfg: Config, pbc: PerBondConstants, universe: BondUniverse) -> np.ndarray:
    """The objective Hessian ``Gamma = D_gamma + zeta'*G*Sigma*G`` (1.5 §2.2)."""
    Sigma = covariance(universe, pbc.sigma)
    G = np.diag(pbc.g)
    return np.diag(pbc.gamma0) + _zeta_prime(cfg) * (G @ Sigma @ G)


def modulus_matrix(cfg: Config, pbc: PerBondConstants, universe: BondUniverse) -> np.ndarray:
    """The dense modulus matrix ``M = beta * Gamma^{-1} * E`` (1.5 §2.3, boxed)."""
    Gamma = curvature_matrix(cfg, pbc, universe)
    E = np.diag(pbc.epsilon)
    return pbc.beta * np.linalg.solve(Gamma, E)


def spectral_radius(M: np.ndarray) -> float:
    """Spectral radius ``rho(M) = max_j |lambda_j(M)|`` (stability iff ``< 1``)."""
    return float(np.max(np.abs(np.linalg.eigvals(M))))


@dataclass
class FactorModulus:
    """Multi-bond stability summary (1.5 §3)."""

    n_bonds: int
    rho: float  # spectral radius of M (stability iff < 1)
    stable: bool
    m_scalar_max: float  # max_i m_1^(i): the uncorrelated-book boundary (1.5 §3.1)
    chi: float  # top-factor amplification rho(M) / m_scalar_max
    top_eigenvector: np.ndarray  # dominant eigenvector of M (the fragile mode)
    market_alignment: float  # |<top eigenvector, 1/sqrt(d)>| (alignment with the market mode)


def factor_modulus(
    cfg: Config,
    universe: Optional[BondUniverse] = None,
    h: Optional[np.ndarray] = None,
    mispricing: Optional[float] = None,
    liquidity_ratio: float = 1.0,
    sigma: Optional[np.ndarray] = None,
    g: Optional[np.ndarray] = None,
    inventory_gain: float = 1.0,
) -> FactorModulus:
    """Compute the multi-bond boundary ``rho(M) < 1`` for ``cfg`` (1.5 §3).

    Builds the dense ``M`` (exact reference), its spectral radius, the dominant
    eigenvector and its alignment with the all-ones market mode, and the
    amplification ``chi = rho(M)/max_i m_1^(i)`` relative to the uncorrelated book.
    """
    if universe is None:
        universe = BondUniverse(cfg.bonds, seed=cfg.seed)
    pbc = per_bond_constants(
        cfg, universe, h=h, mispricing=mispricing, liquidity_ratio=liquidity_ratio,
        sigma=sigma, g=g, inventory_gain=inventory_gain,
    )
    M = modulus_matrix(cfg, pbc, universe)
    eigvals, eigvecs = np.linalg.eig(M)
    idx = int(np.argmax(np.abs(eigvals)))
    rho = float(abs(eigvals[idx]))
    vec = np.real(eigvecs[:, idx])
    vec = vec / (np.linalg.norm(vec) + 1e-12)
    d = universe.n_bonds
    ones = np.ones(d) / math.sqrt(d)
    m_scalar = pbc.beta * pbc.epsilon / pbc.gamma0  # per-bond scalar moduli m_1^(i)
    m_scalar_max = float(np.max(m_scalar))
    return FactorModulus(
        n_bonds=d,
        rho=rho,
        stable=rho < 1.0,
        m_scalar_max=m_scalar_max,
        chi=(rho / m_scalar_max if m_scalar_max > 0 else math.inf),
        top_eigenvector=vec,
        market_alignment=float(abs(vec @ ones)),
    )


# --------------------------------------------------------------------------- #
# Factor reduction: O(d*k^2) Woodbury (1.5 §4)                                 #
# --------------------------------------------------------------------------- #
def factorize_corr(corr: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Factor the correlation matrix ``C ~ B*diag(Lambda)*B^T + diag(D_idio)`` (1.5 §4).

    Returns the top-``k`` orthonormal loadings ``B`` (``d x k``), factor variances
    ``Lambda`` (``k``), and the idiosyncratic residual diagonal ``D_idio`` (``d``)
    so that ``diag(B*diag(Lambda)*B^T) + D_idio == diag(C)``.  ``k = 1 + n_sectors``
    for the `bonds.py` covariance.
    """
    C = np.asarray(corr, dtype=float)
    evals, evecs = np.linalg.eigh(C)  # ascending
    order = np.argsort(evals)[::-1]
    evals, evecs = evals[order], evecs[:, order]
    k = int(min(max(k, 0), C.shape[0]))
    B = evecs[:, :k]
    Lambda = evals[:k]
    Ck = (B * Lambda) @ B.T
    D_idio = np.diag(C) - np.diag(Ck)
    return B, Lambda, D_idio


def gamma_inverse_woodbury(D_diag: np.ndarray, U: np.ndarray, Lambda: np.ndarray) -> np.ndarray:
    """Invert ``Gamma = diag(D_diag) + U*diag(Lambda)*U^T`` by Woodbury (1.5 §4).

    ``Gamma^{-1} = D^{-1} - D^{-1}U (Lambda^{-1} + U^T D^{-1} U)^{-1} U^T D^{-1}`` --
    ``O(d*k^2 + k^3)`` instead of ``O(d^3)``.
    """
    Dinv = 1.0 / np.asarray(D_diag, dtype=float)
    U = np.asarray(U, dtype=float)
    DinvU = Dinv[:, None] * U  # d x k
    k = U.shape[1]
    inner = np.diag(1.0 / np.asarray(Lambda, dtype=float)) + U.T @ DinvU  # k x k
    correction = DinvU @ np.linalg.solve(inner, DinvU.T)
    return np.diag(Dinv) - correction


def spectral_radius_woodbury(
    cfg: Config,
    universe: BondUniverse,
    k: int,
    pbc: Optional[PerBondConstants] = None,
    mispricing: Optional[float] = None,
    liquidity_ratio: float = 1.0,
) -> float:
    """``rho(M)`` via the ``O(d*k^2)`` factor reduction (1.5 §4).

    Truncates ``C`` to ``k`` factors, builds ``Gamma = D + U*Lambda*U^T`` with
    ``D = D_gamma + zeta'*G*D_sigma*D_idio*D_sigma*G`` and
    ``U = sqrt(zeta')*G*D_sigma*B``, inverts by Woodbury and returns
    ``rho(M_k) = rho(beta*Gamma_k^{-1}*E)`` via power iteration on ``M_k``.
    """
    if pbc is None:
        pbc = per_bond_constants(cfg, universe, mispricing=mispricing, liquidity_ratio=liquidity_ratio)
    C = universe.corr.numpy().astype(float)
    B, Lambda, D_idio = factorize_corr(C, k)
    zp = _zeta_prime(cfg)
    Gs = pbc.g * pbc.sigma  # diag(G*D_sigma)
    D = pbc.gamma0 + zp * (Gs ** 2) * D_idio  # diagonal part
    U = math.sqrt(max(zp, 0.0)) * (Gs[:, None] * B)  # d x k
    Ginv = gamma_inverse_woodbury(D, U, Lambda)
    M = pbc.beta * (Ginv * pbc.epsilon[None, :])  # M = beta * Ginv * diag(epsilon)
    # power iteration for the dominant eigenvalue magnitude
    d = universe.n_bonds
    v = np.ones(d) / math.sqrt(d)
    lam = 0.0
    for _ in range(200):
        w = M @ v
        nw = np.linalg.norm(w)
        if nw < 1e-15:
            return 0.0
        v = w / nw
        lam_new = float(v @ (M @ v))
        if abs(lam_new - lam) < 1e-12:
            lam = lam_new
            break
        lam = lam_new
    return abs(lam)


# --------------------------------------------------------------------------- #
# Truncation error bound (Theorem 1, 1.5 §5)                                   #
# --------------------------------------------------------------------------- #
@dataclass
class TruncationBound:
    """Factor-truncation error bound (1.5 §5, Theorem 1)."""

    k: int
    residual_variance: float  # ||R||_2 = lambda_{k+1}(C)
    m_error_bound: float  # bound on ||M - M_k||_2
    rho_error_measured: float  # actual |rho(M) - rho(M_k)| (dense vs Woodbury-k)


def truncation_error_bound(
    cfg: Config,
    universe: BondUniverse,
    k: int,
    mispricing: Optional[float] = None,
    liquidity_ratio: float = 1.0,
) -> TruncationBound:
    """The ``O(lambda_{k+1}(C))`` error bound of Theorem 1 (1.5 §5).

    Reports the residual factor variance ``lambda_{k+1}(C)``, the closed-form bound
    on ``||M - M_k||_2``, and the *measured* ``|rho(M) - rho(M_k)|`` (dense full-rank
    vs ``k``-factor Woodbury) for cross-checking.
    """
    pbc = per_bond_constants(cfg, universe, mispricing=mispricing, liquidity_ratio=liquidity_ratio)
    C = universe.corr.numpy().astype(float)
    evals = np.sort(np.linalg.eigvalsh(C))[::-1]
    resid = float(evals[k]) if k < len(evals) else 0.0  # lambda_{k+1}(C)

    P = pbc.beta
    eps_max = float(np.max(pbc.epsilon))
    zp = _zeta_prime(cfg)
    g_max = float(np.max(np.abs(pbc.g)))
    sigma_max = float(np.max(pbc.sigma))
    gamma_min = float(np.min(pbc.gamma0))
    # ||M - M_k|| <= (P*eps_max*zeta'*g_max^2*sigma_max^2 / gamma_min^2) * ||R||
    bound = (P * eps_max * zp * g_max ** 2 * sigma_max ** 2 / gamma_min ** 2) * resid

    rho_full = spectral_radius_woodbury(cfg, universe, k=len(evals), pbc=pbc)
    rho_k = spectral_radius_woodbury(cfg, universe, k=k, pbc=pbc)
    return TruncationBound(
        k=k,
        residual_variance=resid,
        m_error_bound=bound,
        rho_error_measured=abs(rho_full - rho_k),
    )


# --------------------------------------------------------------------------- #
# Diversification law and 1.3 x 1.5 composition (1.5 §7)                       #
# --------------------------------------------------------------------------- #
def sweep_global_factor(
    cfg: Config,
    global_factors: List[float],
    seed: Optional[int] = None,
    mispricing: Optional[float] = None,
    liquidity_ratio: float = 1.0,
    inventory_gain: float = 1.0,
) -> List[FactorModulus]:
    """``rho(M)`` vs the global-factor loading (the diversification law, 1.5 §7.1).

    Each point rebuilds the universe at a different ``bonds.global_factor``.  With
    the clean priced-risk ``M`` (see the module note), correlation enters through the
    inventory curvature, so ``rho(M)`` *falls* as ``global_factor`` rises.
    """
    import copy

    out: List[FactorModulus] = []
    for gf in global_factors:
        cfg_g = copy.deepcopy(cfg)
        cfg_g.bonds.global_factor = float(gf)
        universe = BondUniverse(cfg_g.bonds, seed=int(cfg_g.seed if seed is None else seed))
        out.append(
            factor_modulus(
                cfg_g, universe=universe, mispricing=mispricing,
                liquidity_ratio=liquidity_ratio, inventory_gain=inventory_gain,
            )
        )
    return out


def systemic_spectral_radius(m_1: float, n_eff: float, chi: float) -> float:
    """Composed 1.3 x 1.5 systemic modulus ``N_eff * chi * m_1`` (1.5 §7.2).

    The maximally fragile eigenvector is the product of 1.3's common-dealer mode and
    1.5's market-factor mode; the boundary tightens by both the effective dealer
    count ``N_eff`` and the top-factor amplification ``chi``.
    """
    return float(n_eff) * float(chi) * float(m_1)
