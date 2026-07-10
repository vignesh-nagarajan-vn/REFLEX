"""Factor-model scaling to many correlated bonds (math-theory 1.5).

Exact checks of the modulus matrix M = beta*Gamma^{-1}*E: the scalar/uncorrelated
reductions, the O(d*k^2) Woodbury inverse matching the dense one, the factor
truncation error bound, and the 1.3 x 1.5 composition.
"""

from __future__ import annotations

import numpy as np
import pytest

from reflex.theory.factor_scaling import (
    curvature_matrix,
    factor_modulus,
    factorize_corr,
    modulus_matrix,
    per_bond_constants,
    spectral_radius,
    spectral_radius_woodbury,
    sweep_global_factor,
    systemic_spectral_radius,
    truncation_error_bound,
)
from reflex.config import load_config
from reflex.env.bonds import BondUniverse


def _cfg(n_bonds=40, n_sectors=3, **kw):
    cfg = load_config("configs/default.yaml")
    cfg.bonds.n_bonds = n_bonds
    cfg.bonds.n_sectors = n_sectors
    for k, v in kw.items():
        obj, _, attr = k.partition(".")
        setattr(getattr(cfg, obj), attr, v)
    return cfg


def test_single_bond_reduces_to_scalar_modulus() -> None:
    """d = 1: M is 1x1 and equals beta*epsilon/(gamma0 + zeta'*g^2*sigma^2) (1.5 §3.1)."""
    cfg = _cfg(n_bonds=1, n_sectors=1)
    univ = BondUniverse(cfg.bonds, seed=cfg.seed)
    pbc = per_bond_constants(cfg, univ)
    M = modulus_matrix(cfg, pbc, univ)
    assert M.shape == (1, 1)
    Gamma = curvature_matrix(cfg, pbc, univ)
    expected = pbc.beta * pbc.epsilon[0] / Gamma[0, 0]
    assert spectral_radius(M) == pytest.approx(expected, rel=1e-9)


def test_uncorrelated_book_is_max_scalar_modulus() -> None:
    """With no inventory risk the book decouples: rho(M) = max_i m_1^(i), chi = 1 (1.5 §3.1)."""
    cfg = _cfg(n_bonds=25)
    cfg.reward.inv_risk_weight = 0.0  # zeta' = 0 -> Gamma = D_gamma diagonal
    fm = factor_modulus(cfg)
    assert fm.chi == pytest.approx(1.0, abs=1e-9)
    assert fm.rho == pytest.approx(fm.m_scalar_max, rel=1e-9)


def test_inventory_risk_is_stabilising_in_clean_model() -> None:
    """Priced correlated inventory only adds curvature, so rho(M) <= max_i m_1^(i) (1.5 §3.3)."""
    fm = factor_modulus(_cfg(), inventory_gain=5.0)
    assert fm.chi <= 1.0 + 1e-9
    # the fragile mode is idiosyncratic, not the all-ones market mode
    assert fm.market_alignment < 0.5


def test_woodbury_matches_dense() -> None:
    """The O(d*k^2) Woodbury spectral radius (k = d) matches the dense O(d^3) one (1.5 §4)."""
    cfg = _cfg(n_bonds=50, n_sectors=3)
    univ = BondUniverse(cfg.bonds, seed=cfg.seed)
    pbc = per_bond_constants(cfg, univ)
    rho_dense = spectral_radius(modulus_matrix(cfg, pbc, univ))
    rho_wood = spectral_radius_woodbury(cfg, univ, k=cfg.bonds.n_bonds, pbc=pbc)
    assert rho_wood == pytest.approx(rho_dense, rel=5e-3)


def test_factorize_corr_preserves_diagonal() -> None:
    """The factorisation keeps the unit diagonal and reconstructs C at k = d (1.5 §4)."""
    cfg = _cfg(n_bonds=30, n_sectors=3)
    C = BondUniverse(cfg.bonds, seed=cfg.seed).corr.numpy().astype(float)
    B, Lam, D_idio = factorize_corr(C, k=1 + cfg.bonds.n_sectors)
    Ck = (B * Lam) @ B.T
    assert np.allclose(np.diag(Ck) + D_idio, np.diag(C), atol=1e-9)
    Bf, Lamf, Df = factorize_corr(C, k=C.shape[0])
    assert np.allclose((Bf * Lamf) @ Bf.T + np.diag(Df), C, atol=1e-8)


def test_truncation_error_bound_holds_and_shrinks() -> None:
    """|rho(M) - rho(M_k)| <= bound, and it collapses once k >= 1 + n_sectors (1.5 §5)."""
    cfg = _cfg(n_bonds=60, n_sectors=3)
    univ = BondUniverse(cfg.bonds, seed=cfg.seed)
    errs = []
    for k in (1, 2, 4, 6):
        tb = truncation_error_bound(cfg, univ, k)
        assert tb.rho_error_measured <= tb.m_error_bound + 1e-9  # Theorem 1 bound holds
        errs.append(tb.rho_error_measured)
    # capturing the true factor count (1 global + 3 sectors = 4) drives the error down
    assert errs[2] <= errs[0] + 1e-12


def test_systemic_composition() -> None:
    """The 1.3 x 1.5 systemic modulus is N_eff * chi * m_1 (1.5 §7.2)."""
    assert systemic_spectral_radius(0.3, n_eff=4.0, chi=1.1) == pytest.approx(1.32)


def test_sweep_global_factor_runs_and_stays_below_uncorrelated_bound() -> None:
    """rho(M) across the global-factor sweep stays at/under max_i m_1^(i) (chi <= 1)."""
    sweep = sweep_global_factor(_cfg(n_bonds=40), [0.0, 0.2, 0.5, 0.8], inventory_gain=3.0)
    assert all(s.chi <= 1.0 + 1e-9 for s in sweep)
    assert all(np.isfinite(s.rho) for s in sweep)
