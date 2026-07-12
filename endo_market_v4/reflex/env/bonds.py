"""Cross-sectional structure of the tradable bond universe.

The :class:`BondUniverse` fixes, for a given seed, the *static* features of each
bond (duration, credit rating, issue size, sector) and a **cross-bond
correlation matrix** with a block-by-sector structure plus a single global
market factor.  That correlation matrix is the channel through which the latent
liquidity field and the fundamental shocks are coupled across bonds, so that the
market behaves like a genuine portfolio rather than ``N`` independent toys.

Nothing here is trained -- the universe is part of the structural ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from ..config import BondsConfig

# Discrete credit-rating ladder used to map a latent quality factor to a label.
_RATINGS = ["AAA", "AA", "A", "BBB", "BB", "B"]


@dataclass
class BondUniverse:
    """A fixed universe of ``n_bonds`` corporate bonds.

    Parameters
    ----------
    cfg:
        Bond-universe configuration (number of bonds/sectors and correlation
        strengths).
    seed:
        Seed controlling the (otherwise fixed) draw of features and factor
        loadings.  The same seed always yields the same universe.

    Attributes
    ----------
    features:
        Float tensor ``[N, 4]`` with columns ``(duration, rating_idx,
        issue_size, sector_idx)``.  ``rating_idx`` is the index into
        :data:`_RATINGS` (0 = AAA, higher = riskier) and ``sector_idx`` is the
        integer sector label.
    corr:
        Symmetric positive-definite correlation matrix ``[N, N]`` used to couple
        cross-bond shocks.
    chol:
        Lower-triangular Cholesky factor of :attr:`corr`; multiply i.i.d. normal
        draws ``z`` by ``chol`` to obtain correlated shocks ``chol @ z``.
    """

    cfg: BondsConfig
    seed: int = 0

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        n = int(self.cfg.n_bonds)
        n_sectors = max(1, int(self.cfg.n_sectors))

        # --- static per-bond features -------------------------------------- #
        sector = rng.integers(0, n_sectors, size=n)
        duration = rng.uniform(1.0, 12.0, size=n)  # years
        issue_size = rng.uniform(0.2, 2.0, size=n)  # $bn notional outstanding
        # A latent "quality" factor -> discrete rating.  Larger, longer issues
        # skew slightly riskier purely to give the cross-section some texture.
        quality = rng.normal(0.0, 1.0, size=n) + 0.05 * (duration - 6.0)
        rating_idx = np.clip(
            np.digitize(quality, bins=np.linspace(-1.5, 1.5, len(_RATINGS) - 1)),
            0,
            len(_RATINGS) - 1,
        )

        features = np.stack(
            [duration, rating_idx.astype(float), issue_size, sector.astype(float)],
            axis=1,
        )
        self.features: Tensor = torch.as_tensor(features, dtype=torch.float32)
        self._sector = sector

        # --- cross-bond correlation matrix --------------------------------- #
        self.corr: Tensor = self._build_corr(sector, n_sectors)
        # Cholesky factor for generating correlated shocks; jitter guards PD-ness.
        self.chol: Tensor = torch.linalg.cholesky(self.corr)

    # ------------------------------------------------------------------ #
    # Construction helpers                                                #
    # ------------------------------------------------------------------ #
    def _build_corr(self, sector: np.ndarray, n_sectors: int) -> Tensor:
        """Build a block (per-sector) + global-factor correlation matrix."""
        n = sector.shape[0]
        g = float(self.cfg.global_factor)
        w = float(self.cfg.within_sector_corr)

        corr = np.full((n, n), g, dtype=np.float64)  # everyone shares global factor
        same_sector = sector[:, None] == sector[None, :]
        corr[same_sector] += w  # extra correlation inside a sector block
        np.fill_diagonal(corr, 1.0)
        # Clip off-diagonals into a valid range, then project to PD via a small
        # diagonal jitter (cheap and robust for these tiny matrices).
        corr = np.clip(corr, -0.95, 0.95)
        np.fill_diagonal(corr, 1.0)
        corr = 0.5 * (corr + corr.T)
        # Ensure positive-definiteness.
        eigmin = np.linalg.eigvalsh(corr).min()
        if eigmin <= 1e-6:
            corr = corr + (1e-6 - eigmin + 1e-6) * np.eye(n)
            d = np.sqrt(np.diag(corr))
            corr = corr / np.outer(d, d)  # renormalise to unit diagonal
        return torch.as_tensor(corr, dtype=torch.float32)

    # ------------------------------------------------------------------ #
    # Convenience accessors                                               #
    # ------------------------------------------------------------------ #
    @property
    def n_bonds(self) -> int:
        """Number of bonds in the universe."""
        return int(self.cfg.n_bonds)

    @property
    def sector(self) -> Tensor:
        """Integer sector label per bond (``[N]`` long tensor)."""
        return torch.as_tensor(self._sector, dtype=torch.long)

    def correlated_normal(self, generator: torch.Generator | None = None) -> Tensor:
        """Draw a single vector of cross-bond–correlated standard normals.

        Returns ``chol @ z`` where ``z`` is i.i.d. ``N(0, 1)``; the result has
        covariance equal to :attr:`corr`.
        """
        z = torch.randn(self.n_bonds, generator=generator)
        return self.chol @ z
