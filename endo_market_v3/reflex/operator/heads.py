"""Output distribution heads for the Market Response Operator.

Each head maps a hidden feature vector to a distribution over the operator's
``D``-dimensional target (the next-state observable deltas concatenated with the
P&L component channels).  All heads expose the same trio used by the rest of the
codebase:

* the ``forward`` call returns a ``torch.distributions.Distribution``;
* ``.rsample()`` on that distribution gives **reparameterised** samples for
  pathwise gradients (where supported);
* ``.log_prob(target)`` gives the per-row log-likelihood for MLE fitting.

The default :class:`GaussianHead` is diagonal, which is fully reparameterisable
and is all we need: by predicting the *conditional mean* of each P&L channel
(notably the adverse-selection loss) the operator captures the economically
relevant signal without requiring a full covariance.  A full-covariance Gaussian
and a :class:`MixtureHead` are provided for capacity / multimodality.
"""

from __future__ import annotations

import torch
import torch.distributions as dist
from torch import Tensor, nn


class GaussianHead(nn.Module):
    """Gaussian output head (diagonal by default, optional full covariance).

    Parameters
    ----------
    in_dim:
        Input (hidden) dimension.
    out_dim:
        Dimension ``D`` of the predicted target.
    min_logstd, max_logstd:
        Clamp range for the predicted log standard deviations (numerical safety).
    full_cov:
        If ``True`` predict a lower-triangular Cholesky factor and return a
        :class:`~torch.distributions.MultivariateNormal`; otherwise predict a
        diagonal and return an independent ``Normal``.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        min_logstd: float = -4.0,
        max_logstd: float = 2.0,
        full_cov: bool = False,
    ) -> None:
        super().__init__()
        self.out_dim = int(out_dim)
        self.min_logstd = float(min_logstd)
        self.max_logstd = float(max_logstd)
        self.full_cov = bool(full_cov)

        self.mean = nn.Linear(in_dim, out_dim)
        self.logstd = nn.Linear(in_dim, out_dim)
        if self.full_cov:
            # Predict the strictly-lower-triangular off-diagonal entries.
            self._n_offdiag = out_dim * (out_dim - 1) // 2
            self.offdiag = nn.Linear(in_dim, self._n_offdiag)
            tril_idx = torch.tril_indices(out_dim, out_dim, offset=-1)
            self.register_buffer("_tril_row", tril_idx[0])
            self.register_buffer("_tril_col", tril_idx[1])

    def forward(self, x: Tensor) -> dist.Distribution:
        """Map hidden features ``x`` to a distribution over the target."""
        mean = self.mean(x)
        log_std = self.logstd(x).clamp(self.min_logstd, self.max_logstd)
        std = torch.exp(log_std)
        if not self.full_cov:
            return dist.Independent(dist.Normal(mean, std), 1)

        # Build a lower-triangular Cholesky factor with positive diagonal.
        batch = mean.shape[:-1]
        scale_tril = torch.diag_embed(std)
        if self._n_offdiag > 0:
            off = self.offdiag(x)
            scale_tril[..., self._tril_row, self._tril_col] = off
        return dist.MultivariateNormal(mean, scale_tril=scale_tril)

    def rsample(self, x: Tensor, generator: torch.Generator | None = None) -> Tensor:
        """Reparameterised sample of shape ``[..., D]`` (generator-controlled).

        For the diagonal case the reparameterisation ``mean + std * eps`` is done
        explicitly so a :class:`torch.Generator` makes rollouts reproducible.  The
        full-covariance case defers to the distribution's ``rsample`` (which does
        not accept a generator).
        """
        mean = self.mean(x)
        log_std = self.logstd(x).clamp(self.min_logstd, self.max_logstd)
        std = torch.exp(log_std)
        if not self.full_cov:
            eps = torch.randn(mean.shape, generator=generator, dtype=mean.dtype, device=mean.device)
            return mean + std * eps
        return self.forward(x).rsample()


class MixtureHead(nn.Module):
    """Mixture-of-diagonal-Gaussians head for multimodal predictions.

    ``forward`` returns a :class:`~torch.distributions.MixtureSameFamily`.  That
    distribution supports ``log_prob`` (used for fitting) and ``sample`` but not
    ``rsample``; :meth:`rsample` here provides a pathwise-friendly sample by
    selecting a component with a detached categorical draw and then
    reparameterising within it (a biased but practical relaxation).  The default
    operator uses :class:`GaussianHead`, so this path is opt-in.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        n_components: int = 3,
        min_logstd: float = -4.0,
        max_logstd: float = 2.0,
    ) -> None:
        super().__init__()
        self.out_dim = int(out_dim)
        self.k = int(n_components)
        self.min_logstd = float(min_logstd)
        self.max_logstd = float(max_logstd)

        self.logits = nn.Linear(in_dim, self.k)
        self.mean = nn.Linear(in_dim, self.k * out_dim)
        self.logstd = nn.Linear(in_dim, self.k * out_dim)

    def _params(self, x: Tensor):
        batch = x.shape[:-1]
        logits = self.logits(x)  # [..., K]
        mean = self.mean(x).reshape(*batch, self.k, self.out_dim)
        log_std = self.logstd(x).clamp(self.min_logstd, self.max_logstd)
        std = torch.exp(log_std).reshape(*batch, self.k, self.out_dim)
        return logits, mean, std

    def forward(self, x: Tensor) -> dist.Distribution:
        """Return a mixture-of-diagonal-Gaussians distribution."""
        logits, mean, std = self._params(x)
        mix = dist.Categorical(logits=logits)
        comp = dist.Independent(dist.Normal(mean, std), 1)
        return dist.MixtureSameFamily(mix, comp)

    def rsample(self, x: Tensor, generator: torch.Generator | None = None) -> Tensor:
        """Pathwise-friendly sample (detached component choice + reparam draw)."""
        logits, mean, std = self._params(x)
        # Detached categorical component selection.
        probs = torch.softmax(logits, dim=-1)
        idx = torch.multinomial(
            probs.reshape(-1, self.k), 1, generator=generator
        ).reshape(probs.shape[:-1])
        idx_exp = idx[..., None, None].expand(*idx.shape, 1, self.out_dim)
        chosen_mean = torch.gather(mean, -2, idx_exp).squeeze(-2)
        chosen_std = torch.gather(std, -2, idx_exp).squeeze(-2)
        eps = torch.randn(
            chosen_mean.shape, generator=generator, dtype=chosen_mean.dtype, device=chosen_mean.device
        )
        return chosen_mean + chosen_std * eps


def build_head(
    head_type: str,
    in_dim: int,
    out_dim: int,
    *,
    n_mixture: int = 3,
    min_logstd: float = -4.0,
    max_logstd: float = 2.0,
) -> nn.Module:
    """Factory for the operator output head named by ``head_type``."""
    if head_type in ("gaussian", "gaussian_diag"):
        return GaussianHead(in_dim, out_dim, min_logstd, max_logstd, full_cov=False)
    if head_type == "gaussian_full":
        return GaussianHead(in_dim, out_dim, min_logstd, max_logstd, full_cov=True)
    if head_type == "mixture":
        return MixtureHead(in_dim, out_dim, n_mixture, min_logstd, max_logstd)
    raise ValueError(f"unknown head_type {head_type!r}")
