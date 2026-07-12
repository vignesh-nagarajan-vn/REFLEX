"""Dealer quoting policies ``pi_phi``.

A policy maps the **observable** market state to quotes (a positive half-spread
and an inventory-driven skew) per bond.  Two concrete variants share one
interface:

* :class:`LinearPolicy` -- half-spread is ``softplus`` of an affine function of
  the observable features; the *bias* term sets the central half-spread, which
  is the dominant coordinate of the policy->distribution iterate map studied in
  the convergence experiment.
* :class:`MLPPolicy` -- the same interface backed by a small MLP for capacity.

Both consume only observable features (inventory, recent signed flow, recent
gross volume); they never see the latent fundamental or liquidity.  Both expose
:meth:`flatten`/:meth:`load_flat` so the RRM loop can measure iterate distances
``||phi_{k+1} - phi_k||`` in a flat parameter space, and remain fully
differentiable so policy optimization can backpropagate pathwise gradients
through the learned operator ``T_theta``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..config import PolicyConfig
from ..types import MarketState, Quotes


def _inverse_softplus(y: float) -> float:
    """Return ``x`` such that ``softplus(x) == y`` (for initialising the bias)."""
    import math

    y = max(float(y), 1e-4)
    # softplus(x) = log(1 + e^x)  =>  x = log(e^y - 1)
    return math.log(math.expm1(y))


class DealerPolicy(ABC, nn.Module):
    """Abstract dealer policy.

    Subclasses implement :meth:`quote`.  The base class provides flat-parameter
    serialisation used for iterate-distance computations in RRM.
    """

    #: registered buffer with per-feature scales for conditioning the linear map.
    feat_scale: Tensor

    def __init__(self, cfg: PolicyConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.max_half_spread = float(cfg.max_half_spread)
        # Rough magnitudes of (inventory, signed flow, gross volume) to keep the
        # affine map well-conditioned; these are fixed, not learned.
        self.register_buffer("feat_scale", torch.tensor([1.0, 1.0, 5.0]))

    # ------------------------------------------------------------------ #
    # Interface                                                           #
    # ------------------------------------------------------------------ #
    @abstractmethod
    def quote(self, state: MarketState) -> Quotes:
        """Return :class:`Quotes` for every bond given the current state."""
        raise NotImplementedError

    def _scaled_features(self, state: MarketState) -> Tensor:
        """Observable features ``[N, 3]`` divided by their reference scales."""
        return state.observable_features() / self.feat_scale

    def _finish(self, raw_h: Tensor, skew: Tensor) -> Quotes:
        """Map a raw pre-activation to a positive, capped half-spread."""
        half_spread = F.softplus(raw_h).clamp(max=self.max_half_spread)
        return Quotes(half_spread=half_spread, skew=skew)

    # ------------------------------------------------------------------ #
    # Flat-parameter (de)serialisation                                    #
    # ------------------------------------------------------------------ #
    def flatten(self) -> Tensor:
        """Return all trainable parameters concatenated into one 1-D tensor."""
        return torch.cat([p.detach().reshape(-1) for p in self.parameters()])

    @torch.no_grad()
    def load_flat(self, flat: Tensor) -> None:
        """Load parameters from a flat tensor produced by :meth:`flatten`."""
        offset = 0
        for p in self.parameters():
            numel = p.numel()
            p.copy_(flat[offset : offset + numel].reshape(p.shape))
            offset += numel
        if offset != flat.numel():
            raise ValueError(
                f"flat tensor has {flat.numel()} elements but policy expects {offset}"
            )

    @property
    def n_params(self) -> int:
        """Total number of trainable scalar parameters."""
        return sum(p.numel() for p in self.parameters())


class LinearPolicy(DealerPolicy):
    """Affine-in-features half-spread (via softplus) with inventory skew."""

    def __init__(self, cfg: PolicyConfig) -> None:
        super().__init__(cfg)
        # Half-spread = softplus(w_h . features + b_h); b_h sets the central level.
        self.w_h = nn.Parameter(0.01 * torch.randn(3))
        self.b_h = nn.Parameter(torch.tensor(_inverse_softplus(cfg.init_half_spread)))
        # Skew leans against inventory: skew = w_skew * inventory + b_skew.
        self.w_skew = nn.Parameter(torch.zeros(1))
        self.b_skew = nn.Parameter(torch.zeros(1))

    def quote(self, state: MarketState) -> Quotes:
        feats = self._scaled_features(state)  # [N, 3]
        raw_h = feats @ self.w_h + self.b_h  # [N]
        skew = self.w_skew * state.inventory + self.b_skew  # [N]
        return self._finish(raw_h, skew)


class MLPPolicy(DealerPolicy):
    """Small MLP mapping observable features to (half-spread, skew)."""

    def __init__(self, cfg: PolicyConfig) -> None:
        super().__init__(cfg)
        hidden = int(cfg.hidden)
        self.net = nn.Sequential(
            nn.Linear(3, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 2),
        )
        # Bias the half-spread output toward the configured initial level.
        with torch.no_grad():
            self.net[-1].bias[0] = _inverse_softplus(cfg.init_half_spread)
            self.net[-1].bias[1] = 0.0

    def quote(self, state: MarketState) -> Quotes:
        feats = self._scaled_features(state)  # [N, 3]
        out = self.net(feats)  # [N, 2]
        raw_h = out[:, 0]
        skew = out[:, 1]
        return self._finish(raw_h, skew)


def build_policy(cfg: PolicyConfig) -> DealerPolicy:
    """Factory: construct the policy named by ``cfg.type``."""
    if cfg.type == "linear":
        return LinearPolicy(cfg)
    if cfg.type == "mlp":
        return MLPPolicy(cfg)
    raise ValueError(f"unknown policy type {cfg.type!r} (expected 'linear' or 'mlp')")
