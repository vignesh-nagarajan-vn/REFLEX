"""Stability diagnostics / regularizers for the dealer objective.

These quantify how 'aggressive' or concentrated a policy is and, optionally,
penalise it.  By default every weight in :class:`StabilityConfig` is 0, so these
are pure diagnostics; turning a weight up adds the corresponding term to the
policy's optimization objective (as a penalty), which can be used to study
whether explicitly discouraging toxic exposure or quote concentration stabilises
the retraining loop.

Terms:

* **entropy floor** -- encourages spreading volume across bonds (negative HHI
  acts as a concentration penalty; an entropy bonus rewards diversification);
* **HHI** -- Herfindahl concentration of quoted tightness across bonds;
* **toxicity exposure** -- predicted adverse-selection magnitude;
* **Lipschitz / spectral penalty** -- sensitivity of the operator's predicted
  response to a perturbation of the policy summary (a local performativity
  proxy): we perturb the summary by ``lipschitz_eps`` and measure the change in
  the predicted next-state/P&L means.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from ..config import StabilityConfig
from ..types import MarketState, Quotes


@dataclass
class StabilityTerms:
    """Diagnostic stability quantities (each a scalar tensor)."""

    entropy: Tensor
    hhi: Tensor
    toxicity_exposure: Tensor
    lipschitz: Tensor
    penalty: Tensor  # weighted sum to ADD to the loss (subtract from objective)


def _quote_shares(quotes: Quotes) -> Tensor:
    """Normalised 'tightness' weights across bonds (tighter quote => larger share)."""
    tightness = torch.softmax(-quotes.half_spread, dim=-1)
    return tightness


def compute_stability(
    state: MarketState,
    quotes: Quotes,
    cfg: StabilityConfig,
    operator=None,
    policy=None,
    summary: Tensor | None = None,
) -> StabilityTerms:
    """Compute stability diagnostics for a state/quotes pair.

    The Lipschitz term is only computed when ``operator``, ``summary`` are
    provided; otherwise it is zero.
    """
    shares = _quote_shares(quotes)
    eps = 1e-9
    entropy = -(shares * (shares + eps).log()).sum()
    hhi = (shares ** 2).sum()
    # Toxicity exposure proxy: tighter-than-reference quoting invites toxic flow.
    toxicity_exposure = torch.relu(1.0 - quotes.half_spread).pow(2).mean()

    lipschitz = state.inventory.new_zeros(())
    if operator is not None and summary is not None:
        with torch.no_grad():
            base = operator.predict(state, quotes, summary).mean
            perturbed_summary = summary + cfg.lipschitz_eps
            pert = operator.predict(state, quotes, perturbed_summary).mean
            lipschitz = (pert - base).abs().mean() / max(cfg.lipschitz_eps, 1e-9)

    penalty = (
        -cfg.entropy_w * entropy  # higher entropy is good -> negative penalty
        + cfg.hhi_w * hhi
        + cfg.toxicity_w * toxicity_exposure
        + cfg.lipschitz_w * lipschitz
    )
    return StabilityTerms(
        entropy=entropy,
        hhi=hhi,
        toxicity_exposure=toxicity_exposure,
        lipschitz=lipschitz,
        penalty=penalty,
    )
