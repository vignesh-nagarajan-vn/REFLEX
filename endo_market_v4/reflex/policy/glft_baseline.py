"""Closed-form GLFT/Avellaneda-Stoikov baseline quoting policy.

A *non-learned* sanity anchor for the learned policies: it quotes the constant
half-spread given by the closed-form theory of ``theory/01`` (and, in
``mode="optimum"``, ``theory/02``):

* ``mode="stable_point"`` -- the self-consistent RRM fixed point ``h_SP = h*``
  (1.1 §4): the spread that is its own frozen-environment best response.  This
  is where blind repeated retraining lands *if it converges*.
* ``mode="optimum"`` -- the performative optimum ``h_PO`` (1.2 §1-§2): the
  maximiser of the true objective ``Phi(h) = J(h; tau(h))``, i.e. what the
  dealer *should* quote once the distribution response ``d tau/dh`` is priced
  in.  ``h_SP - h_PO`` is the echo-chamber decision gap.

Both are pure functions of the config (evaluated once at construction), so the
policy is deterministic, gradient-free and costless at quote time.  Zero skew,
matching the zero-skew restriction (A1) under which the closed forms are
derived.  Use it to (a) validate the learned policies' converged spreads, and
(b) measure realised P&L at the theory's own solution.

The theory imports are deliberately *lazy* (inside ``__init__``): the
estimator/theory modules themselves import :mod:`reflex.policy`, so a
module-level import here would create a cycle.
"""

from __future__ import annotations

import torch

from ..config import Config
from ..types import MarketState, Quotes
from .dealer_policy import DealerPolicy


class GLFTBaselinePolicy(DealerPolicy):
    """Constant closed-form half-spread from the analytic theory (zero skew)."""

    def __init__(self, cfg: Config, mode: str = "stable_point") -> None:
        if not isinstance(cfg, Config):
            raise TypeError(
                "GLFTBaselinePolicy needs the full Config (the closed forms read "
                "clients/reward/simulator sections), not just PolicyConfig"
            )
        super().__init__(cfg.policy)
        if mode not in ("stable_point", "optimum"):
            raise ValueError(f"unknown mode {mode!r} (expected 'stable_point' or 'optimum')")
        self.mode = mode

        # Lazy imports -- see module docstring (cycle via estimators -> policy).
        from ..theory.analytic_boundary import reference_state, solve_fixed_point

        ref = reference_state(cfg)
        if mode == "stable_point":
            h = solve_fixed_point(cfg, ref)
        else:
            from ..theory.perfgd import solve_performative_optimum

            h = solve_performative_optimum(cfg, ref)
        # Clamp into the policy's admissible range.  The exact solve is kept as a
        # python float (full double precision, for theory comparisons); the
        # float32 buffer is what quoting uses, so the policy serialises/flattens
        # like every other DealerPolicy (n_params=0).
        h = float(min(max(h, 0.0), self.max_half_spread))
        self._h_exact = h
        self.register_buffer("h_const", torch.tensor(h))

    @property
    def half_spread_value(self) -> float:
        """The constant closed-form half-spread this policy quotes (exact solve)."""
        return self._h_exact

    def quote(self, state: MarketState) -> Quotes:
        n = state.n_bonds
        h = self.h_const.expand(n).clone()
        skew = torch.zeros(n)
        return Quotes(half_spread=h, skew=skew)


def build_glft_baseline(cfg: Config, mode: str = "stable_point") -> GLFTBaselinePolicy:
    """Factory mirroring :func:`reflex.policy.build_policy` for the baseline."""
    return GLFTBaselinePolicy(cfg, mode=mode)
