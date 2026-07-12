"""Fit the Market Response Operator ``T_theta`` by maximum likelihood.

Given transitions collected under deployed policies, we build per-bond
``(feature, target)`` rows and fit the operator to maximise the held-out
log-likelihood of the next-state deltas and P&L channels.  Early stopping on a
validation split guards against overfitting the small dataset.

Two fitting conventions:

* :func:`fit_operator` -- **one deployment** (plain repeated retraining, the v2
  baseline): the policy summary is ~constant across the training rows, so the
  operator cannot identify ``d(prediction)/d(summary)`` -- it is *blind* to the
  distribution response ``dD/dphi``.
* :func:`fit_operator_windowed` -- **a window of recent deployments** (v3
  un-blinding): each deployment's rows carry *its own* policy summary (from the
  policy snapshot deployed at the time), so the summary varies across the
  training set and the operator's summary-dependence -- the learned ``dD/dphi``
  read out by
  :meth:`~reflex.operator.response_operator.MarketResponseOperator.distribution_response`
  -- is identified.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, List, Sequence, Tuple

import torch
from torch import Tensor

from ..config import OperatorConfig
from ..objective.reward import RewardConfig  # noqa: F401  (kept for type discoverability)
from ..operator.response_operator import (
    DELTA_KEYS,
    PNL_KEYS,
    MarketResponseOperator,
)
from ..types import Transition, policy_summary


@dataclass
class FitResult:
    """Diagnostics returned by :func:`fit_operator`."""

    train_nll: List[float] = field(default_factory=list)
    val_nll: List[float] = field(default_factory=list)
    baseline_val_nll: float = float("nan")  # held-out NLL before any training
    best_val_nll: float = float("nan")
    epochs_run: int = 0
    n_rows: int = 0


def transition_rows(transition: Transition, policy) -> Tuple[Tensor, Tensor]:
    """Build ``(features [N, F], target [N, D])`` for a single transition.

    The features are ``[obs(3), quotes(2), policy_summary(3)]`` and the target is
    ``[d_inventory, d_mid, d_flow, d_vol, spread_capture, inventory_pnl,
    adverse_selection_loss]``.  Everything is detached -- these are fixed data.
    """
    s = transition.state
    nxt = transition.next_state
    with torch.no_grad():
        summ = policy_summary(s, policy)
        feats = MarketResponseOperator.build_features(s, transition.quotes, summ)

        deltas = torch.stack(
            [
                nxt.inventory - s.inventory,
                nxt.mid - s.mid,
                nxt.flow_recent - s.flow_recent,
                nxt.vol_recent - s.vol_recent,
            ],
            dim=-1,
        )
        pnl = torch.stack(
            [transition.pnl_components[k] for k in PNL_KEYS], dim=-1
        )
        target = torch.cat([deltas, pnl], dim=-1)
    return feats.detach(), target.detach()


def build_dataset(
    transitions: Sequence[Transition], policy
) -> Tuple[Tensor, Tensor]:
    """Stack all transitions into row tensors ``(X [M, F], Y [M, D])``."""
    feats: List[Tensor] = []
    targs: List[Tensor] = []
    for tr in transitions:
        f, y = transition_rows(tr, policy)
        feats.append(f)
        targs.append(y)
    X = torch.cat(feats, dim=0)
    Y = torch.cat(targs, dim=0)
    return X, Y


def _nll(operator: MarketResponseOperator, X: Tensor, Y: Tensor) -> Tensor:
    """Mean negative log-likelihood over rows."""
    return -operator.log_prob(X, Y).mean()


@dataclass
class DeploymentRecord:
    """One deployment's data: its transitions plus the deployed-policy snapshot.

    The snapshot (a deep copy taken at deployment time) is what makes windowed
    fitting honest: each row's policy summary is computed from the policy that
    actually generated it, not from whatever the policy has since become.
    """

    transitions: List[Transition]
    policy: Any  # deep-copied DealerPolicy snapshot


def fit_operator_windowed(
    operator: MarketResponseOperator,
    deployments: Sequence[DeploymentRecord],
    cfg: OperatorConfig,
    generator: torch.Generator | None = None,
) -> FitResult:
    """Fit ``operator`` on a window of deployments (the v3 un-blinding).

    Rows from all deployments are pooled; each deployment's rows carry its own
    policy summary, so the summary *varies* across the training set and the
    operator can learn the distribution response ``dD/dphi``.  With a single
    deployment this reduces exactly to :func:`fit_operator`.
    """
    feats: List[Tensor] = []
    targs: List[Tensor] = []
    for dep in deployments:
        X_i, Y_i = build_dataset(dep.transitions, dep.policy)
        if X_i.shape[0]:
            feats.append(X_i)
            targs.append(Y_i)
    if not feats:
        return FitResult(n_rows=0)
    X = torch.cat(feats, dim=0)
    Y = torch.cat(targs, dim=0)
    return _train_operator(operator, X, Y, cfg, generator=generator)


def fit_operator(
    operator: MarketResponseOperator,
    transitions: Sequence[Transition],
    policy,
    cfg: OperatorConfig,
    generator: torch.Generator | None = None,
) -> FitResult:
    """Fit ``operator`` to ``transitions`` by MLE with early stopping.

    Parameters
    ----------
    operator:
        The operator to train (modified in place; left holding the best-val weights).
    transitions:
        Transitions from the current deployment.
    policy:
        The policy under which the data was collected (needed for the summary).
    cfg:
        Operator configuration (lr, epochs, batch size, patience, ...).
    generator:
        Optional RNG controlling the train/val shuffle and minibatching.

    Returns
    -------
    FitResult
        Loss curves and the baseline/best held-out NLL.
    """
    X, Y = build_dataset(transitions, policy)
    return _train_operator(operator, X, Y, cfg, generator=generator)


def _train_operator(
    operator: MarketResponseOperator,
    X: Tensor,
    Y: Tensor,
    cfg: OperatorConfig,
    generator: torch.Generator | None = None,
) -> FitResult:
    """Shared MLE training loop over pre-built rows (early stopping on val NLL)."""
    n = X.shape[0]
    result = FitResult(n_rows=n)
    if n == 0:
        return result

    # Standardize targets using this deployment's statistics.
    mean = Y.mean(dim=0)
    std = Y.std(dim=0).clamp_min(1e-6)
    operator.set_normalizer(mean, std)

    # Train/val split.
    perm = torch.randperm(n, generator=generator)
    n_val = max(1, int(cfg.val_fraction * n))
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    if train_idx.numel() == 0:  # degenerate tiny dataset
        train_idx = perm
    Xtr, Ytr = X[train_idx], Y[train_idx]
    Xval, Yval = X[val_idx], Y[val_idx]

    # Baseline held-out NLL before any optimization (for the improvement test).
    operator.eval()
    with torch.no_grad():
        result.baseline_val_nll = float(_nll(operator, Xval, Yval))

    opt = torch.optim.Adam(
        operator.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    best_val = float("inf")
    best_state = copy.deepcopy(operator.state_dict())
    patience_left = cfg.patience
    batch = int(cfg.batch_size)

    operator.train()
    for epoch in range(int(cfg.epochs)):
        # Shuffle training rows each epoch.
        ep_perm = torch.randperm(Xtr.shape[0], generator=generator)
        epoch_losses: List[float] = []
        for start in range(0, Xtr.shape[0], batch):
            sel = ep_perm[start : start + batch]
            opt.zero_grad()
            loss = _nll(operator, Xtr[sel], Ytr[sel])
            loss.backward()
            opt.step()
            epoch_losses.append(loss.item())

        operator.eval()
        with torch.no_grad():
            val = float(_nll(operator, Xval, Yval))
        operator.train()

        result.train_nll.append(sum(epoch_losses) / max(len(epoch_losses), 1))
        result.val_nll.append(val)
        result.epochs_run = epoch + 1

        if val < best_val - 1e-5:
            best_val = val
            best_state = copy.deepcopy(operator.state_dict())
            patience_left = cfg.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    operator.load_state_dict(best_state)
    operator.eval()
    result.best_val_nll = best_val
    return result
