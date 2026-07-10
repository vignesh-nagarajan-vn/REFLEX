"""Configuration schema and loading for the reflex (endo_market_v3) package.

The whole project is driven by a single nested :class:`Config` dataclass that is
populated from a YAML file (plus optional overrides).  Keeping configuration in
typed dataclasses means every experiment is fully described by ``(config, seed)``
and is therefore reproducible.

Design choices worth knowing:

* The :class:`ClientsConfig` carries the *adversariality* knob ``alpha`` and the
  parameters that govern how toxic (informed) flow reacts to the dealer's
  quotes.  ``toxicity_feedback`` is the explicit lever that scales the
  sensitivity ``epsilon`` of the distribution map ``D(phi)`` with ``alpha`` --
  it is the central object of the convergence study.
* Unknown keys in the YAML are ignored (with a warning) so configs can carry
  documentation/comment-like keys without breaking the loader.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, Type, TypeVar, get_type_hints

import yaml

T = TypeVar("T")


# --------------------------------------------------------------------------- #
# Section dataclasses                                                         #
# --------------------------------------------------------------------------- #
@dataclass
class BondsConfig:
    """Cross-sectional structure of the tradable universe."""

    n_bonds: int = 8
    n_sectors: int = 2
    global_factor: float = 0.25  # weight of a single market-wide factor in corr
    within_sector_corr: float = 0.45  # extra correlation inside a sector block


@dataclass
class ClientsConfig:
    """Client order-flow model (the policy-dependence channel).

    ``alpha`` controls the *fraction and signal strength* of informed (toxic)
    flow.  Tighter quotes admit proportionally more informed flow, and the
    sensitivity of toxicity to the dealer's spread scales with ``alpha`` -- this
    is what makes the distribution map ``D(phi)`` more sensitive (larger
    ``epsilon``) as the market becomes more adversarial.
    """

    alpha: float = 0.15
    base_arrival_rate: float = 1.0  # uninformed notional arrival per step per bond
    demand_elasticity: float = 1.5  # how strongly volume decays with half-spread
    info_signal_noise: float = 0.6  # std of informed traders' signal noise
    info_intensity: float = 1.4  # scale of *spread-responsive* informed notional (slope term)
    info_base_intensity: float = 0.60  # alpha-independent baseline toxic level (sets response regime)
    info_spread_decay: float = 1.5  # decay of toxic spread-responsiveness in h
    info_cap: float = 8.0  # saturation cap on informed notional (bounds the toxicity)
    toxicity_feedback: float = 0.22  # explicit gain of toxicity-vs-spread feedback (~epsilon scaler)
    spread_ref: float = 0.5  # reference half-spread around which feedback is centred
    # Multi-dealer competition (math-theory 1.3).  ``n_dealers = 1`` and
    # ``toxic_spillover = 0`` reproduce the single-dealer market bit-for-bit; the
    # analytic N-dealer boundary (analysis/multi_dealer_modulus.py) reads these.
    n_dealers: int = 1  # N: number of dealers sharing one informed-flow pool
    toxic_spillover: float = 0.0  # kappa in [0, 1]: cross-dealer toxic spillover


@dataclass
class SimulatorConfig:
    """Structural ground-truth dynamics ``T_true``."""

    horizon: int = 64
    fundamental_vol: float = 0.25  # per-step random-walk vol of latent fundamental
    impact: float = 0.30  # price impact of net client flow on the mid
    mid_noise: float = 0.02  # idiosyncratic mid noise
    mid_reversion: float = 0.10  # baseline pull of mid toward fundamental (price discovery)
    mid_move_cap: float = 4.0  # cap on |mid change| per step (bounds runaway feedback)
    inventory_decay: float = 0.0  # optional exogenous inventory run-off
    init_mispricing_vol: float = 0.5  # std of initial mid-vs-fundamental gap
    # latent liquidity field
    liq_mean: float = 1.0
    liq_reversion: float = 0.15
    liq_vol: float = 0.05
    liq_flow_boost: float = 0.10  # transient liquidity boost from attracting flow
    liq_overtighten_decay: float = 0.20  # persistent degradation from over-tight quotes
    liq_coupling: float = 0.30  # cross-bond coupling strength via corr matrix


@dataclass
class OperatorConfig:
    """Learned, differentiable Market Response Operator ``T_theta``."""

    hidden: int = 64
    head_type: str = "gaussian"  # {"gaussian", "gaussian_diag", "mixture"}
    n_mixture: int = 3  # components if head_type == "mixture"
    lr: float = 1e-3
    epochs: int = 60
    batch_size: int = 256
    weight_decay: float = 1e-5
    val_fraction: float = 0.2
    patience: int = 8  # early-stopping patience on held-out NLL
    min_logstd: float = -4.0
    max_logstd: float = 2.0


@dataclass
class PolicyConfig:
    """Dealer policy ``pi_phi``."""

    type: str = "linear"  # {"linear", "mlp"}
    hidden: int = 32  # MLP hidden width (ignored for linear)
    lr: float = 3e-2
    inner_steps: int = 60  # gradient steps per RRM policy re-optimization
    rollout_horizon: int = 16  # horizon used when optimizing against T_theta
    n_rollouts: int = 16  # parallel reparameterized rollouts (Monte Carlo)
    init_half_spread: float = 0.8  # initial central half-spread (via bias)
    max_half_spread: float = 6.0  # clamp for numerical stability
    reinforce: bool = False  # force REINFORCE path instead of pathwise gradients


@dataclass
class RewardConfig:
    """Dealer objective weights."""

    inv_risk_weight: float = 0.05  # quadratic inventory-risk penalty
    pnl_scale: float = 1.0  # global scale on the economic P&L
    quote_anchor_weight: float = 0.25  # quadratic quoting-cost convexity on the half-spread
    quote_anchor_ref: float = 1.0  # reference half-spread the quoting cost pulls toward


@dataclass
class StabilityConfig:
    """Config-weighted stability penalties / diagnostics."""

    entropy_w: float = 0.0  # anti-collapse: floor on induced flow entropy
    hhi_w: float = 0.0  # penalize pathological concentration across bonds
    toxicity_w: float = 0.0  # penalize P&L lost to informed flow
    lipschitz_w: float = 0.0  # penalize policy->distribution sensitivity (the key lever)
    lipschitz_eps: float = 0.05  # perturbation size for the Lipschitz estimate


@dataclass
class RRMConfig:
    """Repeated-Risk-Minimization outer loop."""

    max_iters: int = 15
    tol: float = 1e-3  # stop when ||phi_{k+1}-phi_k|| < tol
    n_episodes: int = 24  # episodes of T_true data collected per iteration
    collection_jitter: float = 0.20  # exploration noise on quotes during collection
    eval_episodes: int = 16  # fresh T_true rollouts to estimate performative risk
    warm_start_policy: bool = True  # re-optimize from previous iterate (vs. fresh init)
    refit_window: int = 1  # number of recent iterations' data used to fit T_theta
    # Policy-update rule for the outer loop:
    #   "rgd" -- repeated gradient descent: a few controlled gradient steps on the
    #            risk under the freshly-refit operator (the standard, well-behaved
    #            repeated-retraining scheme whose modulus is ~ 1 - eta*gamma +
    #            eta*c*alpha, linear in alpha with a tunable threshold); or
    #   "rrm" -- full best-response re-optimization (uses optimize_policy).
    update_rule: str = "rgd"
    rgd_steps: int = 5  # number of gradient steps per deployment when update_rule="rgd"
    rgd_lr: float = 0.08  # step size for the repeated-gradient-descent update


@dataclass
class LoggingConfig:
    """Output / logging behaviour."""

    outdir: str = "outputs"
    use_wandb: bool = False
    wandb_project: str = "reflex"
    verbose: bool = True


@dataclass
class Config:
    """Top-level configuration for a single RRM run."""

    seed: int = 0
    device: str = "cpu"
    dtype: str = "float32"
    bonds: BondsConfig = field(default_factory=BondsConfig)
    clients: ClientsConfig = field(default_factory=ClientsConfig)
    simulator: SimulatorConfig = field(default_factory=SimulatorConfig)
    operator: OperatorConfig = field(default_factory=OperatorConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    stability: StabilityConfig = field(default_factory=StabilityConfig)
    rrm: RRMConfig = field(default_factory=RRMConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


# --------------------------------------------------------------------------- #
# Loading helpers                                                             #
# --------------------------------------------------------------------------- #
def _from_dict(cls: Type[T], data: Dict[str, Any]) -> T:
    """Recursively build a (possibly nested) dataclass from a dict.

    Unknown keys are ignored with a printed warning so that configs may contain
    extra documentation keys without breaking loading.  ``from __future__ import
    annotations`` turns field annotations into strings, so we resolve them to
    real types with :func:`typing.get_type_hints` before checking for nested
    dataclasses.
    """
    if not is_dataclass(cls):
        return data  # type: ignore[return-value]
    resolved = get_type_hints(cls)
    field_names = {f.name for f in fields(cls)}
    kwargs: Dict[str, Any] = {}
    for key, value in (data or {}).items():
        if key not in field_names:
            print(f"[config] warning: ignoring unknown key '{key}' for {cls.__name__}")
            continue
        ftype = resolved.get(key)
        if is_dataclass(ftype) and isinstance(value, dict):
            kwargs[key] = _from_dict(ftype, value)
        else:
            kwargs[key] = value
    return cls(**kwargs)  # type: ignore[arg-type]


def load_config(path: str | Path, overrides: Dict[str, Any] | None = None) -> Config:
    """Load a :class:`Config` from a YAML file, applying optional overrides.

    Parameters
    ----------
    path:
        Path to a YAML config file.
    overrides:
        Optional nested dict merged on top of the file contents (deep merge).
    """
    with open(path, "r") as fh:
        raw = yaml.safe_load(fh) or {}
    if overrides:
        raw = deep_merge(raw, overrides)
    return _from_dict(Config, raw)


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Return a deep-merged copy of ``base`` updated by ``override``."""
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def config_to_dict(cfg: Config) -> Dict[str, Any]:
    """Convert a :class:`Config` (nested dataclasses) into a plain dict."""
    return dataclasses.asdict(cfg)


def torch_dtype(cfg: Config):
    """Resolve the configured dtype string to a torch dtype object."""
    import torch

    return {"float32": torch.float32, "float64": torch.float64}[cfg.dtype]
