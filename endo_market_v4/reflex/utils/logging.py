"""Lightweight logging utilities with an optional, config-gated wandb backend."""

from __future__ import annotations

from typing import Any, Dict, Optional


class RunLogger:
    """Minimal metric logger.

    Prints to stdout when ``verbose`` and, only if explicitly enabled in config,
    mirrors scalar metrics to Weights & Biases.  wandb is *off by default* and
    imported lazily so it is never a hard dependency.
    """

    def __init__(
        self,
        verbose: bool = True,
        use_wandb: bool = False,
        wandb_project: str = "reflex",
        run_name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.verbose = verbose
        self.use_wandb = use_wandb
        self._wandb = None
        if use_wandb:
            try:
                import wandb  # type: ignore

                self._wandb = wandb
                wandb.init(project=wandb_project, name=run_name, config=config or {})
            except Exception as exc:  # pragma: no cover - optional path
                print(f"[logging] wandb requested but unavailable ({exc}); disabling.")
                self.use_wandb = False

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """Log a dict of scalar metrics."""
        if self.verbose:
            prefix = f"[step {step}] " if step is not None else ""
            body = "  ".join(
                f"{k}={v:.4g}" if isinstance(v, (int, float)) else f"{k}={v}"
                for k, v in metrics.items()
            )
            print(prefix + body)
        if self.use_wandb and self._wandb is not None:
            self._wandb.log(metrics, step=step)

    def info(self, message: str) -> None:
        if self.verbose:
            print(message)

    def finish(self) -> None:
        if self.use_wandb and self._wandb is not None:  # pragma: no cover
            self._wandb.finish()
