"""Plotting: the stability phase diagram and RRM trajectory figures.

Uses a non-interactive matplotlib backend so figures can be written to disk on a
headless machine.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .sweep import SweepResult


def plot_phase_diagram(
    result: SweepResult,
    path: str | Path,
    title: Optional[str] = None,
) -> str:
    """Plot median best-response modulus vs the swept variable with an IQR band.

    The stability boundary ``m = 1`` and the located critical value are marked.
    """
    values = result.values
    medians = result.medians
    q25 = np.array([p.q25 for p in result.points])
    q75 = np.array([p.q75 for p in result.points])

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.fill_between(values, q25, q75, alpha=0.2, color="C0", label="IQR across seeds")
    ax.plot(values, medians, "o-", color="C0", label="median modulus  m̂")
    ax.axhline(1.0, color="crimson", ls="--", lw=1.2, label="stability boundary  m = 1")

    if result.critical_value is not None:
        ax.axvline(result.critical_value, color="k", ls=":", lw=1.0)
        ax.annotate(
            f"critical {result.variable} ≈ {result.critical_value:.2f}",
            xy=(result.critical_value, 1.0),
            xytext=(result.critical_value, max(medians) * 0.6 + 0.2),
            fontsize=9,
            ha="center",
        )

    # Shade stable / unstable regions.
    ax.axhspan(0, 1, color="green", alpha=0.04)
    ax.axhspan(1, max(2.0, float(np.nanmax(q75)) * 1.05), color="red", alpha=0.04)

    xlabel = {
        "toxicity_feedback": "performative-feedback gain  ε  (toxicity_feedback)",
        "alpha": "adversariality  α",
    }.get(result.variable, result.variable)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("best-response contraction modulus  m̂")
    ax.set_title(title or "Stability phase diagram")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    out = str(path)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def plot_rrm_trajectory(
    trajectory,
    path: str | Path,
    title: Optional[str] = None,
) -> str:
    """Plot the central-spread iterate path and successive step sizes of an RRM run."""
    h = trajectory.central_spreads
    steps = trajectory.step_sizes
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.plot(range(len(h)), h, "o-", color="C0")
    ax1.set_xlabel("RRM iteration  k")
    ax1.set_ylabel("central half-spread  hₖ")
    ax1.set_title("Policy iterate path")
    ax1.grid(alpha=0.25)

    if len(steps) > 0:
        ax2.semilogy(range(1, len(steps) + 1), np.maximum(steps, 1e-6), "s-", color="C3")
    ax2.set_xlabel("RRM iteration  k")
    ax2.set_ylabel("step size  ‖φₖ − φₖ₋₁‖  (log)")
    ax2.set_title("Successive iterate distances")
    ax2.grid(alpha=0.25)

    fig.suptitle(title or f"RRM run (α={trajectory.alpha}, seed={trajectory.seed})")
    fig.tight_layout()
    out = str(path)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out
