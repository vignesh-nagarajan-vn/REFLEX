"""Run a stability phase-diagram sweep and save the figure + results.csv.

Usage::

    python -m experiments.run_sweep --config configs/sweep_feedback.yaml
    endo-run-sweep --config configs/sweep_feedback.yaml --outdir outputs

The primary sweep (``sweep_feedback.yaml``) varies the performative-feedback gain
ε (``toxicity_feedback``) and locates ε* where the median best-response modulus
crosses 1.  The secondary sweep (``sweep_adversariality.yaml``) varies α.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from reflex.analysis import load_sweep_spec, run_sweep
from reflex.analysis.plots import plot_phase_diagram


def main() -> None:
    ap = argparse.ArgumentParser(description="Run a stability phase-diagram sweep.")
    ap.add_argument("--config", default="configs/sweep_feedback.yaml")
    ap.add_argument("--outdir", default="outputs")
    args = ap.parse_args()

    spec = load_sweep_spec(args.config)
    variable = spec["sweep"]["variable"]
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"=== sweep over '{variable}' ===")
    result = run_sweep(spec, verbose=True)

    if result.critical_value is not None:
        print(f"\ncritical {variable}* (median modulus crosses 1) ≈ {result.critical_value:.3f}")
    else:
        print(f"\nmedian modulus does not cross 1 within the swept range of {variable}")

    # CSV
    csv_path = outdir / f"sweep_{variable}_results.csv"
    rows = result.to_rows()
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved results -> {csv_path}")

    # Figure
    fig_path = outdir / f"phase_diagram_{variable}.png"
    plot_phase_diagram(result, fig_path)
    print(f"saved phase diagram -> {fig_path}")


if __name__ == "__main__":
    main()
