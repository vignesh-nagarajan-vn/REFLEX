"""Run the stability phase-diagram sweep with the analytic overlay + robust bands.

The predict-then-verify protocol (theory 1.1 §8 + 1.4):

1. **Predict**: the closed-form modulus curve ``m_pred(f)`` and its boundary
   crossing ``f*`` are computed *before any simulation* from the same base
   config the sweep uses.
2. **Measure**: the CRN BR-slope modulus across seeds per grid point (median +
   IQR), as in the v2 headline experiment.
3. **Certify**: per grid point, the cross-seed ambiguity radius and the robust
   stability certificate (stable / unstable / undecided) of theory 1.4.

Usage::

    python -m experiments.run_sweep --config configs/sweep_feedback.yaml
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from reflex.analysis import load_sweep_spec, run_sweep
from reflex.analysis.phase import predicted_crossing, predicted_epsilon_sweep
from reflex.analysis.sweep import _config_from_base
from reflex.theory.robust import empirical_radius, robust_certificate


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Stability phase-diagram sweep (predict-then-verify).")
    ap.add_argument("--config", default="configs/sweep_feedback.yaml")
    ap.add_argument("--outdir", default="outputs")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args(argv)

    spec = load_sweep_spec(args.config)
    variable = spec["sweep"]["variable"]
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # 1) Predict (closed form, before any simulation).
    predicted = None
    f_star_pred = None
    if variable == "toxicity_feedback":
        base_cfg = _config_from_base(spec)
        grid = [float(x) for x in spec["sweep"]["grid"]]
        predicted = predicted_epsilon_sweep(base_cfg, grid)
        f_star_pred = predicted_crossing(predicted)
        print("=== analytic prediction (theory 1.1, evaluated before the sweep) ===")
        for p in predicted:
            print(f"  f={p.value:6.2f}: m_pred={p.m_pred:.3f} (h*={p.h_star:.3f})")
        if f_star_pred is not None:
            print(f"predicted boundary crossing f* ~ {f_star_pred:.3f}")

    # 2) Measure.
    print(f"\n=== measured sweep over '{variable}' ===")
    result = run_sweep(spec, verbose=True)
    if result.critical_value is not None:
        print(f"\nmeasured critical {variable}* ~ {result.critical_value:.3f}")
        if f_star_pred is not None:
            print(f"predicted vs measured crossing: {f_star_pred:.3f} vs {result.critical_value:.3f}")

    # 3) Certify (robust bands per grid point, theory 1.4).
    certs = []
    for p in result.points:
        mean, radius, _sigma = empirical_radius(p.moduli)
        certs.append(robust_certificate(mean, radius))

    # CSV (measured + predicted + certificate columns).
    rows = result.to_rows()
    for i, row in enumerate(rows):
        row["robust_verdict"] = certs[i].verdict
        row["robust_upper"] = certs[i].upper
        row["robust_lower"] = certs[i].lower
        if predicted is not None:
            row["m_pred"] = predicted[i].m_pred
            row["h_star_pred"] = predicted[i].h_star
    csv_path = outdir / f"sweep_{variable}_results.csv"
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved results -> {csv_path}")

    if not args.no_plot:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.fill_between(
            result.values,
            [p.q25 for p in result.points],
            [p.q75 for p in result.points],
            alpha=0.25, label="measured IQR (seeds)",
        )
        ax.plot(result.values, result.medians, "o-", label="measured median m")
        ax.fill_between(
            result.values,
            [c.lower for c in certs],
            [c.upper for c in certs],
            alpha=0.15, color="purple", label="robust band (1.4)",
        )
        if predicted is not None:
            ax.plot(result.values, [p.m_pred for p in predicted], "k--",
                    label="analytic m_pred (1.1)")
        ax.axhline(1.0, color="gray", ls=":", lw=1.0)
        if result.critical_value is not None:
            ax.axvline(result.critical_value, color="C0", ls="--", lw=0.9,
                       label=f"measured f* ~ {result.critical_value:.2f}")
        if f_star_pred is not None:
            ax.axvline(f_star_pred, color="black", ls=":", lw=0.9,
                       label=f"predicted f* ~ {f_star_pred:.2f}")
        ax.set_xlabel(variable)
        ax.set_ylabel("best-response contraction modulus m")
        ax.set_title("Stability phase diagram: predict (closed form) then verify (measured)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig_path = outdir / f"phase_diagram_{variable}.png"
        fig.savefig(fig_path, dpi=150)
        plt.close(fig)
        print(f"saved phase diagram -> {fig_path}")


if __name__ == "__main__":
    main()
