"""Calibrated stability boundaries per market regime (real-data microstructure).

For each ``(rating, regime)`` cell of the fitted calibration table, build the
calibrated config (real units), evaluate the closed-form boundary a-priori
(theory 1.1 on data-identified ``A, k, sigma, h``), and -- optionally -- verify
one cell with the measured BR-slope modulus.  The headline table: the stability
headroom ``eps*`` shrinks monotonically calm -> crisis.

Usage::

    python -m experiments.run_calibrated --outdir outputs [--measure] [--seeds 2]
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from reflex.calibration import REGIMES, calibrated_config
from reflex.calibration.regimes import REGIME_COLORS
from reflex.config import load_config
from reflex.theory.analytic_boundary import analytic_boundary


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Calibrated per-regime stability boundaries.")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--outdir", default="outputs")
    ap.add_argument("--measure", action="store_true",
                    help="also measure the BR-slope modulus for IG cells (slow)")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args(argv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    base = load_config(args.config)
    rows = []
    print(f"{'rating':>6} {'regime':>9} {'h_obs':>8} {'h*':>8} {'gamma':>9} "
          f"{'eps*':>9} {'m_pred':>7}")
    for rating in ("IG", "HY"):
        for regime in REGIMES:
            cfg, info = calibrated_config(base, rating=rating, regime=regime)
            ab = analytic_boundary(cfg)
            row = {
                "rating": rating,
                "regime": regime,
                "h_observed": info.h_100,
                "h_star": ab.h_star,
                "gamma": ab.gamma,
                "beta": ab.beta,
                "eps_star": ab.boundary_epsilon,
                "m_pred": ab.modulus,
                "degenerate_fit": info.degenerate,
                "A": info.A_100,
                "k": info.k_100,
                "vol_step": info.vol_100,
            }
            if args.measure and rating == "IG" and not info.degenerate:
                import numpy as np

                from reflex.estimators import measure_response_modulus

                moduli = [
                    measure_response_modulus(
                        cfg, seed=s, h_ref=ab.h_star, delta=0.25 * ab.h_star
                    ).modulus
                    for s in range(args.seeds)
                ]
                row["m_measured_median"] = float(np.median(moduli))
            rows.append(row)
            print(
                f"{rating:>6} {regime:>9} {info.h_100:8.3f} {ab.h_star:8.3f} "
                f"{ab.gamma:9.2f} {ab.boundary_epsilon:9.2f} {ab.modulus:7.3f}"
                + ("  [degenerate k fit]" if info.degenerate else "")
                + (f"  m_meas={row.get('m_measured_median', float('nan')):.3f}"
                   if "m_measured_median" in row else "")
            )

    csv_path = outdir / "calibrated_boundaries.csv"
    fieldnames = sorted({key for row in rows for key in row})
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved table -> {csv_path}")

    if not args.no_plot:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for rating, hatch in (("IG", None), ("HY", "//")):
            vals = [r["eps_star"] for r in rows if r["rating"] == rating]
            xs = [i + (0.0 if rating == "IG" else 0.38) for i in range(len(REGIMES))]
            ax.bar(
                xs, vals, width=0.36, hatch=hatch,
                color=[REGIME_COLORS[r] for r in REGIMES],
                edgecolor="black", linewidth=0.6,
                label=rating, alpha=0.9 if rating == "IG" else 0.6,
            )
        ax.set_xticks([i + 0.19 for i in range(len(REGIMES))])
        ax.set_xticklabels(REGIMES)
        ax.set_ylabel("stability headroom  eps* = gamma / beta")
        ax.set_title("Calibrated a-priori stability boundary per market regime")
        ax.legend()
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        path = outdir / "calibrated_boundaries.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"saved figure -> {path}")


if __name__ == "__main__":
    main()
