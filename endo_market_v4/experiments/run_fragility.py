"""The REFLEX market-fragility index: the analytic boundary on 36 years of data.

Computes, for every trading day 1990-2026 in the shipped master panel, the
closed-form stability headroom ``eps*(t) = gamma(t)/beta`` and the fragility
index (median headroom / headroom) for IG and HY, saves the daily panels and
regime summaries, and renders the headline figure (fragility + VIX, coloured by
regime; GFC and COVID annotated).

Usage::

    python -m experiments.run_fragility --outdir outputs
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from reflex.analysis.fragility import compute_fragility, save_fragility
from reflex.calibration.regimes import REGIME_COLORS


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Daily market-fragility index from real data.")
    ap.add_argument("--outdir", default="outputs")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args(argv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    results = {}
    for rating in ("IG", "HY"):
        res = compute_fragility(rating=rating)
        daily_path, regime_path = save_fragility(res, outdir)
        results[rating] = res
        print(f"=== fragility index ({rating}) ===")
        print(res.by_regime.round(3).to_string())
        print(f"saved -> {daily_path}\nsaved -> {regime_path}\n")

    ig = results["IG"].daily
    peak = ig.loc[ig["fragility"].idxmax()]
    print(
        f"IG fragility peak: {peak['fragility']:.2f} on {peak['date'].date()} "
        f"(regime={peak['regime']}, VIX={peak['vix_close']:.1f})"
    )

    if not args.no_plot:
        fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
        for rating, style in (("IG", "-"), ("HY", "--")):
            df = results[rating].daily
            axes[0].plot(
                df["date"], df["fragility"], style, lw=0.8,
                label=f"{rating} fragility", color="#333333", alpha=0.85,
            )
        # regime shading from the IG panel
        colors = ig["regime"].map(REGIME_COLORS)
        axes[1].scatter(ig["date"], ig["vix_close"], c=colors, s=1.5, rasterized=True)
        for ax, ylab in ((axes[0], "fragility (median eps* / eps*)"), (axes[1], "VIX")):
            ax.set_ylabel(ylab)
            ax.grid(alpha=0.25)
        axes[0].axhline(1.0, color="gray", lw=0.8, ls=":")
        for date, label in (("2008-10-06", "Lehman"), ("2020-03-16", "COVID freeze")):
            axes[0].axvline(
                __import__("pandas").Timestamp(date), color="#e41a1c", lw=0.8, ls="--"
            )
            axes[0].annotate(
                label, xy=(__import__("pandas").Timestamp(date), axes[0].get_ylim()[1]),
                fontsize=8, color="#e41a1c", rotation=90, va="top",
            )
        axes[0].legend(loc="upper left", fontsize=8)
        axes[0].set_title(
            "REFLEX market-fragility index (closed-form stability headroom on real data, 1990-2026)"
        )
        fig.tight_layout()
        path = outdir / "fragility_index.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"saved figure -> {path}")


if __name__ == "__main__":
    main()
