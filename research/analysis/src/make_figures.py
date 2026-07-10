"""Derived comparison figures from the curated full-profile results.

Reads the raw CSVs in research/results/<run>/ and renders the analysis
figures into research/analysis/figures/.  Raw artifacts are never modified.

Run from anywhere:

    python -u research/analysis/src/make_figures.py \
        --results research/results/07-10-2026 \
        --outdir research/analysis/figures
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def fig_dealer_amplification(results: Path, outdir: Path) -> None:
    df = pd.read_csv(results / "dealers" / "dealer_joint_moduli.csv")
    x = np.arange(len(df))
    w = 0.38
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.bar(x - w / 2, df["modulus_common_measured"], w, label="measured (genuine market)",
           color="#3b6fb6", edgecolor="black", linewidth=0.5)
    ax.bar(x + w / 2, df["modulus_common_predicted"], w, label="predicted N_eff * m_1 (1.3)",
           color="#c8c8c8", edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"N={int(n)}" for n in df["n_dealers"]])
    ax.set_ylabel("common-mode joint modulus")
    f_probe = df["probe_feedback"].iloc[0] if "probe_feedback" in df.columns else float("nan")
    ax.set_title(f"Competition amplifies the performative modulus "
                 f"(interior probe, f={f_probe})", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(outdir / "dealer_amplification.png", dpi=150)
    plt.close(fig)


def fig_triangulation(results: Path, outdir: Path) -> None:
    df = pd.read_csv(results / "triangulation" / "epsilon_triangulation.csv")
    row = df.iloc[0]
    names = ["analytic\n(a-priori A2)", "analytic\n(realized state)",
             "BR-slope", "Sinkhorn/W1", "CKS flow-curve"]
    vals = [row["epsilon_analytic"], row["epsilon_analytic_realized"],
            row["epsilon_br"], row["epsilon_sinkhorn"], row["epsilon_cks"]]
    colors = ["#c8c8c8", "#8a8a8a", "#3b6fb6", "#3b6fb6", "#3b6fb6"]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(names, vals, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_ylabel("epsilon at the operating spread")
    ax.set_title(f"Three-way epsilon triangulation at h_ref={row['h_ref']:.2f} "
                 f"(realized rho={row.get('realized_rho', float('nan')):.2f})",
                 fontsize=10)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(outdir / "triangulation_bars.png", dpi=150)
    plt.close(fig)


def fig_calibrated_measured(results: Path, outdir: Path) -> None:
    df = pd.read_csv(results / "calibrated" / "calibrated_boundaries.csv")
    m = df.dropna(subset=["m_measured_median"]) if "m_measured_median" in df.columns else df.iloc[0:0]
    if m.empty:
        return
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.scatter(m["m_pred_at_hobs"], m["m_measured_median"], s=45, zorder=3,
               color="#3b6fb6", edgecolor="black", linewidth=0.5)
    for _, r in m.iterrows():
        ax.annotate(f"{r['rating']}/{r['regime']}",
                    (r["m_pred_at_hobs"], r["m_measured_median"]),
                    textcoords="offset points", xytext=(6, 4), fontsize=7)
    lim = max(float(m["m_pred_at_hobs"].max()), float(m["m_measured_median"].max())) * 1.15
    ax.plot([0, lim], [0, lim], "k--", lw=0.8, label="y = x")
    ax.set_xlabel("closed-form m at observed spread (a-priori)")
    ax.set_ylabel("measured BR-slope modulus (median across seeds)")
    ax.set_title("Calibrated regimes: predicted vs measured local modulus", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(outdir / "calibrated_pred_vs_measured.png", dpi=150)
    plt.close(fig)


def fig_fragility_headroom(results: Path, outdir: Path) -> None:
    frames = {}
    for rating in ("IG", "HY"):
        p = results / "fragility" / f"fragility_index_{rating}_by_regime.csv"
        if p.exists():
            frames[rating] = pd.read_csv(p)
    if not frames:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    regimes = list(frames["IG"]["regime"]) if "IG" in frames else list(frames["HY"]["regime"])
    x = np.arange(len(regimes))
    w = 0.38
    for i, (rating, df) in enumerate(frames.items()):
        ax.bar(x + (i - 0.5) * w, df["eps_star"], w, label=rating,
               edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(regimes)
    ax.set_yscale("log")
    ax.set_ylabel("stability headroom eps* = gamma/beta (regime median, log)")
    ax.set_title("Stability headroom collapses calm -> crisis (real data, 1990-2026)",
                 fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(outdir / "fragility_headroom_by_regime.png", dpi=150)
    plt.close(fig)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Derived analysis figures.")
    ap.add_argument("--results", required=True)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args(argv)
    results = Path(args.results)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    for fn in (fig_dealer_amplification, fig_triangulation,
               fig_calibrated_measured, fig_fragility_headroom):
        try:
            fn(results, outdir)
            print(f"{fn.__name__}: ok")
        except Exception as exc:  # keep going; report at the end
            print(f"{fn.__name__}: SKIPPED ({exc})")


if __name__ == "__main__":
    main()
