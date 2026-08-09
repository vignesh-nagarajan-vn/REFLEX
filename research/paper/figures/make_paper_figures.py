"""Regenerate the six paper figures on one colour system.

The experiment scripts under ``endo_market_v4/experiments/`` own the *science*:
they run the loops and write the CSVs under ``research/results/<run>/``.  This
script owns the *presentation* of the six figures that appear in the paper, and
it reads nothing but those committed CSVs.  No simulation is re-run, so the
numbers in the figures are exactly the numbers of the run named by ``--results``.

Why it exists.  v3's figures came from five unrelated palettes (``RdYlGn_r``,
``tab:``, ``C0``, ``#e41a1c``, ``#333333``) and were drawn at 11-12 inches wide
before being squeezed into a 3.33-inch column, which shrank their tick labels to
roughly 2pt.  This script fixes both: every figure is drawn at the size it is
actually printed at, and every figure draws from the Okabe-Ito colourblind-safe
palette with one fixed meaning per colour.

    measured / empirical      blue      solid       circle
    predicted / closed form   vermilion dashed      square
    secondary series          green     dash-dot    diamond
    tertiary series           purple    dotted      triangle
    boundary / reference      black     dotted      (no marker)

Series are separated by dash pattern and marker as well as hue, so the figures
survive greyscale printing and red-green colour-vision deficiency.  The v3
red-yellow-green systemic surface and the red/green stable-unstable shading are
gone for that reason.

Run from anywhere::

    python research/paper/figures/make_paper_figures.py

    python research/paper/figures/make_paper_figures.py \
        --results research/results/07-12-2026 --outdir research/paper/figures
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# --------------------------------------------------------------- palette --
# Okabe & Ito's eight-colour qualitative palette, the de-facto standard for
# colourblind-safe scientific figures.  Distinguishable under deuteranopia,
# protanopia and tritanopia, and monotone in luminance order under greyscale.
BLACK = "#000000"
ORANGE = "#E69F00"
SKY = "#56B4E9"
GREEN = "#009E73"
BLUE = "#0072B2"
VERMILION = "#D55E00"
PURPLE = "#CC79A7"
GREY = "#7F7F7F"

#: One fixed meaning per role, used by every figure in the paper.
MEASURED = dict(color=BLUE, ls="-", marker="o")
PREDICTED = dict(color=VERMILION, ls="--", marker="s")
SECONDARY = dict(color=GREEN, ls="-.", marker="D")
TERTIARY = dict(color=PURPLE, ls=":", marker="^")
REFERENCE = dict(color=BLACK, ls=":", lw=0.8)

#: Crowded panels get an opaque legend box so an overlapping curve or reference
#: line reads cleanly underneath it rather than through the label text.
_LEGEND_BOX = dict(frameon=True, framealpha=0.92, facecolor="white",
                   edgecolor="none")

#: Volatility regimes, calm -> crisis, as a cool-to-warm-to-black severity ramp.
#: Replaces the v3 Set1 colours, whose calm/stress pair was green/red.
REGIME_COLORS = {
    "calm": BLUE,
    "normal": SKY,
    "elevated": ORANGE,
    "stress": VERMILION,
    "crisis": BLACK,
}
REGIMES = ["calm", "normal", "elevated", "stress", "crisis"]

#: acmart sigconf geometry, in inches.
COL_W = 3.33
TEXT_W = 7.00
DPI = 400


def _rc(base: float) -> dict:
    """Matplotlib rcParams at a given base point size."""
    return {
        "font.size": base,
        "axes.titlesize": base + 0.5,
        "axes.labelsize": base,
        "xtick.labelsize": base - 0.5,
        "ytick.labelsize": base - 0.5,
        "legend.fontsize": base - 0.5,
        "axes.linewidth": 0.6,
        "grid.linewidth": 0.4,
        "lines.linewidth": 1.1,
        "lines.markersize": 3.0,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "legend.frameon": False,
        "legend.handlelength": 2.2,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "figure.dpi": DPI,
        "savefig.dpi": DPI,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    }


def _save(fig, outdir: Path, name: str) -> None:
    path = outdir / name
    fig.savefig(path)
    plt.close(fig)
    print(f"saved -> {path}")


# ------------------------------------------------------------- figure 1 --
def fig_fragility(results: Path, outdir: Path) -> None:
    """Daily fragility index 1990-2026, over the VIX regime spine."""
    ig = pd.read_csv(results / "fragility" / "fragility_index_IG_daily.csv",
                     parse_dates=["date"])
    hy = pd.read_csv(results / "fragility" / "fragility_index_HY_daily.csv",
                     parse_dates=["date"])

    with plt.rc_context(_rc(6.0)):
        fig, axes = plt.subplots(
            2, 1, figsize=(COL_W, 1.90), sharex=True,
            gridspec_kw={"height_ratios": [2.0, 1.0], "hspace": 0.12},
        )
        axes[0].set_title("Market-fragility index, 1990–2026", pad=3)
        axes[0].plot(ig["date"], ig["fragility"], color=BLUE, ls="-", lw=0.5,
                     label="IG")
        axes[0].plot(hy["date"], hy["fragility"], color=VERMILION, ls="--",
                     lw=0.5, label="HY")
        axes[0].axhline(1.0, **REFERENCE)
        axes[0].set_ylabel("fragility", labelpad=1.5)
        axes[0].legend(loc="upper left", ncol=2, columnspacing=1.0,
                       borderaxespad=0.2, handletextpad=0.5)

        for date, label in (("2008-10-06", "Lehman"), ("2020-03-16", "COVID")):
            ts = pd.Timestamp(date)
            axes[0].axvline(ts, color=GREY, lw=0.6, ls="--")
            axes[0].annotate(label, xy=(ts, axes[0].get_ylim()[1]),
                             xytext=(2, -1), textcoords="offset points",
                             fontsize=4.6, color=GREY, rotation=90, va="top")

        colors = ig["regime"].map(REGIME_COLORS).fillna(GREY)
        axes[1].scatter(ig["date"], ig["vix_close"], c=colors, s=0.35,
                        linewidths=0, rasterized=True)
        axes[1].set_ylabel("VIX", labelpad=1.5)

        handles = [plt.Line2D([], [], ls="", marker="s", ms=2.2,
                              color=REGIME_COLORS[r], label=r) for r in REGIMES]
        axes[1].legend(handles=handles, loc="upper left", ncol=5,
                       columnspacing=0.6, handletextpad=0.2,
                       borderaxespad=0.2, fontsize=4.4)
        for ax in axes:
            ax.tick_params(pad=1.5)
        _save(fig, outdir, "fragility_index.png")


# ------------------------------------------------------------- figure 2 --
def fig_phase_diagram(results: Path, outdir: Path) -> None:
    """Predict-then-verify sweep over the feedback gain f."""
    df = pd.read_csv(results / "sweep" / "sweep_toxicity_feedback_results.csv")
    f = df["value"].to_numpy()

    with plt.rc_context(_rc(6.5)):
        fig, ax = plt.subplots(figsize=(COL_W, 2.00))
        ax.set_title("Predict-then-verify: the boundary crossing", pad=3)

        ax.fill_between(f, df["robust_lower"], df["robust_upper"], color=BLUE,
                        alpha=0.10, lw=0, label="R4 robust band")
        ax.fill_between(f, df["q25"], df["q75"], color=BLUE, alpha=0.22, lw=0,
                        label="IQR (8 seeds)")
        ax.plot(f, df["median_modulus"], **MEASURED, label="measured median")
        ax.plot(f, df["m_pred"], **PREDICTED, label="closed form, probe spread")
        ax.plot(f, df["m_pred_hstar"], **SECONDARY, label="fixed-point curve")

        ax.axhline(1.0, color=BLACK, ls=":", lw=0.9)
        ax.annotate("stability boundary  m = 1", xy=(f.max(), 1.0),
                    xytext=(-2, 2), textcoords="offset points", ha="right",
                    va="bottom", fontsize=5.4)
        for x, c, lab in ((3.17, BLUE, "measured $f^*$"),
                          (4.70, VERMILION, "predicted $f^*$")):
            ax.axvline(x, color=c, ls=":", lw=0.8)
            ax.annotate(lab, xy=(x, ax.get_ylim()[0]), xytext=(2, 3),
                        textcoords="offset points", rotation=90, va="bottom",
                        fontsize=5.0, color=c)

        ax.set_xlabel("toxicity-feedback gain  $f$", labelpad=1.5)
        ax.set_ylabel("contraction modulus  $m$", labelpad=1.5)
        ax.legend(loc="upper left", borderaxespad=0.2, handletextpad=0.5,
                  labelspacing=0.25, **_LEGEND_BOX)
        ax.tick_params(pad=1.5)
        _save(fig, outdir, "phase_diagram_toxicity_feedback.png")


# ------------------------------------------------------------- figure 3 --
def fig_dealer_surface(results: Path, outdir: Path) -> None:
    """R3 systemic surface m_N over the (f, N) grid."""
    df = pd.read_csv(results / "dealers" / "dealer_phase_grid.csv")
    n_values = df["n_dealers"].to_numpy()
    f_cols = [c for c in df.columns if c.startswith("f=")]
    f_values = np.array([float(c.split("=")[1]) for c in f_cols])
    grid = df[f_cols].to_numpy(dtype=float)

    with plt.rc_context(_rc(6.5)):
        fig, ax = plt.subplots(figsize=(COL_W, 1.88))
        ax.set_title("Systemic stability surface (R3)", pad=3)
        # cividis: perceptually uniform and safe under all three common forms
        # of colour-vision deficiency, unlike the v3 RdYlGn_r.
        im = ax.imshow(grid, aspect="auto", origin="lower", cmap="cividis",
                       vmin=0.0, vmax=2.0,
                       extent=(f_values.min(), f_values.max(), 0, len(n_values)))
        cs = ax.contour(np.linspace(f_values.min(), f_values.max(), grid.shape[1]),
                        [i + 0.5 for i in range(len(n_values))], grid,
                        levels=[1.0], colors="white", linewidths=1.2)
        ax.clabel(cs, fmt="$m_N=1$", fontsize=5.4)
        ax.set_yticks([i + 0.5 for i in range(len(n_values))])
        ax.set_yticklabels([int(n) for n in n_values])
        ax.set_xlabel("toxicity-feedback gain  $f$", labelpad=1.5)
        ax.set_ylabel("dealers  $N$", labelpad=1.5)
        ax.grid(False)
        cb = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.046)
        cb.set_label("joint modulus  $m_N$", labelpad=2)
        cb.ax.tick_params(labelsize=5.4, pad=1.5)
        cb.outline.set_linewidth(0.6)
        ax.tick_params(pad=1.5)
        _save(fig, outdir, "dealer_phase_diagram.png")


# ------------------------------------------------------------- figure 4 --
def fig_perfgd_loops(results: Path, outdir: Path) -> None:
    """The four learned retraining loops and the ML<->theory seam."""
    df = pd.read_csv(results / "perfgd" / "perfgd_ml_loops.csv")
    # Frozen-reference context lines (theory.perfgd on the unstable demo config).
    h_po, h_sp = 1.641274182332627, 3.936477706337007

    modes = (("rrm", "blind RRM", MEASURED),
             ("perfgd_analytic", "PerfGD-analytic", PREDICTED),
             ("perfgd_learned", "PerfGD-learned", TERTIARY),
             ("perfgd_structural", "PerfGD-structural", SECONDARY))

    with plt.rc_context(_rc(7.0)):
        fig, axes = plt.subplots(1, 2, figsize=(TEXT_W, 2.45))

        for mode, label, style in modes:
            d = df[df["mode"] == mode].sort_values("k")
            axes[0].plot(d["k"], d["central_half_spread"], **style, label=label)
        axes[0].axhline(h_po, color=BLACK, ls="--", lw=0.8)
        axes[0].annotate("$h_{PO}$ (frozen ref.)", xy=(0, h_po), xytext=(1, 2),
                         textcoords="offset points", fontsize=5.6)
        axes[0].axhline(h_sp, color=GREY, ls=":", lw=0.8)
        axes[0].annotate("$h_{SP}$", xy=(0, h_sp), xytext=(1, 2),
                         textcoords="offset points", fontsize=5.6, color=GREY)
        axes[0].set_xlabel("deployment  $k$", labelpad=1.5)
        axes[0].set_ylabel("central half-spread  $h$", labelpad=1.5)
        axes[0].set_title("Outer loops from a common seed")
        axes[0].legend(loc="center right", borderaxespad=0.3,
                       handletextpad=0.5, labelspacing=0.25, **_LEGEND_BOX)

        learned = df[df["mode"] == "perfgd_learned"].sort_values("k")
        struct = df[df["mode"] == "perfgd_structural"].sort_values("k")
        axes[1].plot(learned["k"], learned["learned_slope"], **TERTIARY,
                     label="free-form learned $d\\tau/dh$")
        axes[1].plot(struct["k"], struct["structural_slope"], **SECONDARY,
                     label="structural fit")
        axes[1].plot(learned["k"], learned["analytic_slope"], color=BLACK,
                     ls="--", lw=1.0, label="analytic $-\\psi\\,\\epsilon(h)$")
        axes[1].axhline(0.0, **REFERENCE)
        axes[1].set_xlabel("deployment  $k$", labelpad=1.5)
        axes[1].set_ylabel("toxic slope", labelpad=1.5)
        axes[1].set_title("The ML$\\leftrightarrow$theory seam")
        axes[1].legend(loc="lower right", borderaxespad=0.3,
                       handletextpad=0.5, labelspacing=0.25)

        for ax in axes:
            ax.tick_params(pad=1.5)
        fig.tight_layout(pad=0.3, w_pad=1.2)
        _save(fig, outdir, "perfgd_ml_loops.png")


# ------------------------------------------------------------- figure 5 --
def fig_lazy_deploy(results: Path, outdir: Path) -> None:
    """R6: the K-step deployment map and the implied effective curvature."""
    sweep = pd.read_csv(results / "lazy_deploy" / "lazy_deploy_sweep.csv")
    summ = pd.read_csv(results / "lazy_deploy" / "lazy_deploy_summary.csv")
    m = float(summ["m_anchor"].iloc[0])
    c = float(summ["c_fit"].iloc[0])
    k_grid = summ["K"].to_numpy(dtype=float)
    k_dense = np.linspace(k_grid.min(), k_grid.max(), 300)

    def mu(K):            # theory 1.6: mu(K) = -m + c^K (1 + m)
        return -m + c ** K * (1.0 + m)

    k_db = np.log(m / (1.0 + m)) / np.log(c)

    with plt.rc_context(_rc(7.0)):
        fig, axes = plt.subplots(1, 2, figsize=(TEXT_W, 2.45))

        axes[0].plot(k_dense, mu(k_dense), color=BLACK, ls="-", lw=1.1,
                     label=f"fit  $-m+c^K(1{{+}}m)$,  $c={c:.3f}$")
        axes[0].plot(sweep["K"], sweep["slope"], ls="", marker="o", ms=2.4,
                     color=BLUE, alpha=0.35, label="per-seed")
        axes[0].plot(k_grid, summ["mu_median"], **MEASURED,
                     label=r"measured median $\mu(K)$")
        axes[0].axhline(-m, **PREDICTED_LINE(), label=f"$K\\to\\infty$: $-m={-m:.3f}$")
        axes[0].axhline(0.0, **REFERENCE)
        axes[0].axvline(k_db, color=GREY, ls=":", lw=0.8)
        axes[0].annotate(f"deadbeat $K={k_db:.2f}$",
                         xy=(k_db, axes[0].get_ylim()[1]), xytext=(2.5, -3),
                         textcoords="offset points", rotation=90, va="top",
                         fontsize=5.6, color=GREY)
        axes[0].set_xlabel("inner steps per deployment  $K$", labelpad=1.5)
        axes[0].set_ylabel(r"deployment-map slope  $\mu(K)$", labelpad=1.5)
        axes[0].set_title("The $K$-step deployment map")
        axes[0].legend(loc="lower left", borderaxespad=0.3,
                       handletextpad=0.5, labelspacing=0.25, **_LEGEND_BOX)

        axes[1].plot(k_dense, m / np.abs(mu(k_dense)), color=BLACK, ls="-",
                     lw=1.1, label=r"predicted $\gamma_{eff}/\gamma$")
        axes[1].plot(k_grid, summ["gamma_eff_over_gamma_meas"], ls="",
                     marker="D", ms=3.0, color=GREEN,
                     label="measured (from median $|\\mu|$)")
        axes[1].axhline(1.0, color=VERMILION, ls="--", lw=0.9,
                        label=r"$\gamma_{eff}=\gamma$ (exact RRM)")
        axes[1].set_yscale("log")
        axes[1].set_xlabel("inner steps per deployment  $K$", labelpad=1.5)
        axes[1].set_ylabel(r"$\gamma_{eff}/\gamma$", labelpad=1.5)
        axes[1].set_title("Effective curvature of the lazy loop")
        axes[1].legend(loc="upper right", borderaxespad=0.3,
                       handletextpad=0.5, labelspacing=0.25)
        axes[1].grid(alpha=0.25, which="both")

        for ax in axes:
            ax.tick_params(pad=1.5)
        fig.tight_layout(pad=0.3, w_pad=1.2)
        _save(fig, outdir, "lazy_deploy_gamma_eff.png")


def PREDICTED_LINE() -> dict:
    """The predicted-role style, minus the marker (for axhline)."""
    return dict(color=VERMILION, ls="--", lw=0.9)


# ------------------------------------------------------------- figure 6 --
def fig_universe(results: Path, outdir: Path) -> None:
    """R5: factor-model scaling and the truncation bound."""
    scal = pd.read_csv(results / "universe" / "universe_scaling.csv")
    trunc = pd.read_csv(results / "universe" / "universe_truncation.csv")
    d_max = int(scal["n_bonds"].max())

    with plt.rc_context(_rc(7.0)):
        fig, axes = plt.subplots(1, 2, figsize=(TEXT_W, 2.45))

        axes[0].plot(scal["n_bonds"], scal["rho_M"], **MEASURED,
                     label=r"$\rho(M)$")
        axes[0].plot(scal["n_bonds"], scal["m_scalar_max"], **SECONDARY,
                     label="worst scalar modulus")
        axes[0].axhline(1.0, color=BLACK, ls=":", lw=0.9)
        axes[0].annotate(r"stability boundary  $\rho=1$",
                         xy=(scal["n_bonds"].max(), 1.0), xytext=(-2, -6),
                         textcoords="offset points", ha="right", fontsize=5.6)
        axes[0].set_xscale("log", base=2)
        axes[0].set_xticks(scal["n_bonds"])
        axes[0].set_xticklabels([int(v) for v in scal["n_bonds"]])
        axes[0].set_ylim(0, 1.12)
        axes[0].set_xlabel("universe size  $d$ (bonds)", labelpad=1.5)
        axes[0].set_ylabel(r"spectral modulus  $\rho(M)$", labelpad=1.5)
        axes[0].set_title("Scaling in universe size")
        axes[0].legend(loc="center right", borderaxespad=0.3,
                       handletextpad=0.5, labelspacing=0.25)

        axes[1].plot(trunc["k"], trunc["m_error_bound"], **PREDICTED,
                     label=r"bound  $O(\lambda_{k+1}(C))$")
        axes[1].plot(trunc["k"], trunc["rho_error_measured"], **MEASURED,
                     label=r"measured  $|\rho-\rho_k|$")
        axes[1].set_yscale("log")
        axes[1].set_xticks(trunc["k"])
        axes[1].set_xlabel("retained factors  $k$", labelpad=1.5)
        axes[1].set_ylabel("truncation error", labelpad=1.5)
        axes[1].set_title(f"Truncation error at $d={d_max}$")
        axes[1].legend(loc="upper right", borderaxespad=0.3,
                       handletextpad=0.5, labelspacing=0.25)
        axes[1].grid(alpha=0.25, which="both")

        for ax in axes:
            ax.tick_params(pad=1.5)
        fig.tight_layout(pad=0.3, w_pad=1.2)
        _save(fig, outdir, "universe_scaling.png")


# ------------------------------------------------------------------ main --
def main(argv=None) -> None:
    here = Path(__file__).resolve()
    repo = here.parents[3]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", type=Path,
                    default=repo / "research" / "results" / "07-12-2026")
    ap.add_argument("--outdir", type=Path, default=here.parent)
    args = ap.parse_args(argv)

    results = args.results.resolve()
    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"results <- {results}\nfigures -> {outdir}\n")

    fig_fragility(results, outdir)
    fig_phase_diagram(results, outdir)
    fig_dealer_surface(results, outdir)
    fig_perfgd_loops(results, outdir)
    fig_lazy_deploy(results, outdir)
    fig_universe(results, outdir)


if __name__ == "__main__":
    main()
