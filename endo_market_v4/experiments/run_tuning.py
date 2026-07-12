"""Estimator tuning: the Sinkhorn blur and the robust ambiguity radius (v4).

Discharges the two "tune ..." items of the research to-do:

1. **Sinkhorn entropic regularisation.**  In 1-D the exact quantile ``W1`` is
   available, so the debiased-divergence bias is directly measurable.  The
   experiment traces the U-shaped bias curve over a *scale-relative* blur grid
   on (a) synthetic location-shift pairs with known ``W1`` (validates the
   tuner itself) and (b) the config's own CRN toxic-flow samples (the
   operational tuning), and reports the bias-minimising relative blur baked
   into ``reflex.estimators.sinkhorn.TUNED_REL_REG`` (the ``reg="auto"``
   path of ``estimate_epsilon_sinkhorn``).

2. **Robust ambiguity radius.**  Monte-Carlo *frequentist coverage* of the
   ``z*s`` radius (1.4 section 3.1) vs the v4-calibrated
   ``max(z*s, empirical quantile)`` radius (``reflex.theory.robust.
   calibrate_radius``) under normal, heavy-tailed (Student-t) and skewed
   (lognormal) estimate distributions -- plus the same calibration run on the
   *actual* CRN modulus-probe estimates of the config, answering "is z*s
   well-calibrated for our probe".

Usage::

    python -m experiments.run_tuning --config configs/default.yaml
    python -m experiments.run_tuning --config configs/smoke.yaml \
        --episodes 2 --probe-seeds 2 --mc 100
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

from reflex.config import load_config
from reflex.estimators.sinkhorn import (
    TUNED_REL_REG,
    tune_sinkhorn_reg,
    tune_sinkhorn_reg_for_config,
)
from reflex.theory.robust import calibrate_radius, measure_modulus_estimates

REL_GRID = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]


def _coverage_mc(sampler, true_mean: float, n_per: int, n_mc: int,
                 confidence: float, rng: np.random.Generator) -> tuple:
    """One-sided frequentist coverage of the two radii under a known truth."""
    z = float(norm.ppf(confidence))
    cov_norm = 0
    cov_cal = 0
    mult = []
    for _ in range(int(n_mc)):
        x = sampler(n_per, rng)
        mean = float(x.mean())
        std = float(x.std(ddof=1))
        z_rad = z * std
        q_rad = max(float(np.quantile(x - mean, confidence)), 0.0)
        cal_rad = max(z_rad, q_rad)
        if true_mean <= mean + z_rad:
            cov_norm += 1
        if true_mean <= mean + cal_rad:
            cov_cal += 1
        mult.append(q_rad / z_rad if z_rad > 0 else np.nan)
    return cov_norm / n_mc, cov_cal / n_mc, float(np.nanmedian(mult))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Tune the Sinkhorn blur and the robust radius.")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--outdir", default="outputs")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--episodes", type=int, default=8,
                    help="CRN episodes for the config-sample Sinkhorn tuning")
    ap.add_argument("--h-ref", type=float, default=None,
                    help="probe spread (default: reward.quote_anchor_ref)")
    ap.add_argument("--probe-seeds", type=int, default=6,
                    help="seeds for the probe-based radius calibration (0 = skip)")
    ap.add_argument("--mc", type=int, default=400, help="Monte-Carlo replications")
    ap.add_argument("--n-per", type=int, default=6, help="estimates per replication")
    ap.add_argument("--confidence", type=float, default=0.95)
    args = ap.parse_args(argv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    cfg = load_config(args.config)
    h_ref = float(cfg.reward.quote_anchor_ref if args.h_ref is None else args.h_ref)

    # ---------------- 1. Sinkhorn blur ------------------------------------- #
    print("=== Sinkhorn blur tuning (bias vs the exact 1-D quantile W1) ===")
    print("synthetic location shift (true W1 = |mean shift| = 0.30):")
    x = rng.normal(0.0, 1.0, size=1200)
    y = rng.normal(0.3, 1.0, size=1200)
    tun_syn = tune_sinkhorn_reg(x, y, rel_grid=REL_GRID)
    for r, d, b in zip(tun_syn.rel_grid, tun_syn.divergences, tun_syn.rel_bias):
        print(f"  rel_reg={r:5.2f}: S={d:.4f}  rel_bias={b:6.1%}")
    print(f"  exact W1 = {tun_syn.w1_exact:.4f}; best rel_reg = {tun_syn.best_rel_reg} "
          f"(bias {tun_syn.best_rel_bias:.1%})")

    print(f"\nconfig CRN toxic samples at h_ref={h_ref:.3f} "
          f"({args.episodes} episodes/side):")
    t0 = time.time()
    tun_cfg = tune_sinkhorn_reg_for_config(
        cfg, h_ref=h_ref, seed=args.seed, n_episodes=args.episodes, rel_grid=REL_GRID
    )
    for r, d, b in zip(tun_cfg.rel_grid, tun_cfg.divergences, tun_cfg.rel_bias):
        print(f"  rel_reg={r:5.2f}: S={d:.4f}  rel_bias={b:6.1%}")
    print(f"  exact W1 = {tun_cfg.w1_exact:.4f}; best rel_reg = {tun_cfg.best_rel_reg} "
          f"(bias {tun_cfg.best_rel_bias:.1%}) ({time.time()-t0:.0f}s)")
    print(f"  baked-in default TUNED_REL_REG = {TUNED_REL_REG} "
          f"(reg='auto' in estimate_epsilon_sinkhorn)")

    sink_csv = outdir / "sinkhorn_tuning.csv"
    with open(sink_csv, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["case", "rel_reg", "abs_reg", "divergence", "rel_bias",
                         "w1_exact", "scale", "best_rel_reg"])
        for name, tun in (("synthetic", tun_syn), ("config_crn", tun_cfg)):
            for r, a, d, b in zip(tun.rel_grid, tun.abs_grid, tun.divergences, tun.rel_bias):
                writer.writerow([name, r, a, d, b, tun.w1_exact, tun.scale, tun.best_rel_reg])
    print(f"saved -> {sink_csv}")

    # ---------------- 2. Robust ambiguity radius --------------------------- #
    print("\n=== Robust-radius calibration (one-sided coverage at "
          f"{args.confidence:.0%} nominal) ===")
    def _contaminated(n, g):
        # rare far-outlier seeds: the railed-probe / bifurcation pattern
        x = g.normal(0.5, 0.05, size=n)
        x[g.random(size=n) < 0.06] = 1.1
        return x

    cases = {
        "normal": (lambda n, g: g.normal(0.5, 0.1, size=n), 0.5),
        "student_t(2.5)": (lambda n, g: 0.5 + 0.1 * g.standard_t(2.5, size=n), 0.5),
        "lognormal": (lambda n, g: g.lognormal(-0.75, 0.5, size=n),
                      float(np.exp(-0.75 + 0.5 ** 2 / 2.0))),
        "contaminated(6%)": (_contaminated, 0.5 * 0.94 + 1.1 * 0.06),
    }
    rad_rows = []
    for name, (sampler, true_mean) in cases.items():
        cov_n, cov_c, mult = _coverage_mc(
            sampler, true_mean, args.n_per, args.mc, args.confidence, rng
        )
        rad_rows.append({"case": name, "coverage_normal": cov_n,
                         "coverage_calibrated": cov_c, "median_multiplier": mult})
        print(f"  {name:16s}: coverage z*s = {cov_n:.3f}, calibrated = {cov_c:.3f} "
              f"(median multiplier {mult:.2f})")

    if args.probe_seeds > 0:
        print(f"\nprobe-based calibration ({args.probe_seeds} CRN modulus probes):")
        t0 = time.time()
        ms = measure_modulus_estimates(
            cfg, seeds=[args.seed + s for s in range(args.probe_seeds)], h_ref=h_ref
        )
        cal = calibrate_radius(ms, confidence=args.confidence)
        print(f"  estimates: {', '.join(f'{m:.3f}' for m in ms)} ({time.time()-t0:.0f}s)")
        print(f"  z*s radius = {cal.z_radius:.4f}; quantile radius = {cal.quantile_radius:.4f} "
              f"(multiplier {cal.multiplier:.2f})")
        print(f"  bootstrap coverage: z*s = {cal.coverage_normal:.3f}, "
              f"calibrated = {cal.coverage_calibrated:.3f}")
        verdict = ("z*s is adequately calibrated for the CRN probe"
                   if cal.multiplier <= 1.1 else
                   "heavy-tailed probe estimates: use the calibrated radius")
        print(f"  -> {verdict}")
        rad_rows.append({"case": "crn_probe(bootstrap)",
                         "coverage_normal": cal.coverage_normal,
                         "coverage_calibrated": cal.coverage_calibrated,
                         "median_multiplier": cal.multiplier})

    rad_csv = outdir / "radius_calibration.csv"
    with open(rad_csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rad_rows[0].keys()))
        writer.writeheader()
        writer.writerows(rad_rows)
    print(f"saved -> {rad_csv}")

    # ---------------- figure ----------------------------------------------- #
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(tun_syn.rel_grid, tun_syn.rel_bias, "o-", label="synthetic (W1 known)")
    axes[0].plot(tun_cfg.rel_grid, tun_cfg.rel_bias, "s-", label="config CRN samples")
    axes[0].axvline(TUNED_REL_REG, color="tab:red", ls="--", lw=0.9,
                    label=f"tuned default {TUNED_REL_REG}")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("entropic blur / sample std")
    axes[0].set_ylabel("|S_reg - W1_exact| / W1_exact")
    axes[0].set_title("Sinkhorn blur: the U-shaped bias curve", fontsize=10)
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.25, which="both")

    names = [r["case"] for r in rad_rows]
    xpos = np.arange(len(names))
    axes[1].bar(xpos - 0.18, [r["coverage_normal"] for r in rad_rows], 0.36,
                label="z*s radius", color="tab:blue", alpha=0.8)
    axes[1].bar(xpos + 0.18, [r["coverage_calibrated"] for r in rad_rows], 0.36,
                label="calibrated radius", color="tab:green", alpha=0.8)
    axes[1].axhline(args.confidence, color="tab:red", ls="--", lw=0.9,
                    label=f"nominal {args.confidence:.0%}")
    axes[1].set_xticks(xpos)
    axes[1].set_xticklabels(names, fontsize=7, rotation=12)
    axes[1].set_ylim(0.5, 1.02)
    axes[1].set_ylabel("one-sided coverage")
    axes[1].set_title("Ambiguity-radius calibration", fontsize=10)
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig_path = outdir / "estimator_tuning.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"saved figure -> {fig_path}")


if __name__ == "__main__":
    main()
