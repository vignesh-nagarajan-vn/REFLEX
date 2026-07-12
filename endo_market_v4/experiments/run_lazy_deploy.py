"""Lazy-deploy sweep: the K-step RGD map vs the closed form (theory 1.6).

Sweeps the number of inner gradient steps per deployment ``K`` (the
``rrm.rgd_steps`` knob) and measures the **signed** K-step deployment-map slope
``mu_hat(K)`` with the CRN probe
(:func:`reflex.estimators.br_slope.measure_rgd_response`).  The closed form of
``research/math-theory/06-lazy-deployment.md`` predicts

    mu(K) = -m + c^K * (1 + m),

with ``m`` the exact-BR cobweb modulus (measured independently by the full BR
probe as the ``K -> infinity`` anchor) and ``c`` the inner per-step contraction
in spread units -- not knowable a priori from ``rrm.rgd_lr`` (parameter space),
so it is fitted from the measured curve (one parameter).  Reported alongside:
the effective curvature ``gamma_eff(K) = gamma * m / |mu(K)|`` -- the
sweep's answer to "what does laziness do to the effective gamma".

Usage::

    python -m experiments.run_lazy_deploy --config configs/default.yaml
    python -m experiments.run_lazy_deploy --config configs/smoke.yaml \
        --k-grid 1 3 8 --seeds 1 --episodes 2
"""

from __future__ import annotations

import argparse
import copy
import csv
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from reflex.config import load_config
from reflex.estimators.br_slope import measure_response_modulus, measure_rgd_response
from reflex.theory.analytic_boundary import gamma as gamma_of, reference_state
from reflex.theory.lazy_deploy import (
    deadbeat_k,
    fit_inner_contraction,
    gamma_eff,
    k_step_slope,
    lazy_deploy_curve,
    max_stable_k,
)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Lazy-deploy K sweep vs the theory-06 closed form.")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--outdir", default="outputs")
    ap.add_argument("--seed", type=int, default=0, help="base seed")
    ap.add_argument("--seeds", type=int, default=3, help="number of probe seeds per K")
    ap.add_argument("--k-grid", type=int, nargs="+", default=[1, 2, 3, 5, 8, 12, 20])
    ap.add_argument("--h-ref", type=float, default=None,
                    help="probe spread (default: reward.quote_anchor_ref, the operating spread)")
    ap.add_argument("--delta-frac", type=float, default=0.25,
                    help="probe half-width as a fraction of h_ref")
    ap.add_argument("--lr", type=float, default=None,
                    help="inner RGD step size (default: rrm.rgd_lr)")
    ap.add_argument("--episodes", type=int, default=None,
                    help="override rrm.n_episodes for faster probes")
    args = ap.parse_args(argv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(args.config)
    if args.episodes is not None:
        cfg.rrm.n_episodes = int(args.episodes)
    h_ref = float(cfg.reward.quote_anchor_ref if args.h_ref is None else args.h_ref)
    delta = args.delta_frac * h_ref
    lr = float(cfg.rrm.rgd_lr if args.lr is None else args.lr)
    seeds = [args.seed + s for s in range(int(args.seeds))]
    k_grid = sorted(set(int(k) for k in args.k_grid))

    # ---- the K -> infinity anchor: the full-BR modulus ---------------------- #
    print(f"=== lazy-deploy sweep at h_ref={h_ref:.3f} (delta={delta:.3f}, lr={lr}) ===")
    print("full-BR anchor (K -> infinity):")
    m_full = []
    for s in seeds:
        t0 = time.time()
        res = measure_response_modulus(cfg, seed=s, h_ref=h_ref, delta=delta)
        m_full.append(res.modulus)
        print(f"  seed {s}: m_hat = {res.modulus:.4f} ({time.time()-t0:.0f}s)")
    m_anchor = float(np.median(m_full))
    print(f"  median m_hat = {m_anchor:.4f}")

    # ---- the K grid ---------------------------------------------------------- #
    rows = []
    medians = {}
    for k in k_grid:
        slopes = []
        t0 = time.time()
        for s in seeds:
            r = measure_rgd_response(cfg, seed=s, h_ref=h_ref, delta=delta, n_steps=k, lr=lr)
            slopes.append(r.slope)
            rows.append({
                "K": k, "seed": s, "slope": r.slope, "modulus": r.modulus,
                "out_plus": r.out_plus, "out_minus": r.out_minus,
                "h_ref": h_ref, "delta": delta, "lr": lr,
            })
        medians[k] = float(np.median(slopes))
        print(f"  K={k:3d}: median mu_hat = {medians[k]:+.4f} "
              f"(seeds: {', '.join(f'{x:+.3f}' for x in slopes)}) ({time.time()-t0:.0f}s)")

    # ---- fit the inner contraction c and assemble the closed-form curve ------ #
    ks = np.array(k_grid, dtype=float)
    mu_meds = np.array([medians[k] for k in k_grid])
    c_fit = fit_inner_contraction(ks, mu_meds, m_anchor)
    ref = reference_state(cfg)
    g_ref = gamma_of(cfg, h_ref, ref)
    curve = lazy_deploy_curve(cfg, k_grid, c_fit, h_eval=h_ref)
    print(f"\nfitted inner contraction c = {c_fit:.4f} "
          f"(lam_1 = {1.0 - c_fit:.3f} of the exact cobweb per deployment)")
    print(f"closed-form m at h_ref (a-priori state): {curve.m:.4f}; "
          f"measured anchor: {m_anchor:.4f}")
    db = deadbeat_k(m_anchor, c_fit)
    kmax = max_stable_k(m_anchor, c_fit)
    print(f"deadbeat K (measured anchor): {db:.2f}; max stable K: "
          f"{'inf (m <= 1)' if kmax == float('inf') else f'{kmax:.2f}'}")

    print("\n  K   mu_pred   mu_meas   gamma_eff/gamma")
    ge_meas = []
    for k in k_grid:
        mu_p = k_step_slope(m_anchor, c_fit, k)
        ge = gamma_eff(g_ref, m_anchor, c_fit, k)
        ge_meas.append(g_ref * m_anchor / max(abs(medians[k]), 1e-12))
        print(f"  {k:3d}  {mu_p:+.4f}  {medians[k]:+.4f}   {ge / g_ref:8.3f}")

    # ---- artifacts ------------------------------------------------------------ #
    csv_path = outdir / "lazy_deploy_sweep.csv"
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    summary_path = outdir / "lazy_deploy_summary.csv"
    with open(summary_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["K", "mu_median", "mu_pred_fit", "gamma_eff_over_gamma_meas",
                         "m_anchor", "c_fit", "h_ref", "lr"])
        for i, k in enumerate(k_grid):
            writer.writerow([k, medians[k], k_step_slope(m_anchor, c_fit, k),
                             ge_meas[i] / g_ref, m_anchor, c_fit, h_ref, lr])
    print(f"saved sweep -> {csv_path}")
    print(f"saved summary -> {summary_path}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    k_dense = np.linspace(min(k_grid), max(k_grid), 200)
    axes[0].plot(k_dense, [k_step_slope(m_anchor, c_fit, k) for k in k_dense],
                 "k-", lw=1.2, label=f"fit: -m + c^K(1+m), c={c_fit:.3f}")
    for k in k_grid:
        pts = [r["slope"] for r in rows if r["K"] == k]
        axes[0].plot([k] * len(pts), pts, "o", color="tab:blue", alpha=0.35, ms=4)
    axes[0].plot(k_grid, mu_meds, "s-", color="tab:blue", ms=5, label="measured median mu(K)")
    axes[0].axhline(-m_anchor, color="tab:red", ls="--", lw=0.9,
                    label=f"K->inf: -m = {-m_anchor:.3f}")
    axes[0].axhline(0.0, color="gray", ls=":", lw=0.8)
    if np.isfinite(db) and min(k_grid) <= db <= max(k_grid):
        axes[0].axvline(db, color="gray", ls=":", lw=0.8)
        axes[0].annotate(f"deadbeat K={db:.1f}", (db, 0.02), fontsize=7, rotation=90)
    axes[0].set_xlabel("inner steps per deployment K")
    axes[0].set_ylabel("deployment-map slope mu(K)")
    axes[0].set_title("Lazy deployment: the K-step map (theory 1.6)", fontsize=10)
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.25)

    axes[1].plot(k_dense, [gamma_eff(g_ref, m_anchor, c_fit, k) / g_ref for k in k_dense],
                 "k-", lw=1.2, label="predicted gamma_eff/gamma")
    axes[1].plot(k_grid, np.array(ge_meas) / g_ref, "s", color="tab:green", ms=5,
                 label="measured (from median |mu|)")
    axes[1].axhline(1.0, color="tab:red", ls="--", lw=0.9, label="gamma_eff = gamma (exact RRM)")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("inner steps per deployment K")
    axes[1].set_ylabel("gamma_eff / gamma")
    axes[1].set_title("Effective curvature of the lazy loop", fontsize=10)
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.25, which="both")
    fig.tight_layout()
    fig_path = outdir / "lazy_deploy_gamma_eff.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"saved figure -> {fig_path}")


if __name__ == "__main__":
    main()
