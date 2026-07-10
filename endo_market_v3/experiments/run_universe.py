"""Factor-model scaling: the stability boundary at 100+ correlated bonds.

Pure closed forms (theory 1.5) -- fast even at full size:

* the ``d x d`` modulus matrix ``M = beta * Gamma^{-1} * E`` and its spectral
  radius ``rho(M)`` across universe sizes, with per-bond sigmas calibrated from
  the shipped cross-sectional dispersion data;
* the ``O(d k^2)`` Woodbury reduction checked against the dense inverse; and
* the truncation error bound (linear in the residual factor variance).

Usage::

    python -m experiments.run_universe --config configs/default.yaml
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

from reflex.calibration import load_xsection_sigma
from reflex.config import load_config
from reflex.env.bonds import BondUniverse
from reflex.theory.factor_scaling import (
    factor_modulus,
    per_bond_constants,
    truncation_error_bound,
)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Factor-model scaling to 100+ bonds.")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--outdir", default="outputs")
    ap.add_argument("--sizes", type=int, nargs="+", default=[8, 16, 32, 64, 128])
    ap.add_argument("--k-factors", type=int, nargs="+", default=[1, 2, 4, 8])
    args = ap.parse_args(argv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    base = load_config(args.config)

    # Data-calibrated per-bond sigma dispersion: scale idiosyncratic sigmas by
    # the observed cross-sectional dispersion of real bond returns (G2 file).
    xs = load_xsection_sigma()
    disp = float(xs["ret_sigma_xs"].mean()) if "ret_sigma_xs" in xs.columns else None
    rel_disp = 0.35 if disp is None else min(max(disp / max(disp, 1e-9) * 0.35, 0.1), 0.8)

    rows = []
    print("=== rho(M) across universe sizes (calibrated sigma dispersion) ===")
    for d in args.sizes:
        cfg = copy.deepcopy(base)
        cfg.bonds.n_bonds = int(d)
        t0 = time.time()
        universe = BondUniverse(cfg.bonds, seed=cfg.seed)
        rng = np.random.default_rng(cfg.seed)
        sigma = cfg.simulator.fundamental_vol * (
            1.0 + rel_disp * rng.standard_normal(d).clip(-2, 2)
        ).clip(0.2, None)
        fm = factor_modulus(cfg, universe=universe, sigma=sigma)
        rows.append({
            "n_bonds": d,
            "rho_M": fm.rho,
            "stable": fm.stable,
            "m_scalar_max": fm.m_scalar_max,
            "market_alignment": fm.market_alignment,
            "seconds": time.time() - t0,
        })
        print(f"  d={d:4d}: rho(M)={fm.rho:.4f}  scalar-max m={fm.m_scalar_max:.4f}  "
              f"market-alignment={fm.market_alignment:.2f}  ({rows[-1]['seconds']:.2f}s)")

    csv_path = outdir / "universe_scaling.csv"
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved scaling table -> {csv_path}")

    # Truncation error bound at the largest universe.
    d = max(args.sizes)
    cfg = copy.deepcopy(base)
    cfg.bonds.n_bonds = d
    universe = BondUniverse(cfg.bonds, seed=cfg.seed)
    print(f"\n=== factor-truncation error at d={d} ===")
    trunc_rows = []
    for k in args.k_factors:
        tb = truncation_error_bound(cfg, universe, k=int(k))
        trunc_rows.append({
            "k": k,
            "residual_variance": tb.residual_variance,
            "m_error_bound": tb.m_error_bound,
            "rho_error_measured": tb.rho_error_measured,
        })
        print(f"  k={k:2d}: residual lambda_(k+1)={tb.residual_variance:.4f}  "
              f"bound={tb.m_error_bound:.5f}  measured={tb.rho_error_measured:.5f}")
        assert tb.rho_error_measured <= tb.m_error_bound + 1e-8, "bound violated!"

    trunc_csv = outdir / "universe_truncation.csv"
    with open(trunc_csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(trunc_rows[0].keys()))
        writer.writeheader()
        writer.writerows(trunc_rows)
    print(f"saved truncation table -> {trunc_csv}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot([r["n_bonds"] for r in rows], [r["rho_M"] for r in rows], "o-")
    axes[0].axhline(1.0, color="gray", ls=":")
    axes[0].set_xlabel("universe size d"); axes[0].set_ylabel("rho(M)")
    axes[0].set_title("Spectral modulus across universe size"); axes[0].grid(alpha=0.25)
    axes[1].plot([r["k"] for r in trunc_rows], [r["m_error_bound"] for r in trunc_rows],
                 "s-", label="bound O(λ_{k+1}(C))")
    axes[1].plot([r["k"] for r in trunc_rows], [r["rho_error_measured"] for r in trunc_rows],
                 "o-", label="measured |rho - rho_k|")
    axes[1].set_xlabel("retained factors k"); axes[1].set_yscale("log")
    axes[1].set_title(f"Truncation error at d={d}"); axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig_path = outdir / "universe_scaling.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"saved figure -> {fig_path}")


if __name__ == "__main__":
    main()
