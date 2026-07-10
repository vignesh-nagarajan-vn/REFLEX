"""Multi-dealer systemic risk: the (N, f) phase diagram + genuine-market probes.

Three layers:

1. **Analytic surface (always).**  The joint modulus ``m_N = N_eff * m_1`` on
   an ``(N, toxicity_feedback)`` grid, with the systemic boundary ``m_N = 1``
   -- competition destabilises a factor ``N_eff`` before any single dealer
   would (theory 1.3).
2. **Simulated joint cobweb (always).**  The joint best-response iteration on
   the *genuine* shared-pool market (:mod:`reflex.env.multi_dealer`).
3. **CRN joint-modulus probes (``--probe``).**  Measured in-phase (common-mode)
   and anti-phase (differential) moduli on the real environment vs their
   closed-form predictions ``N_eff * m_1`` and ``(1-kappa) * m_1``.

Usage::

    python -m experiments.run_dealers --config configs/default.yaml [--probe]
"""

from __future__ import annotations

import argparse
import copy
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from reflex.analysis.phase import dealer_phase_grid
from reflex.config import load_config
from reflex.equilibrium import measure_joint_modulus_sim, run_joint_cobweb_sim
from reflex.theory.multi_dealer import multi_dealer_boundary, n_eff


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Multi-dealer (N, f) phase diagram.")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--outdir", default="outputs")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--kappa", type=float, default=1.0)
    ap.add_argument("--n-values", type=int, nargs="+", default=[1, 2, 3, 5, 8, 12])
    ap.add_argument("--grid", type=float, nargs="+",
                    default=[0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0])
    ap.add_argument("--probe", action="store_true",
                    help="also run the CRN joint-modulus probes on the real env")
    ap.add_argument("--episodes", type=int, default=4)
    args = ap.parse_args(argv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    base = load_config(args.config)
    base.clients.toxic_spillover = float(args.kappa)

    # ---- layer 1: analytic (N, f) surface ---------------------------------- #
    print(f"=== analytic joint modulus m_N on (N, f) grid (kappa={args.kappa}) ===")
    grid = dealer_phase_grid(base, args.n_values, args.grid, kappa=args.kappa)
    csv_path = outdir / "dealer_phase_grid.csv"
    with open(csv_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["n_dealers"] + [f"f={f}" for f in args.grid])
        for i, n in enumerate(args.n_values):
            writer.writerow([n] + [f"{v:.4f}" for v in grid[i]])
    print(f"saved grid -> {csv_path}")

    mb = multi_dealer_boundary(base, n_dealers=max(args.n_values), kappa=args.kappa)
    print(f"at N={max(args.n_values)}: N_eff={mb.n_eff:.1f}, m_N={mb.m_N:.3f}, "
          f"critical dealer count N_c={mb.n_critical:.1f}")

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    im = ax.imshow(
        grid, aspect="auto", origin="lower", cmap="RdYlGn_r", vmin=0.0, vmax=2.0,
        extent=(min(args.grid), max(args.grid), 0, len(args.n_values)),
    )
    ax.set_yticks([i + 0.5 for i in range(len(args.n_values))])
    ax.set_yticklabels(args.n_values)
    cs = ax.contour(
        np.linspace(min(args.grid), max(args.grid), grid.shape[1]),
        [i + 0.5 for i in range(len(args.n_values))],
        grid, levels=[1.0], colors="black", linewidths=1.5,
    )
    ax.clabel(cs, fmt="m_N = 1")
    ax.set_xlabel("toxicity_feedback f"); ax.set_ylabel("number of dealers N")
    ax.set_title(f"Systemic stability surface m_N = N_eff * m_1 (kappa={args.kappa})")
    fig.colorbar(im, label="joint modulus m_N")
    fig.tight_layout()
    fig_path = outdir / "dealer_phase_diagram.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"saved phase diagram -> {fig_path}")

    # ---- layer 2: simulated joint cobweb on the genuine market ------------- #
    cfg = copy.deepcopy(base)
    cfg.clients.n_dealers = 3
    print("\n=== simulated joint cobweb (N=3, genuine shared-pool market) ===")
    cw = run_joint_cobweb_sim(cfg, n_iters=10, seed=args.seed, n_episodes=args.episodes)
    print(f"common-mode iterates: {np.array2string(cw.common_mode, precision=3)}")
    print(f"converged: {cw.converged}")

    # ---- layer 3: measured joint moduli vs prediction ---------------------- #
    if args.probe:
        print("\n=== CRN joint-modulus probes on the real environment ===")
        rows = []
        m1_ref = None
        for n in (1, 2, 3):
            cfg = copy.deepcopy(base)
            cfg.clients.n_dealers = n
            res = measure_joint_modulus_sim(
                cfg, h_ref=1.0, seed=args.seed, n_episodes=args.episodes
            )
            if n == 1:
                m1_ref = res.modulus_common
            pred = n_eff(n, args.kappa) * (m1_ref if m1_ref else float("nan"))
            rows.append({
                "n_dealers": n,
                "modulus_common_measured": res.modulus_common,
                "modulus_common_predicted": pred,
                "modulus_differential": res.modulus_differential,
            })
            print(f"  N={n}: common={res.modulus_common:.3f} (pred {pred:.3f})  "
                  f"diff={res.modulus_differential:.3f}")
        probe_csv = outdir / "dealer_joint_moduli.csv"
        with open(probe_csv, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"saved probes -> {probe_csv}")


if __name__ == "__main__":
    main()
