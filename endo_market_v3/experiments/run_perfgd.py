"""PerfGD vs blind RRM: stability beyond the boundary + the echo-chamber gap.

Two layers:

1. **Closed form (always).**  Across a ``toxicity_feedback`` grid: the RRM
   modulus, the boundary ``f*``, the echo-chamber decision/value gaps, and the
   exact 1-D dynamics (blind cobweb vs corrected ascent) at a gain beyond
   ``f*`` -- the "RRM diverges, PerfGD converges" figure of theory 1.2.
2. **Full ML loops (``--ml``).**  Run :func:`reflex.equilibrium.run_loop` in
   the three modes (rrm | perfgd_analytic | perfgd_learned) from a common seed
   and plot the central-spread iterates plus the learned-vs-analytic toxic
   slope (the ML<->math seam).

Usage::

    python -m experiments.run_perfgd --config configs/default.yaml [--ml] [--iters 8]
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

from reflex.config import load_config
from reflex.theory.perfgd import analyze_perfgd


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="PerfGD-vs-RRM stability + echo-chamber gap.")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--outdir", default="outputs")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--grid", type=float, nargs="+",
                    default=[0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
    ap.add_argument("--ml", action="store_true", help="also run the full ML loops")
    ap.add_argument("--iters", type=int, default=8, help="ML loop iterations")
    args = ap.parse_args(argv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    base = load_config(args.config)

    # ---- layer 1: closed-form scan + dynamics ----------------------------- #
    rows = []
    print("=== closed-form scan: RRM modulus & echo-chamber gap vs feedback gain ===")
    for f in args.grid:
        cfg = copy.deepcopy(base)
        cfg.clients.toxicity_feedback = float(f)
        res = analyze_perfgd(cfg, run_loops=False)
        rows.append(
            {
                "toxicity_feedback": f,
                "m_rrm": res.modulus_rrm,
                "eps_star": res.epsilon_star,
                "h_SP": res.h_sp,
                "h_PO": res.h_po,
                "decision_gap": res.gap.decision_gap_exact,
                "value_gap": res.gap.value_gap,
                "gamma_PO": res.gamma_po,
                "rrm_stable": res.rrm_stable,
            }
        )
        print(
            f"  f={f:5.2f}: m_rrm={res.modulus_rrm:6.3f} "
            f"h_SP={res.h_sp:.3f} h_PO={res.h_po:.3f} "
            f"gap={res.gap.decision_gap_exact:+.4f} "
            f"{'stable' if res.rrm_stable else 'UNSTABLE'}"
        )

    csv_path = outdir / "perfgd_gap_scan.csv"
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved gap scan -> {csv_path}")

    # Exact dynamics at a gain beyond the boundary.
    unstable = [r for r in rows if not r["rrm_stable"]]
    f_demo = unstable[0]["toxicity_feedback"] if unstable else args.grid[-1]
    cfg = copy.deepcopy(base)
    cfg.clients.toxicity_feedback = float(f_demo)
    demo = analyze_perfgd(cfg, run_loops=True, n_steps=60)
    print(
        f"\n1-D dynamics at f={f_demo} (m_rrm={demo.modulus_rrm:.2f}): "
        f"cobweb {'converged' if demo.rrm_converged else 'did NOT converge'}, "
        f"PerfGD {'converged' if demo.perfgd_converged else 'did NOT converge'}"
    )

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(demo.rrm_trajectory, "o-", ms=3, label="blind RRM cobweb")
    axes[0].plot(demo.perfgd_trajectory, "s-", ms=3, label="PerfGD (analytic)")
    axes[0].axhline(demo.h_sp, color="gray", ls=":", lw=0.8, label="h_SP")
    axes[0].axhline(demo.h_po, color="black", ls="--", lw=0.8, label="h_PO")
    axes[0].set_xlabel("iteration"); axes[0].set_ylabel("half-spread h")
    axes[0].set_title(f"Beyond the boundary (f={f_demo}, m={demo.modulus_rrm:.2f})")
    axes[0].legend(fontsize=8)
    fs = [r["toxicity_feedback"] for r in rows]
    axes[1].plot(fs, [r["decision_gap"] for r in rows], "o-", label="decision gap h_SP - h_PO")
    axes[1].plot(fs, [r["value_gap"] for r in rows], "s-", label="value gap (O(eps^2))")
    axes[1].set_xlabel("toxicity_feedback f"); axes[1].set_title("Echo-chamber gap")
    axes[1].legend(fontsize=8); axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig_path = outdir / "perfgd_vs_rrm.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"saved figure -> {fig_path}")

    # ---- layer 2: the full ML loops --------------------------------------- #
    if args.ml:
        from reflex.equilibrium import run_loop

        cfg = copy.deepcopy(base)
        cfg.clients.toxicity_feedback = float(f_demo)
        cfg.rrm.max_iters = int(args.iters)
        print(f"\n=== ML loops at f={f_demo} ({args.iters} iterations each) ===")
        results = {}
        for mode in ("rrm", "perfgd_analytic", "perfgd_learned"):
            results[mode] = run_loop(cfg, mode=mode, seed=args.seed, verbose=True)

        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for mode, style in (("rrm", "o-"), ("perfgd_analytic", "s-"), ("perfgd_learned", "^-")):
            axes[0].plot(
                results[mode].trajectory.central_spreads, style, ms=3, label=mode
            )
        axes[0].set_xlabel("deployment k"); axes[0].set_ylabel("central half-spread")
        axes[0].set_title("ML outer loops from a common seed"); axes[0].legend(fontsize=8)
        d = results["perfgd_learned"]
        axes[1].plot(d.learned_slopes, "^-", ms=3, label="learned d[adv]/dh (operator)")
        axes[1].plot(d.analytic_slopes, "k--", lw=1.0, label="analytic -psi*eps(h)")
        axes[1].set_xlabel("deployment k"); axes[1].set_title("The ML<->math seam")
        axes[1].legend(fontsize=8); axes[1].grid(alpha=0.25)
        fig.tight_layout()
        ml_path = outdir / "perfgd_ml_loops.png"
        fig.savefig(ml_path, dpi=150)
        plt.close(fig)
        print(f"saved ML-loop figure -> {ml_path}")

        ml_csv = outdir / "perfgd_ml_loops.csv"
        with open(ml_csv, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["mode", "k", "central_half_spread", "step_size",
                             "learned_slope", "analytic_slope"])
            for mode, res in results.items():
                slopes = res.learned_slopes
                aslopes = res.analytic_slopes
                for it in res.trajectory.iterates:
                    writer.writerow([
                        mode, it.k, it.central_half_spread, it.step_size,
                        slopes[it.k] if it.k < len(slopes) else np.nan,
                        aslopes[it.k] if it.k < len(aslopes) else np.nan,
                    ])
        print(f"saved ML-loop data -> {ml_csv}")


if __name__ == "__main__":
    main()
