"""Budget sensitivity of the measured BR-slope modulus (lazy-deploy study).

Substantiates the audit finding that the measured modulus is a property of
the finite-budget retraining map: at a fixed probe point, sweep the
per-deployment optimisation budget (policy.inner_steps -- the lazy-deploy
knob K of theory 1.1 Section 6.2) and record m_hat.  Also the roadmap item
"Sweep lazy-deploy K and report effect on gamma_eff".

Run from inside archive/endo_market_v3/ with the repo venv:

    ../.venv/Scripts/python -u ../research/analysis/src/budget_sensitivity.py \
        --outdir ../research/analysis/figures
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

from reflex.config import load_config
from reflex.estimators import measure_response_modulus
from reflex.theory.analytic_boundary import (
    beta as beta_of,
    epsilon as epsilon_of,
    gamma as gamma_of,
    reference_state,
)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Measured modulus vs optimizer budget.")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--outdir", default="../research/analysis/figures")
    ap.add_argument("--h-ref", type=float, default=1.0)
    ap.add_argument("--gains", type=float, nargs="+", default=[2.0, 5.0])
    ap.add_argument("--budgets", type=int, nargs="+",
                    default=[15, 30, 60, 120, 240, 480])
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args(argv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    base = load_config(args.config)
    ref = reference_state(base)
    rows = []
    print("=== measured modulus vs per-deployment optimisation budget ===")
    for f in args.gains:
        cfg_f = copy.deepcopy(base)
        cfg_f.clients.toxicity_feedback = float(f)
        m_pred = (epsilon_of(cfg_f, args.h_ref, ref) * beta_of(cfg_f)
                  / gamma_of(cfg_f, args.h_ref, ref))
        for steps in args.budgets:
            t0 = time.time()
            ms = []
            for s in range(args.seeds):
                cfg = copy.deepcopy(cfg_f)
                cfg.policy.inner_steps = int(steps)
                r = measure_response_modulus(cfg, seed=s, h_ref=args.h_ref, delta=0.25)
                ms.append(r.modulus)
            ms_sorted = sorted(ms)
            med = ms_sorted[len(ms_sorted) // 2]
            rows.append({
                "toxicity_feedback": f,
                "inner_steps": steps,
                "m_median": med,
                "m_all": ";".join(f"{m:.4f}" for m in ms),
                "m_pred_structural": m_pred,
            })
            print(f"  f={f:4.1f} steps={steps:4d}: median m_hat={med:.4f} "
                  f"(structural m={m_pred:.3f})  ({time.time()-t0:.0f}s)")

    csv_path = outdir / "budget_sensitivity.csv"
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"saved -> {csv_path}")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for f, style in zip(args.gains, ("o-", "s-")):
        pts = [r for r in rows if r["toxicity_feedback"] == f]
        ax.plot([r["inner_steps"] for r in pts], [r["m_median"] for r in pts],
                style, label=f"measured (f={f})")
        ax.axhline(pts[0]["m_pred_structural"], ls=":", lw=1.0, color="gray")
    ax.axhline(1.0, color="black", ls="--", lw=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("per-deployment optimisation budget (inner steps, lazy-deploy K)")
    ax.set_ylabel("measured BR-slope modulus m_hat")
    ax.set_title("The measured modulus is a property of the finite-budget "
                 "retraining map", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig_path = outdir / "budget_sensitivity.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"saved -> {fig_path}")


if __name__ == "__main__":
    main()
