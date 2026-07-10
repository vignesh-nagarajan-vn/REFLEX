"""Triangulate the distribution sensitivity epsilon three independent ways.

Runs the BR-slope, Sinkhorn/Wasserstein and CKS flow-curve estimators at the
analytic fixed point and reports them next to the closed-form epsilon of
theory 1.1 -- the evidentiary protocol for the analytic boundary.

Usage::

    python -m experiments.run_triangulation --config configs/default.yaml --episodes 8
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from reflex.config import load_config
from reflex.estimators import triangulate_epsilon


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Three-way epsilon triangulation.")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--outdir", default="outputs")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--h-ref", type=float, default=None,
                    help="probe spread (default: the quoting anchor, where the "
                         "retraining loop operates)")
    args = ap.parse_args(argv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(args.config)
    print("=== epsilon triangulation (three measured legs vs the closed form) ===")
    res = triangulate_epsilon(cfg, h_ref=args.h_ref, seed=args.seed,
                              n_episodes=args.episodes)

    print(f"reference spread h_ref = {res.h_ref:.4f} (quoting anchor / operating point)")
    print(f"  analytic  epsilon (a-priori A2 state)  = {res.epsilon_analytic:8.4f}")
    print(f"  analytic  epsilon (realized state, S9) = {res.epsilon_analytic_realized:8.4f}"
          f"   (rho={res.realized_rho:.2f}, |g|={res.realized_mispricing:.3f})")
    print(f"  BR-slope  epsilon_hat    = {res.epsilon_br:8.4f}   (m_hat={res.modulus_br:.3f} x gamma_real/beta)")
    print(f"  Sinkhorn  epsilon_hat    = {res.epsilon_sinkhorn:8.4f}   (W1 rate of induced toxic flow)")
    print(f"  CKS       epsilon_hat    = {res.epsilon_cks:8.4f}   (fitted flow-curve slope)")
    print(f"  max pairwise spread across measured legs: {res.max_relative_spread:.2%}")

    csv_path = outdir / "epsilon_triangulation.csv"
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(res.as_dict().keys()))
        writer.writeheader()
        writer.writerow(res.as_dict())
    print(f"saved -> {csv_path}")


if __name__ == "__main__":
    main()
