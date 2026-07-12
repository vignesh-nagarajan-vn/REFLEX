"""Run a single RRM experiment: one trajectory, the BR-slope modulus, market
metrics, and a saved trajectory plot.

Usage::

    python -m experiments.run_single --config configs/default.yaml --seed 0
    endo-run-single --config configs/default.yaml --outdir outputs
"""

from __future__ import annotations

import argparse
from pathlib import Path

from reflex.analysis import (
    classify_run,
    compute_metrics,
    empirical_lipschitz,
)
from reflex.estimators import measure_response_modulus
from reflex.analysis.plots import plot_rrm_trajectory
from reflex.config import load_config
from reflex.env import StructuralSimulator
from reflex.equilibrium import run_loop
from reflex.policy import build_policy


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Run a single outer-loop experiment.")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--mode", default=None,
                    help="loop mode: rrm | perfgd_analytic | perfgd_learned "
                         "(default: config rrm.loop_mode)")
    ap.add_argument("--outdir", default="outputs")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    seed = args.seed if args.seed is not None else cfg.seed
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    eps = cfg.clients.toxicity_feedback
    mode = args.mode or cfg.rrm.loop_mode
    print(f"=== single {mode} run: alpha={cfg.clients.alpha}, "
          f"epsilon(toxicity_feedback)={eps}, seed={seed} ===")

    # 1) Outer-loop trajectory (with the ML<->math seam diagnostics).
    loop_res = run_loop(cfg, mode=mode, seed=seed, verbose=True)
    traj = loop_res.trajectory
    if loop_res.diagnostics:
        d = loop_res.diagnostics[-1]
        print(
            f"seam diagnostics (final iter): learned d[adv]/dh={d.learned_toxic_slope:.4f}  "
            f"analytic -psi*eps(h)={d.analytic_toxic_slope:.4f}  window={d.window_len}"
        )
    h = traj.central_spreads
    L_traj = empirical_lipschitz(h, burn_in=1)
    cls = classify_run(h, tol=cfg.rrm.tol)
    print(f"\ntrajectory: stop={traj.stop_reason}  trajectory-L_hat={L_traj:.3f}  class={cls}")

    # 2) Robust modulus via the best-response-slope probe.
    mod = measure_response_modulus(cfg, seed=seed)
    print(
        f"best-response modulus m_hat={mod.modulus:.3f}  "
        f"(BR(h+d)={mod.br_plus:.3f}, BR(h-d)={mod.br_minus:.3f})  "
        f"=> {'UNSTABLE (diverges)' if mod.modulus > 1 else 'stable (contracts)'}"
    )

    # 3) Market metrics under the final deployed policy.
    sim = StructuralSimulator(cfg)
    pol = build_policy(cfg.policy)
    pol.load_flat(__import__("torch").tensor(traj.iterates[-1].flat_params))
    m = compute_metrics(cfg, pol, seed=seed, simulator=sim)
    print(
        f"market metrics: realized_h={m.realized_half_spread:.3f}  "
        f"toxicity_share={m.toxicity_share:.3f}  fill_rate={m.fill_rate:.3f}  "
        f"adverse/step={m.adverse_loss_per_step:.3f}  HHI={m.hhi:.3f}"
    )

    if not args.no_plot:
        p = plot_rrm_trajectory(
            traj, outdir / f"rrm_trajectory_a{cfg.clients.alpha}_eps{eps}_s{seed}.png"
        )
        print(f"saved trajectory plot -> {p}")


if __name__ == "__main__":
    main()
