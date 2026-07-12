"""Run the full REFLEX v4 experiment suite (smoke or full profile).

Profiles:

* ``smoke`` -- every experiment end to end on tiny settings (~15-25 min CPU).
  Proves the pipeline and produces real artifacts in ``outputs/``; not for
  scientific claims.
* ``full``  -- paper-grade settings (~10-15 min CPU measured on the current
  configs).  The same artifact names are overwritten with the real results.

Usage::

    python -m experiments.run_all --profile smoke
    python -m experiments.run_all --profile full
"""

from __future__ import annotations

import argparse
import time
import traceback
from pathlib import Path

from . import (
    run_calibrated,
    run_certificates,
    run_dealers,
    run_fragility,
    run_lazy_deploy,
    run_perfgd,
    run_single,
    run_sweep,
    run_triangulation,
    run_tuning,
    run_universe,
)

PROFILES = {
    "smoke": [
        ("certificates", run_certificates,
         ["--config", "configs/default.yaml", "--outdir", "outputs"]),
        ("fragility", run_fragility, ["--outdir", "outputs"]),
        ("calibrated", run_calibrated, ["--outdir", "outputs"]),
        ("universe", run_universe, ["--outdir", "outputs"]),
        ("perfgd", run_perfgd,
         ["--config", "configs/smoke.yaml", "--outdir", "outputs",
          "--grid", "0.5", "2.0", "6.0", "--ml", "--iters", "3"]),
        ("dealers", run_dealers,
         ["--config", "configs/smoke.yaml", "--outdir", "outputs",
          "--probe", "--episodes", "2",
          "--n-values", "1", "2", "3", "5",
          "--grid", "0.5", "1.0", "2.0", "4.0"]),
        ("triangulation", run_triangulation,
         ["--config", "configs/smoke.yaml", "--outdir", "outputs", "--episodes", "3"]),
        ("sweep", run_sweep,
         ["--config", "configs/sweep_feedback_smoke.yaml", "--outdir", "outputs"]),
        ("lazy_deploy", run_lazy_deploy,
         ["--config", "configs/smoke.yaml", "--outdir", "outputs",
          "--k-grid", "1", "3", "8", "--seeds", "1", "--episodes", "2"]),
        ("tuning", run_tuning,
         ["--config", "configs/smoke.yaml", "--outdir", "outputs",
          "--episodes", "2", "--probe-seeds", "2", "--mc", "100"]),
        ("single", run_single,
         ["--config", "configs/smoke.yaml", "--outdir", "outputs",
          "--mode", "perfgd_analytic"]),
    ],
    "full": [
        ("certificates", run_certificates,
         ["--config", "configs/default.yaml", "--outdir", "outputs"]),
        ("fragility", run_fragility, ["--outdir", "outputs"]),
        ("calibrated", run_calibrated, ["--outdir", "outputs", "--measure", "--seeds", "3"]),
        ("universe", run_universe, ["--outdir", "outputs"]),
        ("perfgd", run_perfgd,
         ["--config", "configs/default.yaml", "--outdir", "outputs", "--ml", "--iters", "10"]),
        ("dealers", run_dealers,
         ["--config", "configs/default.yaml", "--outdir", "outputs",
          "--probe", "--episodes", "8"]),
        ("triangulation", run_triangulation,
         ["--config", "configs/default.yaml", "--outdir", "outputs", "--episodes", "8"]),
        ("sweep", run_sweep, ["--config", "configs/sweep_feedback.yaml", "--outdir", "outputs"]),
        ("lazy_deploy", run_lazy_deploy,
         ["--config", "configs/default.yaml", "--outdir", "outputs"]),
        ("tuning", run_tuning,
         ["--config", "configs/default.yaml", "--outdir", "outputs"]),
        ("single", run_single,
         ["--config", "configs/default.yaml", "--outdir", "outputs",
          "--mode", "perfgd_analytic"]),
    ],
}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Run the full REFLEX v4 experiment suite.")
    ap.add_argument("--profile", choices=sorted(PROFILES), default="smoke")
    ap.add_argument("--only", nargs="+", default=None,
                    help="run only these experiment names")
    args = ap.parse_args(argv)

    Path("outputs").mkdir(exist_ok=True)
    plan = PROFILES[args.profile]
    if args.only:
        plan = [p for p in plan if p[0] in set(args.only)]
        if not plan:
            raise SystemExit(f"no experiments match --only {args.only}")

    print(f"### REFLEX v4 experiment suite -- profile: {args.profile} "
          f"({len(plan)} experiments) ###\n")
    failures = []
    t_start = time.time()
    for name, module, exp_args in plan:
        print(f"\n{'=' * 72}\n### [{name}] python -m experiments.run_{name} "
              f"{' '.join(exp_args)}\n{'=' * 72}")
        t0 = time.time()
        try:
            module.main(exp_args)
            print(f"### [{name}] OK ({time.time() - t0:.0f}s)")
        except Exception:
            failures.append(name)
            print(f"### [{name}] FAILED ({time.time() - t0:.0f}s)")
            traceback.print_exc()

    print(f"\n{'=' * 72}")
    print(f"### suite done in {(time.time() - t_start) / 60:.1f} min -- "
          f"{len(plan) - len(failures)}/{len(plan)} passed"
          + (f"; FAILED: {failures}" if failures else ""))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
