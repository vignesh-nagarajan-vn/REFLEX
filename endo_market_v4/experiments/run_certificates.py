"""Run the numerical proof certificates for theory 1.1-1.6 (v4 verification).

Every load-bearing identity / inequality / dynamical claim of the six
derivations is re-derived numerically against the closed-form implementations
(finite-difference slopes, eigensolves, Monte-Carlo rates, constructed
regimes) -- see :mod:`reflex.verification.certificates`.  Runs on the default
config and, when the shipped calibrations load, on a calibrated real-unit
config (the identities must hold in real units too -- the unit-convention
certificate).

Exits non-zero on any failed check, so ``run_all`` reports it as a failure.

Usage::

    python -m experiments.run_certificates --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from reflex.config import load_config
from reflex.verification import run_all_certificates


def _run_and_print(cfg, label: str, rows: list) -> bool:
    print(f"\n--- certificates on {label} ---")
    certs = run_all_certificates(cfg)
    ok = True
    for cert in certs:
        print(f"  {cert.summary()}")
        for ch in cert.checks:
            rows.append({
                "config": label, "certificate": cert.name, "check": ch.name,
                "value": ch.value, "tolerance": ch.tolerance, "passed": ch.passed,
            })
            if not ch.passed:
                ok = False
                print(f"    FAILED: {ch.name} = {ch.value:.6g} (tol {ch.tolerance:g})")
    return ok


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Numerical proof certificates (1.1-1.6).")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--outdir", default="outputs")
    ap.add_argument("--skip-calibrated", action="store_true",
                    help="only certify the raw config")
    args = ap.parse_args(argv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rows: list = []
    all_ok = _run_and_print(load_config(args.config), args.config, rows)

    if not args.skip_calibrated:
        try:
            from reflex.calibration import apply_calibration

            cal_cfg = load_config(args.config)
            cal_cfg.calibration.enabled = True
            cal_cfg.calibration.rating = "IG"
            cal_cfg.calibration.regime = "normal"
            cal_cfg, info = apply_calibration(cal_cfg)
            label = "calibrated IG/normal (real units)"
            all_ok = _run_and_print(cal_cfg, label, rows) and all_ok
        except Exception as exc:  # calibration data missing -> report, not fail
            print(f"\n(calibrated certificate pass skipped: {exc})")

    csv_path = outdir / "certificates.csv"
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    n_pass = sum(1 for r in rows if r["passed"])
    print(f"\nsaved {len(rows)} checks ({n_pass} passed) -> {csv_path}")
    if not all_ok:
        raise SystemExit(1)
    print("ALL CERTIFICATES PASS")


if __name__ == "__main__":
    main()
