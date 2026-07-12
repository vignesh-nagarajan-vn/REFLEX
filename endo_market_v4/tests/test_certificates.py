"""Tests for the v4 verification layer (numerical proof certificates)."""

from __future__ import annotations

import numpy as np
import pytest

from reflex.config import load_config
from reflex.verification import (
    certify_boundary,
    certify_factor_scaling,
    certify_lazy_deploy,
    certify_multi_dealer,
    certify_perfgd,
    certify_robust,
    run_all_certificates,
)
from reflex.verification.certificates import Certificate

CONFIG = "configs/default.yaml"


@pytest.fixture(scope="module")
def cfg():
    return load_config(CONFIG)


def _assert_pass(cert: Certificate):
    failed = [c for c in cert.checks if not c.passed]
    assert cert.passed, (
        f"{cert.name}: " + "; ".join(f"{c.name}={c.value:.4g} (tol {c.tolerance:g})"
                                     for c in failed)
    )


def test_certify_boundary_default(cfg):
    _assert_pass(certify_boundary(cfg))


def test_certify_perfgd_default(cfg):
    _assert_pass(certify_perfgd(cfg))


def test_certify_perfgd_dynamics_default(cfg):
    from reflex.verification import certify_perfgd_dynamics

    _assert_pass(certify_perfgd_dynamics(cfg))


def test_dynamics_certificate_excluded_on_calibrated(cfg):
    """The raw-unit demo must not be imposed on calibrated (real-unit)
    configs: run_all_certificates auto-excludes it there."""
    import copy

    cal = copy.deepcopy(cfg)
    cal.calibration.enabled = True
    names_cal = [c.name for c in run_all_certificates(cal)]
    assert not any("dynamics" in n for n in names_cal)
    names_raw = [c.name for c in run_all_certificates(cfg)]
    assert any("dynamics" in n for n in names_raw)


def test_certify_multi_dealer_default(cfg):
    _assert_pass(certify_multi_dealer(cfg))
    # a second coupling point
    _assert_pass(certify_multi_dealer(cfg, n_dealers=5, kappa=0.25))


def test_certify_robust():
    _assert_pass(certify_robust(seed=0))


def test_certify_factor_scaling_default(cfg):
    _assert_pass(certify_factor_scaling(cfg))


def test_certify_lazy_deploy():
    _assert_pass(certify_lazy_deploy())


def test_all_certificates_perturbed_config(cfg):
    """The identities are config-independent: perturb the microstructure and
    every certificate must still pass (the closed forms move, the identities
    do not)."""
    import copy

    alt = copy.deepcopy(cfg)
    alt.clients.alpha = 0.8
    alt.clients.toxicity_feedback = 2.0
    alt.clients.info_spread_decay = 1.1
    alt.reward.quote_anchor_weight = 0.4
    alt.bonds.n_bonds = 6
    for cert in run_all_certificates(alt):
        _assert_pass(cert)


def test_certificates_can_fail():
    """A certifier that cannot fail verifies nothing: feed the robust
    machinery an inconsistent constructed case and check the flag trips."""
    from reflex.theory.robust import robust_certificate

    # mean deep in the stable region but claimed unstable -> the trichotomy
    # checks in certify_robust would catch a broken verdict; here we check
    # the check-construction itself flags a wrong verdict.
    cert = robust_certificate(0.5, 0.1)
    assert cert.verdict == "stable"
    from reflex.verification.certificates import _flag

    bad = _flag("deliberately_wrong", cert.verdict == "unstable")
    assert not bad.passed


def test_certificate_summary_format(cfg):
    cert = certify_lazy_deploy()
    s = cert.summary()
    assert "1.6" in s and ("PASS" in s or "FAIL" in s)
    assert all(np.isfinite(c.value) for c in cert.checks)
