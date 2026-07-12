"""Tests for the real-data calibration layer (loader + mapping + regimes)."""

from __future__ import annotations

import math

import pytest

from reflex.calibration import (
    RATINGS,
    REGIMES,
    calibrated_config,
    classify_vix,
    load_intensity_params,
    load_master,
    load_regime_summary,
    regime_microstructure,
)
from reflex.config import load_config
from reflex.theory.analytic_boundary import analytic_boundary

CONFIG = "configs/default.yaml"


@pytest.fixture(scope="module")
def base_cfg():
    return load_config(CONFIG)


# ------------------------------ loader ------------------------------------- #
def test_intensity_table_complete():
    df = load_intensity_params()
    assert len(df) == len(RATINGS) * len(REGIMES)  # 2 x 5 = 10 cells
    for rating in RATINGS:
        for regime in REGIMES:
            assert (rating, regime) in df.index


def test_regime_monotonicity_of_fits():
    """Arrivals thin and spreads widen monotonically calm -> crisis (both buckets)."""
    for rating in RATINGS:
        cells = [regime_microstructure(rating, r) for r in REGIMES]
        A = [c.A for c in cells]
        h = [c.h_mean_decimal for c in cells]
        sigma = [c.sigma_annual for c in cells]
        assert A == sorted(A, reverse=True), f"{rating}: A not decreasing calm->crisis"
        assert h == sorted(h), f"{rating}: h not widening calm->crisis"
        assert sigma == sorted(sigma), f"{rating}: sigma not increasing calm->crisis"


def test_crisis_fit_flagged_degenerate():
    crisis = regime_microstructure("IG", "crisis")
    assert crisis.degenerate  # k pinned to 0 (n=74 days) -- documented
    calm = regime_microstructure("IG", "calm")
    assert not calm.degenerate


def test_loader_rejects_unknown_cells():
    with pytest.raises(ValueError):
        regime_microstructure("AAA", "calm")
    with pytest.raises(ValueError):
        regime_microstructure("IG", "apocalypse")


def test_master_panel_shape():
    df = load_master()
    assert len(df) == 9218
    assert not df["data_is_synthetic"].any()  # all rows real, per dataset contract
    assert set(df["regime"].unique()) <= set(REGIMES)


def test_regime_summary_consistent_with_vix_cutoffs():
    summary = load_regime_summary()
    # per-regime mean VIX must fall inside its own classification band
    for regime in REGIMES:
        vix_mean = float(summary.loc[regime, "vix_mean"])
        assert classify_vix(vix_mean) == regime


# ------------------------------ mapping ------------------------------------ #
def test_calibrated_config_units(base_cfg):
    cfg, info = calibrated_config(base_cfg, rating="IG", regime="normal")
    micro = info.micro
    # unit conversions: h x100, k /100, vol annual -> per-step per-100-par
    assert cfg.policy.init_half_spread == pytest.approx(micro.h_mean_decimal * 100)
    assert cfg.clients.demand_elasticity == pytest.approx(micro.k_decay / 100)
    assert cfg.clients.base_arrival_rate == pytest.approx(micro.A)
    assert cfg.simulator.fundamental_vol == pytest.approx(
        micro.sigma_annual / math.sqrt(252) * 100
    )
    # k*h product is unit-invariant
    assert cfg.clients.demand_elasticity * cfg.policy.init_half_spread == pytest.approx(
        micro.k_decay * micro.h_mean_decimal
    )
    # calibration section is stamped
    assert cfg.calibration.enabled and cfg.calibration.regime == "normal"


def test_calibrated_fixed_point_lands_near_observed_spread(base_cfg):
    """The anchor-stiffness rule must pin h* within ~30% of the data's h."""
    for regime in ("calm", "normal", "elevated", "stress"):
        cfg, info = calibrated_config(base_cfg, rating="IG", regime=regime)
        ab = analytic_boundary(cfg)
        assert 0.7 * info.h_100 <= ab.h_star <= 1.3 * info.h_100, (
            f"{regime}: h*={ab.h_star:.3f} vs observed {info.h_100:.3f}"
        )
        assert ab.gamma > 0 and math.isfinite(ab.modulus)
        assert ab.boundary_epsilon > 0


def test_calibrated_boundary_tightens_toward_crisis(base_cfg):
    """Headline regime ordering: stability headroom eps* shrinks calm -> stress.

    (crisis is excluded from the strict ordering: its k=0 fit is degenerate and
    the boundary there is anchor-floor-dominated.)
    """
    headroom = {}
    for regime in ("calm", "normal", "elevated", "stress"):
        cfg, _ = calibrated_config(base_cfg, rating="IG", regime=regime)
        headroom[regime] = analytic_boundary(cfg).boundary_epsilon
    assert headroom["calm"] > headroom["normal"] > headroom["elevated"] > headroom["stress"]


def test_apply_calibration_noop_when_disabled(base_cfg):
    from reflex.calibration import apply_calibration

    cfg, info = apply_calibration(base_cfg)
    assert info is None and cfg is base_cfg
