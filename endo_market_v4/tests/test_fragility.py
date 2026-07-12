"""Tests for the daily market-fragility index (closed forms on real data)."""

from __future__ import annotations

import numpy as np
import pytest

from reflex.analysis.fragility import compute_fragility, save_fragility


@pytest.fixture(scope="module")
def frag_ig():
    return compute_fragility(rating="IG")


def test_full_panel_computes(frag_ig):
    df = frag_ig.daily
    assert len(df) == 9218  # every trading day 1990-2026
    for col in ("gamma", "epsilon_at_h", "modulus_at_h", "eps_star", "fragility"):
        assert np.isfinite(df[col]).all(), f"{col} has non-finite values"
        assert (df[col] > 0).all(), f"{col} has non-positive values"


def test_crisis_headroom_collapses(frag_ig):
    """The a-priori boundary eps* must be far tighter in crisis than in calm --
    the curvature-collapse mechanism (crisis k=0 -> gamma at the anchor floor)."""
    by = frag_ig.by_regime
    assert by.loc["crisis", "eps_star"] < 0.5 * by.loc["calm", "eps_star"]
    assert by.loc["crisis", "fragility"] > by.loc["calm", "fragility"]


def test_headroom_monotone_calm_to_stress(frag_ig):
    """Regime medians of eps* shrink monotonically calm -> stress -> crisis."""
    by = frag_ig.by_regime["eps_star"]
    assert by["calm"] > by["normal"] > by["elevated"] > by["stress"] > by["crisis"]


def test_gfc_and_covid_spike(frag_ig):
    """Fragility in the two acute crisis windows the dataset documents (the
    Lehman quarter and the March-2020 bond-market freeze) must exceed the
    sample median x2."""
    df = frag_ig.daily.set_index("date")
    med = df["fragility"].median()
    gfc = df.loc["2008-10-01":"2008-12-31", "fragility"].median()
    covid = df.loc["2020-03-01":"2020-03-31", "fragility"].median()
    assert gfc > 2.0 * med, f"GFC fragility {gfc:.2f} vs median {med:.2f}"
    assert covid > 2.0 * med, f"COVID fragility {covid:.2f} vs median {med:.2f}"
    # the all-sample fragility peak must sit inside a crisis regime day
    peak_day = df["fragility"].idxmax()
    assert df.loc[peak_day, "regime"] == "crisis"


def test_defensive_widening_visible_in_modulus(frag_ig):
    """At observed (wide) crisis spreads the toxic slope has decayed: the
    modulus at the operating point does NOT explode in crisis -- the real-data
    counterpart of the documented saturation behaviour."""
    by = frag_ig.by_regime
    assert by.loc["crisis", "modulus_at_h"] <= by.loc["calm", "modulus_at_h"] * 5.0


def test_hy_panel_and_save(tmp_path):
    res = compute_fragility(rating="HY")
    assert len(res.daily) == 9218
    daily_path, regime_path = save_fragility(res, tmp_path)
    assert daily_path.exists() and regime_path.exists()


def test_rejects_unknown_rating():
    with pytest.raises(ValueError):
        compute_fragility(rating="JUNK")
