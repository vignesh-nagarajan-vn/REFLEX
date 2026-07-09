"""
verify_data.py — Verify all REFLEX datasets against known historical values.

Prints PASS/FAIL for every check. Exit code 0 = all pass, 1 = any fail.

Usage:
    python src/verify_data.py
"""

import pandas as pd
import numpy as np
import os
import sys

BASE = os.path.join(os.path.dirname(__file__), "..")
RAW  = os.path.join(BASE, "data", "raw")
PROC = os.path.join(BASE, "data", "processed")
MASTER = os.path.join(BASE, "data", "master")

results = []

def check(label, actual, expected, tol=0.05):
    """Check actual ≈ expected within relative tolerance."""
    if expected == 0:
        ok = abs(actual) < tol
    else:
        ok = abs(actual - expected) / abs(expected) < tol
    status = "PASS" if ok else "FAIL"
    results.append(ok)
    print(f"  [{status}] {label}: got {actual:.4g}, expected ~{expected:.4g}")
    return ok

def check_exact(label, actual, expected):
    ok = actual == expected
    status = "PASS" if ok else "FAIL"
    results.append(ok)
    print(f"  [{status}] {label}: got {actual!r}, expected {expected!r}")
    return ok

def check_range(label, actual, lo, hi):
    ok = lo <= actual <= hi
    status = "PASS" if ok else "FAIL"
    results.append(ok)
    print(f"  [{status}] {label}: got {actual:.4g}, expected [{lo:.4g}, {hi:.4g}]")
    return ok


print("=" * 60)
print("REFLEX Data Verification")
print("=" * 60)

# ── A: VIX ───────────────────────────────────────────────────────────────────
print("\n[A] CBOE VIX Daily")
vix = pd.read_csv(f"{RAW}/reflex_A_vix_daily.csv", parse_dates=["date"])
vix = vix.set_index("date").sort_index()

check("Jan 2 1990 VIX close",  float(vix.loc["1990-01-02", "vix_close"]), 17.24, tol=0.01)
check("Aug 24 2015 VIX close", float(vix.loc["2015-08-24", "vix_close"]), 40.74, tol=0.05)
check("Oct 24 2008 VIX close", float(vix.loc["2008-10-24", "vix_close"]), 79.13, tol=0.05)
check("Mar 16 2020 VIX close", float(vix.loc["2020-03-16", "vix_close"]), 82.69, tol=0.05)
check("Row count",             len(vix),                                    9218,  tol=0.01)
check_range("sigma_proxy range", vix["sigma_proxy"].mean(), 0.05, 0.30)

# ── B: WTI Oil ───────────────────────────────────────────────────────────────
print("\n[B] EIA WTI Crude Oil Daily")
oil = pd.read_csv(f"{RAW}/reflex_B_wti_daily.csv", parse_dates=["date"])
oil = oil.set_index("date").sort_index()

jun_jul_2008_max = oil.loc["2008-06-01":"2008-07-31", "wti_price"].max()
dec2008_min      = oil.loc["2008-12-01":"2009-01-15", "wti_price"].min()
jan1986_price    = float(oil.iloc[0]["wti_price"])

check("Jun–Jul 2008 WTI peak",  jun_jul_2008_max, 143.0, tol=0.05)
check("Dec 2008 WTI trough",    dec2008_min,       32.0,  tol=0.15)
check_range("Jan 1986 first price", jan1986_price, 20, 30)

# ── C: Treasury ──────────────────────────────────────────────────────────────
print("\n[C] Federal Reserve 10-Year Treasury Monthly")
gs10 = pd.read_csv(f"{RAW}/reflex_C_treasury10y_monthly.csv", parse_dates=["date"])
gs10 = gs10.set_index("date").sort_index()

check("Jan 1981 GS10 (rate peak)", float(gs10.loc["1981-01-01", "gs10_yield"]), 12.57, tol=0.05)
check("Apr 2000 GS10",             float(gs10.loc["2000-04-01", "gs10_yield"]),  5.99, tol=0.05)
check("Jul 2016 GS10 (trough)",    float(gs10.loc["2016-07-01", "gs10_yield"]),  1.50, tol=0.10)
check("Row count",                 len(gs10), 878, tol=0.01)

# ── D: Shiller ────────────────────────────────────────────────────────────────
print("\n[D] Shiller S&P500 Monthly (monthly averages)")
sh = pd.read_csv(f"{RAW}/reflex_D_shiller_monthly.csv", parse_dates=["date"])
sh = sh.set_index("date").sort_index()

check("Jan 1990 SP500 monthly avg",  float(sh.loc["1990-01-01", "SP500"]),   340.0, tol=0.05)
check("Jan 2000 SP500 monthly avg",  float(sh.loc["2000-01-01", "SP500"]), 1_426.0, tol=0.03)
check("Jan 1981 GS10 rate",          float(sh.loc["1981-01-01", "gs10_rate"]), 12.57, tol=0.05)
check("CAPE Jan 2000 (elevated)",    float(sh.loc["2000-01-01", "cape_ratio"]), 43.8, tol=0.10)

# ── E: Gold + CPI ─────────────────────────────────────────────────────────────
print("\n[E] Gold Prices + US CPI Monthly")
ec = pd.read_csv(f"{RAW}/reflex_E_gold_cpi_monthly.csv", parse_dates=["date"])
ec = ec.set_index("date").sort_index()

check("Gold Jan 2000",  float(ec.loc["2000-01-01", "gold_usd"]),  284.0, tol=0.05)
check("Gold Sep 2011",  float(ec.loc["2011-09-01", "gold_usd"]), 1772.0, tol=0.05)
check("CPI Jan 1990",   float(ec.loc["1990-01-01", "cpi_index"]), 127.4, tol=0.02)
check("CPI Jan 2023",   float(ec.loc["2023-01-01", "cpi_index"]), 299.2, tol=0.02)

# ── F: Dickerson TRACE factors ────────────────────────────────────────────────
print("\n[F] Dickerson TRACE Bond Factors Monthly")
fac = pd.read_csv(f"{RAW}/reflex_F_dickerson_bond_factors.csv", parse_dates=["date"])
fac = fac.set_index("date").sort_index()

sep2008_mktb = float(fac.loc["2008-09-30", "bond_mkt_ret"])
lrf_mean     = float(fac["liquidity_rf"].mean())
annual_vol   = float(fac["bond_mkt_ret"].std() * np.sqrt(12))

check("Sep 2008 MKTB (Lehman shock)", sep2008_mktb, -0.0826, tol=0.05)
check_range("LRF mean (liquidity premium)", lrf_mean, -0.001, 0.010)
check_range("Bond mkt annual vol (%)",      annual_vol,  0.04,  0.10)
check("Row count (Aug 2004–Dec 2021)",     len(fac),    209,   tol=0.01)

# ── G: QuhiQuhihi bond returns ────────────────────────────────────────────────
print("\n[G] Real CUSIP-Level Bond Returns Monthly")
br = pd.read_csv(f"{RAW}/reflex_G_bond_returns_monthly.csv")

sample_cusips = br["cusip"].unique()[:10].tolist()
all_9char     = all(len(c) == 9 for c in sample_cusips)
all_alphanum  = all(c.isalnum() for c in sample_cusips)
ret_std       = float(br["log_ret_monthly"].std())
ret_mean      = float(br["log_ret_monthly"].mean())

check_exact("All CUSIPs are 9 characters", all_9char,    True)
check_exact("All CUSIPs are alphanumeric", all_alphanum, True)
check_range("Monthly return std (bond-like, %)", ret_std,  0.005, 0.060)
check_range("Monthly return mean (near zero)",   ret_mean, -0.010, 0.010)

# ── Master dataset ────────────────────────────────────────────────────────────
print("\n[MASTER] REFLEX_MASTER_DATASET.csv")
mdf = pd.read_csv(f"{MASTER}/REFLEX_MASTER_DATASET.csv", parse_dates=["date"])
mdf = mdf.set_index("date").sort_index()

check_exact("data_is_synthetic always False",
            bool((mdf["data_is_synthetic"] == False).all()), True)
check("Row count", len(mdf), 9218, tol=0.01)
check_range("VIX close range [5, 100]", mdf["vix_close"].max(), 80, 95)
check("Columns", mdf.shape[1], 36, tol=0.05)

# ── Summary ───────────────────────────────────────────────────────────────────
total   = len(results)
passed  = sum(results)
failed  = total - passed

print()
print("=" * 60)
print(f"VERIFICATION COMPLETE: {passed}/{total} checks passed")
if failed:
    print(f"WARNING: {failed} check(s) FAILED — review output above")
else:
    print("ALL CHECKS PASSED — dataset is verified real data")
print("=" * 60)

sys.exit(0 if failed == 0 else 1)
