"""
build_datasets.py — Build all REFLEX processed datasets and master file.

Reads from data/raw/ and writes to data/processed/ and data/master/.

Usage:
    python src/build_datasets.py
"""

import pandas as pd
import numpy as np
import os

BASE   = os.path.join(os.path.dirname(__file__), "..")
RAW    = os.path.join(BASE, "data", "raw")
PROC   = os.path.join(BASE, "data", "processed")
MASTER = os.path.join(BASE, "data", "master")

for d in [PROC, MASTER]:
    os.makedirs(d, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def vix_regime(v):
    if v < 15: return "calm"
    if v < 20: return "normal"
    if v < 30: return "elevated"
    if v < 50: return "stress"
    return "crisis"


# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Load and enrich raw sources
# ─────────────────────────────────────────────────────────────────────────────

print("Loading raw sources...")

# A: VIX
vix = pd.read_csv(f"{RAW}/finance-vix__vix-daily.csv", parse_dates=["DATE"])
vix.columns = ["date", "vix_open", "vix_high", "vix_low", "vix_close"]
vix = vix.set_index("date").sort_index()
vix["vix_log_ret"]    = np.log(vix["vix_close"]).diff()
vix["vix_20d_std"]    = vix["vix_close"].rolling(20).std()
vix["vix_60d_mean"]   = vix["vix_close"].rolling(60).mean()
vix["sigma_proxy"]    = vix["vix_close"] / 100.0
vix["k_proxy"]        = vix["vix_close"] * 0.030
vix["regime"]         = vix["vix_close"].apply(vix_regime)
vix.reset_index().to_csv(f"{RAW}/reflex_A_vix_daily.csv", index=False)
print(f"  A: VIX daily {vix.index.min().date()} to {vix.index.max().date()} — {len(vix):,} rows")

# B: WTI Oil
oil = pd.read_csv(f"{RAW}/oil-prices__wti-daily.csv", parse_dates=["Date"])
oil = oil.rename(columns={"Date": "date", "Price": "wti_price"}).set_index("date").sort_index()
oil["oil_log_ret"]   = np.log(oil["wti_price"]).diff()
oil["oil_20d_vol"]   = oil["oil_log_ret"].rolling(20).std() * np.sqrt(252)
oil.reset_index().to_csv(f"{RAW}/reflex_B_wti_daily.csv", index=False)
print(f"  B: WTI Oil daily {oil.index.min().date()} to {oil.index.max().date()} — {len(oil):,} rows")

# C: 10y Treasury
gs10 = pd.read_csv(f"{RAW}/bond-yields-us-10y__monthly.csv", parse_dates=["Date"])
gs10 = gs10.rename(columns={"Date": "date", "Rate": "gs10_yield"}).set_index("date").sort_index()
gs10["yield_change"]    = gs10["gs10_yield"].diff()
gs10["yield_12m_vol"]   = gs10["gs10_yield"].rolling(12).std()
gs10["yield_12m_mean"]  = gs10["gs10_yield"].rolling(12).mean()
gs10["dv01_10y"]        = 9.5 / (1 + gs10["gs10_yield"] / 100)
gs10.reset_index().to_csv(f"{RAW}/reflex_C_treasury10y_monthly.csv", index=False)
print(f"  C: 10y Treasury monthly {gs10.index.min().date()} to {gs10.index.max().date()} — {len(gs10):,} rows")

# D: Shiller
shiller = pd.read_csv(f"{RAW}/s-and-p-500__data.csv", parse_dates=["Date"])
shiller = shiller.rename(columns={"Date": "date"}).set_index("date").sort_index()
shiller = shiller[(shiller["SP500"] > 0) & (shiller["Long Interest Rate"] > 0)]
shiller["sp500_log_ret"]   = np.log(shiller["SP500"]).diff()
shiller["sp500_sigma_12m"] = shiller["sp500_log_ret"].rolling(12).std() * np.sqrt(12)
shiller["earnings_yield"]  = shiller["Earnings"] / shiller["SP500"]
shiller = shiller.rename(columns={
    "Long Interest Rate": "gs10_rate",
    "PE10": "cape_ratio",
    "Consumer Price Index": "cpi",
})
cols = ["SP500", "Dividend", "Earnings", "cpi", "gs10_rate", "cape_ratio",
        "sp500_log_ret", "sp500_sigma_12m", "earnings_yield"]
shiller[cols].reset_index().to_csv(f"{RAW}/reflex_D_shiller_monthly.csv", index=False)
print(f"  D: Shiller monthly {shiller.index.min().date()} to {shiller.index.max().date()} — {len(shiller):,} rows")

# E: Gold + CPI
gold = pd.read_csv(f"{RAW}/gold-prices__monthly-processed.csv", parse_dates=["Date"])
gold = gold.rename(columns={"Date": "date", "Price": "gold_usd"}).set_index("date").sort_index()
gold["gold_log_ret"] = np.log(gold["gold_usd"]).diff()
cpi = pd.read_csv(f"{RAW}/cpi-us__cpiai.csv", parse_dates=["Date"])
cpi = cpi.rename(columns={"Date": "date", "Index": "cpi_index", "Inflation": "cpi_mom_pct"}).set_index("date").sort_index()
cpi = cpi[cpi["cpi_index"] > 0]
macro_m = gold.join(cpi, how="outer")
macro_m.reset_index().to_csv(f"{RAW}/reflex_E_gold_cpi_monthly.csv", index=False)
print(f"  E: Gold+CPI monthly {macro_m.index.min().date()} to {macro_m.index.max().date()} — {len(macro_m):,} rows")

# F: Dickerson bond factors
factors = pd.read_csv(f"{RAW}/dickerson_TRACE__all_factors_wrds.csv", parse_dates=["date"])
factors = factors.rename(columns={
    "MKTB": "bond_mkt_ret", "DRF": "duration_rf",
    "CRF": "credit_rf",     "LRF": "liquidity_rf",
})
factors = factors.set_index("date").sort_index()
factors["cum_lrf_12m"] = factors["liquidity_rf"].rolling(12).sum()
factors["cum_crf_12m"] = factors["credit_rf"].rolling(12).sum()
factors["bond_vol_6m"] = factors["bond_mkt_ret"].rolling(6).std() * np.sqrt(12)
factors.reset_index().to_csv(f"{RAW}/reflex_F_dickerson_bond_factors.csv", index=False)
print(f"  F: Dickerson factors monthly {factors.index.min().date()} to {factors.index.max().date()} — {len(factors)} rows")

# G: Bond returns (real CUSIPs)
bond_ret = pd.read_csv(f"{RAW}/Factor-Strategy-for-Corporate-Bond-__Monthly_Log_Returns.csv")
bond_ret["date"] = pd.to_datetime(bond_ret["month_year"])
bond_ret = bond_ret.rename(columns={"cusip_id": "cusip", "log_returns": "log_ret_monthly"})
bond_ret[["cusip", "date", "log_ret_monthly"]].to_csv(f"{RAW}/reflex_G_bond_returns_monthly.csv", index=False)

xs = bond_ret.groupby("date").agg(
    n_bonds=("cusip", "nunique"),
    ret_mean=("log_ret_monthly", "mean"),
    ret_sigma_xs=("log_ret_monthly", "std"),
    ret_p25=("log_ret_monthly", lambda x: x.quantile(0.25)),
    ret_p75=("log_ret_monthly", lambda x: x.quantile(0.75)),
).reset_index()
xs.to_csv(f"{RAW}/reflex_G2_bond_xsection_sigma.csv", index=False)
print(f"  G: Bond returns {bond_ret['date'].min().date()} to {bond_ret['date'].max().date()} — {len(bond_ret):,} rows / {bond_ret['cusip'].nunique()} bonds")


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Build processed/reflex_H — monthly merged calibration table
# ─────────────────────────────────────────────────────────────────────────────

print("\nBuilding monthly calibration table (H)...")

vix_m = (vix["vix_close"].resample("MS").agg(
    vix_mean="mean", vix_max="max", vix_std="std"
).reset_index())
vix_m["regime"] = vix_m["vix_mean"].apply(vix_regime)

gs10_m   = gs10[["gs10_yield"]].resample("MS").last().reset_index()
shiller_m = shiller[["sp500_sigma_12m", "cape_ratio"]].reset_index()
cpi_m    = cpi[["cpi_mom_pct"]].resample("MS").last().reset_index()
gold_m   = gold[["gold_usd"]].resample("MS").last().reset_index()
factors_m = factors.reset_index()[["date","liquidity_rf","credit_rf","bond_mkt_ret"]]
factors_m["date"] = pd.to_datetime(factors_m["date"]).dt.to_period("M").dt.to_timestamp()

merged = (vix_m
    .merge(gs10_m,      on="date", how="left")
    .merge(shiller_m,   on="date", how="left")
    .merge(cpi_m,       on="date", how="left")
    .merge(gold_m,      on="date", how="left")
    .merge(factors_m,   on="date", how="left")
    .merge(xs[["date","ret_sigma_xs","n_bonds"]], on="date", how="left")
)
merged["real_yield"]           = merged["gs10_yield"] - merged["cpi_mom_pct"].fillna(0)
merged["implied_spread_bps"]   = (7.0 * merged["vix_mean"] + 5).clip(20, 3000)
merged["sigma_IG_estimate"]    = merged["sp500_sigma_12m"] * 0.40
merged["sigma_HY_estimate"]    = merged["sp500_sigma_12m"] * 0.72
merged["k_IG_estimate"]        = merged["implied_spread_bps"] / 200.0
merged["k_HY_estimate"]        = merged["implied_spread_bps"] / 80.0
merged = merged[merged["date"] >= "1990-01-01"].dropna(subset=["vix_mean"])
merged.to_csv(f"{PROC}/reflex_H_merged_calibration.csv", index=False)
print(f"  H: Monthly calibration {merged['date'].min().date()} to {merged['date'].max().date()} — {len(merged)} rows")


# ─────────────────────────────────────────────────────────────────────────────
# Step 3: Build processed/reflex_I — regime-level calibration params
# ─────────────────────────────────────────────────────────────────────────────

print("Building calibration params table (I)...")

rows = []
for regime in ["calm", "normal", "elevated", "stress", "crisis"]:
    sub = merged[merged["regime"] == regime]
    if len(sub) < 3:
        continue
    rows.append({
        "regime":                    regime,
        "n_months":                  len(sub),
        "date_first":                str(sub["date"].min().date()),
        "date_last":                 str(sub["date"].max().date()),
        "vix_mean":                  round(sub["vix_mean"].mean(), 2),
        "gs10_yield_mean_pct":       round(sub["gs10_yield"].mean(), 3),
        "real_yield_mean_pct":       round(sub["real_yield"].mean(), 3),
        "sp500_sigma_12m_mean":      round(sub["sp500_sigma_12m"].mean(), 4),
        "implied_spread_bps_mean":   round(sub["implied_spread_bps"].mean(), 1),
        "implied_spread_bps_p25":    round(sub["implied_spread_bps"].quantile(0.25), 1),
        "implied_spread_bps_p75":    round(sub["implied_spread_bps"].quantile(0.75), 1),
        "bond_xsection_sigma_mean":  round(sub["ret_sigma_xs"].mean(), 5),
        "liquidity_rf_mean":         round(sub["liquidity_rf"].mean(), 5),
        "credit_rf_mean":            round(sub["credit_rf"].mean(), 5),
        # ── Simulator-ready parameters ──────────────────────────────────────
        "sim_sigma_IG":     round(sub["sp500_sigma_12m"].mean() * 0.40, 4),
        "sim_sigma_HY":     round(sub["sp500_sigma_12m"].mean() * 0.72, 4),
        "sim_k_IG_lo":      round(sub["implied_spread_bps"].quantile(0.25) / 200, 3),
        "sim_k_IG_hi":      round(sub["implied_spread_bps"].quantile(0.75) / 150, 3),
        "sim_k_HY_lo":      round(sub["implied_spread_bps"].quantile(0.25) / 100, 3),
        "sim_k_HY_hi":      round(sub["implied_spread_bps"].quantile(0.75) / 70, 3),
        "data_is_synthetic": False,
    })

cal_df = pd.DataFrame(rows)
cal_df.to_csv(f"{PROC}/reflex_I_calibration_params.csv", index=False)
print(f"  I: Calibration params — {len(cal_df)} regimes")
print(cal_df[["regime", "n_months", "vix_mean", "implied_spread_bps_mean",
               "sim_sigma_IG", "sim_k_IG_lo", "sim_k_IG_hi"]].to_string(index=False))


# ─────────────────────────────────────────────────────────────────────────────
# Step 4: Build master/REFLEX_MASTER_DATASET.csv
# ─────────────────────────────────────────────────────────────────────────────

print("\nBuilding REFLEX_MASTER_DATASET.csv...")

master = vix[["vix_close", "vix_20d_std", "vix_60d_mean",
               "sigma_proxy", "k_proxy", "regime"]].copy()
master = master.join(oil[["wti_price", "oil_20d_vol"]], how="left")

idx = master.index
master = master.join(gs10[["gs10_yield", "dv01_10y"]].reindex(idx, method="ffill"))
master = master.join(shiller[["SP500","sp500_sigma_12m","cape_ratio","earnings_yield"]].reindex(idx, method="ffill"))
master = master.join(macro_m[["gold_usd","cpi_index","cpi_mom_pct"]].reindex(idx, method="ffill"))
master = master.join(factors[["bond_mkt_ret","duration_rf","credit_rf","liquidity_rf",
                                "cum_lrf_12m","bond_vol_6m"]].reindex(idx, method="ffill"))
master = master.join(xs.set_index("date")[["n_bonds","ret_mean","ret_sigma_xs"]].reindex(idx, method="ffill"))

master["real_yield"]         = master["gs10_yield"] - master["cpi_mom_pct"].fillna(0)
master["implied_spread_bps"] = (7.0 * master["vix_close"] + 5).clip(20, 3000)
master["sigma_IG_estimate"]  = master["sp500_sigma_12m"] * 0.40
master["sigma_HY_estimate"]  = master["sp500_sigma_12m"] * 0.72
master["k_IG_estimate"]      = master["implied_spread_bps"] / 200.0
master["k_HY_estimate"]      = master["implied_spread_bps"] / 80.0
master["A_IG_estimate"]      = np.clip(150.0 / (master["implied_spread_bps"] / 10 + 1), 2, 25)
master["A_HY_estimate"]      = master["A_IG_estimate"] * 0.15
master["data_is_synthetic"]  = False
master["primary_source"]     = "CBOE VIX + Fed H.15 + Shiller Yale + EIA WTI + Dickerson TRACE"

master = master[master.index >= "1990-01-02"]
master.index.name = "date"
master.reset_index().to_csv(f"{MASTER}/REFLEX_MASTER_DATASET.csv", index=False)
print(f"  MASTER: {len(master):,} rows × {master.shape[1]} cols")
print(f"  Date range: {master.index.min().date()} to {master.index.max().date()}")

print("\n" + "="*60)
print("BUILD COMPLETE")
print("="*60)
for path, label in [
    (f"{RAW}/reflex_A_vix_daily.csv",                  "A VIX daily"),
    (f"{RAW}/reflex_B_wti_daily.csv",                  "B WTI daily"),
    (f"{RAW}/reflex_C_treasury10y_monthly.csv",         "C Treasury monthly"),
    (f"{RAW}/reflex_D_shiller_monthly.csv",             "D Shiller monthly"),
    (f"{RAW}/reflex_E_gold_cpi_monthly.csv",            "E Gold+CPI monthly"),
    (f"{RAW}/reflex_F_dickerson_bond_factors.csv",      "F Bond factors monthly"),
    (f"{RAW}/reflex_G_bond_returns_monthly.csv",        "G Bond returns monthly"),
    (f"{RAW}/reflex_G2_bond_xsection_sigma.csv",        "G2 XS sigma monthly"),
    (f"{PROC}/reflex_H_merged_calibration.csv",         "H Merged calibration"),
    (f"{PROC}/reflex_I_calibration_params.csv",         "I Calibration params"),
    (f"{MASTER}/REFLEX_MASTER_DATASET.csv",             "MASTER (daily, all sources)"),
]:
    df = pd.read_csv(path)
    print(f"  {label}: {len(df):,} rows × {df.shape[1]} cols")
