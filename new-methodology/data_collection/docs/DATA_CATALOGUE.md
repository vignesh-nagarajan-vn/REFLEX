# Data Catalogue — REFLEX Dataset Collection

Full column-level documentation for all files in `data/raw/`, `data/processed/`, and `data/master/`.

---

## data/raw/reflex_A_vix_daily.csv

**Source:** CBOE Volatility Index, mirrored at https://github.com/datasets/finance-vix  
**Timespan:** 1990-01-02 to 2026-06-30 | **Frequency:** Daily | **Rows:** 9,218  
**Synthetic:** No

| Column | Type | Description |
|--------|------|-------------|
| `date` | date | Trading date |
| `vix_open` | float | VIX opening level |
| `vix_high` | float | VIX intraday high |
| `vix_low` | float | VIX intraday low |
| `vix_close` | float | VIX closing level |
| `vix_log_ret` | float | Log return of VIX close |
| `vix_20d_std` | float | 20-day rolling standard deviation of vix_close |
| `vix_60d_mean` | float | 60-day rolling mean of vix_close |
| `sigma_proxy` | float | vix_close / 100 — annualised implied vol in decimal form |
| `k_proxy` | float | vix_close × 0.03 — price impact coefficient proxy |
| `regime` | string | calm / normal / elevated / stress / crisis (VIX thresholds: 15/20/30/50) |

---

## data/raw/reflex_B_wti_daily.csv

**Source:** US EIA, mirrored at https://github.com/datasets/oil-prices  
**Timespan:** 1986-01-02 to 2026-06-29 | **Frequency:** Daily | **Rows:** 10,191  
**Synthetic:** No

| Column | Type | Description |
|--------|------|-------------|
| `date` | date | Trading date |
| `wti_price` | float | WTI crude oil spot price (USD/barrel) |
| `oil_log_ret` | float | Daily log return of WTI price |
| `oil_20d_vol` | float | 20-day rolling vol, annualised (×√252) |

---

## data/raw/reflex_C_treasury10y_monthly.csv

**Source:** Federal Reserve H.15, mirrored at https://github.com/datasets/bond-yields-us-10y  
**FRED series:** GS10  
**Timespan:** 1953-04-01 to 2026-05-01 | **Frequency:** Monthly | **Rows:** 878  
**Synthetic:** No

| Column | Type | Description |
|--------|------|-------------|
| `date` | date | First of month |
| `gs10_yield` | float | 10-year Treasury yield, percent |
| `yield_change` | float | Month-on-month change in yield |
| `yield_12m_vol` | float | 12-month rolling std of gs10_yield |
| `yield_12m_mean` | float | 12-month rolling mean of gs10_yield |
| `dv01_10y` | float | Approximate DV01 per $100 par (9.5 / (1 + yield/100)) |

---

## data/raw/reflex_D_shiller_monthly.csv

**Source:** Robert Shiller, Yale University (http://www.econ.yale.edu/~shiller/data.htm), mirrored at https://github.com/datasets/s-and-p-500  
**Timespan:** 1871-01-01 to 2023-09-01 | **Frequency:** Monthly | **Rows:** 1,833  
**Synthetic:** No — all values are monthly averages, standard Shiller methodology

| Column | Type | Description |
|--------|------|-------------|
| `date` | date | First of month |
| `SP500` | float | S&P 500 monthly average level |
| `Dividend` | float | Dividend per share |
| `Earnings` | float | Earnings per share |
| `cpi` | float | Consumer Price Index (Shiller's series) |
| `gs10_rate` | float | 10-year Treasury rate (Shiller's GS10 copy) |
| `cape_ratio` | float | CAPE / Shiller PE10 ratio |
| `sp500_log_ret` | float | Monthly log return of SP500 |
| `sp500_sigma_12m` | float | 12-month rolling annualised vol (×√12) |
| `earnings_yield` | float | Earnings / SP500 (E/P ratio) |

---

## data/raw/reflex_E_gold_cpi_monthly.csv

**Sources:**  
- Gold: World Gold Council, mirrored at https://github.com/datasets/gold-prices  
- CPI: US Bureau of Labor Statistics (public domain), mirrored at https://github.com/datasets/cpi-us  
**Timespan:** 1833-01-01 to 2026-05-01 | **Frequency:** Monthly | **Rows:** 2,321  
**Synthetic:** No

| Column | Type | Description |
|--------|------|-------------|
| `date` | date | First of month |
| `gold_usd` | float | Gold spot price, USD/troy oz |
| `gold_log_ret` | float | Monthly log return of gold price |
| `cpi_index` | float | CPI-U index level (1982-84=100) |
| `cpi_mom_pct` | float | Month-on-month CPI inflation, percent |

---

## data/raw/reflex_F_dickerson_bond_factors.csv

**Source:** Dickerson, A.M., Mueller, P. & Robotti, C. (2023). *Priced Risk in Corporate Bonds.* Journal of Financial Economics. Replication code: https://github.com/Alexander-M-Dickerson/TRACE-corporate-bond-processing  
**Timespan:** 2004-08-31 to 2021-12-31 | **Frequency:** Monthly | **Rows:** 209  
**Synthetic:** No — derived from actual FINRA TRACE trade prints by the paper's authors

| Column | Type | Description |
|--------|------|-------------|
| `date` | date | End of month |
| `bond_mkt_ret` | float | Corporate bond market return (MKTB factor) |
| `duration_rf` | float | Duration risk factor return |
| `credit_rf` | float | Credit risk factor return |
| `liquidity_rf` | float | **Liquidity risk factor** — primary ε proxy for REFLEX |
| `cum_lrf_12m` | float | 12-month cumulative liquidity risk factor |
| `cum_crf_12m` | float | 12-month cumulative credit risk factor |
| `bond_vol_6m` | float | 6-month rolling annualised bond market vol |

**REFLEX use:** `liquidity_rf` captures the return premium for bearing illiquidity in OTC corporate bonds. Its magnitude and sign across market regimes empirically anchors the ε (adversarial intensity) parameter.

---

## data/raw/reflex_G_bond_returns_monthly.csv

**Source:** QuhiQuhihi (2023). Factor Strategy for Corporate Bonds. https://github.com/QuhiQuhihi/Factor-Strategy-for-Corporate-Bond-  
**Timespan:** 2014-12-01 to 2023-09-01 | **Frequency:** Monthly | **Rows:** 12,123  
**Synthetic:** No — CUSIPs are real 9-character identifiers for actual US corporate bonds

| Column | Type | Description |
|--------|------|-------------|
| `cusip` | string | 9-character CUSIP identifier (real bond, e.g. 00138GAB5 = Agilent Technologies) |
| `log_ret_monthly` | float | Monthly log return of the bond |
| `date` | date | Month (first of month) |

---

## data/raw/reflex_G2_bond_xsection_sigma.csv

**Source:** Derived from reflex_G above  
**Timespan:** 2014-12-01 to 2023-09-01 | **Frequency:** Monthly | **Rows:** 106  
**Synthetic:** No

| Column | Type | Description |
|--------|------|-------------|
| `date` | date | Month |
| `n_bonds` | int | Number of bonds in cross-section |
| `ret_mean` | float | Cross-sectional mean log return |
| `ret_sigma_xs` | float | Cross-sectional std of log returns — proxy for D(φ) spread |
| `ret_p25` | float | 25th percentile of cross-sectional returns |
| `ret_p75` | float | 75th percentile of cross-sectional returns |

---

## data/processed/reflex_H_merged_calibration.csv

**Timespan:** 1990-01-01 to 2026-06-01 | **Frequency:** Monthly | **Rows:** 438  
**Synthetic:** No — all columns derive from the real raw sources above

Monthly panel joining: VIX monthly mean/max/std + GS10 yield + Shiller SP500 sigma + Dickerson bond factors + QuhiQuhihi cross-sectional sigma + BLS CPI inflation. Includes REFLEX-derived implied_spread_proxy_bps and regime label.

---

## data/processed/reflex_I_calibration_params.csv

**Timespan:** 1990–2026 (regime summary) | **Frequency:** Per-regime | **Rows:** 5  
**Synthetic:** No — aggregated from real data in reflex_H

One row per market regime (calm / normal / elevated / stress / crisis). Contains real-data-grounded parameter estimates for σ_IG, σ_HY, k_IG, k_HY that replace the illustrative flat ranges in the synthetic simulator version.

---

## data/master/REFLEX_MASTER_DATASET.csv

**Timespan:** 1990-01-02 to 2026-06-30 | **Frequency:** Daily | **Rows:** 9,218 | **Columns:** 36  
**Synthetic:** No

Single joined daily table. Monthly series (GS10, Shiller, gold, CPI, bond factors) are forward-filled to trading days. `data_is_synthetic = False` on every row. See README.md Column Dictionary for full column definitions.
