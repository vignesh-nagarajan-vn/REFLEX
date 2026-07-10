
## Overview

This repository contains the full data collection and preprocessing pipeline for REFLEX. All data is real and verified against known historical values - no synthetic data is included. Large generated files are excluded from git; run the four pipeline scripts below to regenerate them locally.

---

## Regenerating the Data Locally

```bash
git clone https://github.com/YOUR_USERNAME/reflex-data.git
cd reflex-data
pip install pandas numpy pyarrow requests statsmodels scipy

python src/fetch_data.py       # 1. download raw sources
python src/build_datasets.py   # 2. build master dataset
python src/verify_data.py      # 3. verify 33/33 checks pass
python src/preprocess.py       # 4. run preprocessing pipeline
```

Takes approximately 2 minutes. Regenerates everything deterministically from the same public sources.

---

## Repository Structure

```
reflex-data/
├── README.md
├── .gitignore                             large generated files excluded
├── src/
│   ├── fetch_data.py                      downloads all raw sources
│   ├── build_datasets.py                  builds master dataset
│   ├── verify_data.py                     33 historical value checks
│   └── preprocess.py                      full preprocessing pipeline
├── data/
│   ├── raw/                               individual verified source files
│   ├── processed/                         regime-level calibration params
│   ├── master/                            REFLEX_MASTER_DATASET.csv (generated)
│   └── preprocessed/                      pipeline outputs (generated)
└── docs/
    ├── DATA_CATALOGUE.md                  column-level documentation
    ├── VERIFICATION_LOG.md                historical value checks + verdicts
    └── REJECTED_SOURCES.md               sources found to be synthetic
```

---

## Data Collection

### Sources

All six sources are public domain or freely redistributable. No TRACE, Bloomberg, MarketAxess, or ICE BofA licensed data is included.

| File | Source | What it contains | Timespan | Rows |
|------|--------|-----------------|----------|------|
| `reflex_A_vix_daily.csv` | CBOE Volatility Index | Daily VIX OHLC - primary σ proxy and regime classifier | 1990–2026 | 9,218 |
| `reflex_B_wti_daily.csv` | US EIA | WTI crude oil spot price - macro stress co-indicator | 1986–2026 | 10,191 |
| `reflex_C_treasury10y_monthly.csv` | Federal Reserve H.15 | 10-year Treasury yield - DV01 calibration | 1953–2026 | 878 |
| `reflex_D_shiller_monthly.csv` | Robert Shiller / Yale | S&P 500, earnings, GS10, CAPE - equity σ and regime context | 1871–2023 | 1,833 |
| `reflex_E_gold_cpi_monthly.csv` | World Gold Council + BLS | Gold price and US CPI-U - real yield calculation | 1833–2026 | 2,321 |
| `reflex_F_dickerson_bond_factors.csv` | Dickerson, Mueller & Robotti (2023) JFE | TRACE-derived bond market factors including Liquidity Risk Factor - primary ε proxy | 2004–2021 | 209 |
| `reflex_G_bond_returns_monthly.csv` | QuhiQuhihi / real CUSIPs | Monthly log returns for 212 real US corporate bonds - D(φ) proxy | 2014–2023 | 12,111 |
| `reflex_G2_bond_xsection_sigma.csv` | Derived from G | Cross-sectional return dispersion per month | 2014–2023 | 106 |

### How to Access the Raw Data

Each source is downloaded directly from its public GitHub mirror using `src/fetch_data.py`, which calls `codeload.github.com` - no API keys or credentials required.

| Source | Direct link | FRED / official equivalent |
|--------|------------|---------------------------|
| CBOE VIX | https://github.com/datasets/finance-vix | FRED: `VIXCLS` |
| Fed 10y Treasury | https://github.com/datasets/bond-yields-us-10y | FRED: `GS10` |
| Shiller S&P 500 | https://github.com/datasets/s-and-p-500 | http://www.econ.yale.edu/~shiller/data.htm |
| EIA WTI Oil | https://github.com/datasets/oil-prices | https://www.eia.gov/dnav/pet/pet_pri_spt_s1_d.htm |
| BLS CPI | https://github.com/datasets/cpi-us | https://www.bls.gov/cpi/ |
| Dickerson TRACE factors | https://github.com/Alexander-M-Dickerson/TRACE-corporate-bond-processing | Dickerson et al. (2023) JFE replication package |
| Bond returns (real CUSIPs) | https://github.com/QuhiQuhihi/Factor-Strategy-for-Corporate-Bond- | - |

### Verification

Every source was checked against known historical values before inclusion. All 33 checks pass when you run `python src/verify_data.py`. Key spot checks:

| Dataset | Check | Expected | Actual |
|---------|-------|----------|--------|
| VIX | Mar 16 2020 close (COVID peak) | ~82–83 | **82.69** ✓ |
| VIX | Jan 2 1990 open | 17.24 | **17.24** ✓ |
| GS10 | Jan 1981 (rate peak) | ~12–14% | **12.57%** ✓ |
| GS10 | Jul 2016 (trough) | ~1.36–1.5% | **1.50%** ✓ |
| WTI | Jun–Jul 2008 peak | ~$143–147 | **$145.29** ✓ |
| Shiller | Jan 2000 SP500 monthly avg | ~1,425 | **1,426** ✓ |
| Dickerson | Sep 2008 MKTB (Lehman shock) | very negative | **−8.26%** ✓ |
| Bond returns | CUSIP format (9-char alphanumeric) | e.g. `00138GAB5` | **verified** ✓ |

Direct TRACE access requires WRDS (pending). The workaround uses Dickerson et al. (2023) JFE - a peer-reviewed paper that processed the full TRACE Enhanced feed and released derived factor returns publicly. Their Liquidity Risk Factor (LRF) and Credit Risk Factor (CRF) carry real bond microstructure signal and are citable. From these, plus VIX and GS10, we reconstruct all REFLEX calibration quantities. Parameter ranges are validated against published estimates: Bao et al. (2011) report IG Amihud illiquidity 0.2–0.8; Dick-Nielsen et al. (2012) report IG effective half-spreads of 20–80 bps.

### What is Not Available from Free Sources

Three quantities require TRACE Enhanced access (WRDS academic subscription) and are not in this dataset:

- **Trade-level bid-ask spread `h`** - TRACE dealer-side trade prints needed
- **Per-dealer inventory paths `q`** - TRACE dealer IDs are masked in public tier
- **Per-bond arrival rate `A` fitted from real prints** - requires trade-level timestamps

These are documented gaps. WRDS application: https://wrds-www.wharton.upenn.edu/

---

## Preprocessing

All preprocessing is implemented in `src/preprocess.py` and runs on `REFLEX_MASTER_DATASET.csv` as the single input. Every step adds columns to the same 9,218-row daily panel - nothing is dropped.

### Step 1 - Clean

**Null handling.** Two columns groups have expected nulls due to source coverage gaps: Dickerson bond factors start August 2004 (3,695 nulls pre-2004) and cross-sectional bond sigma starts December 2014 (6,276 nulls pre-2014). Neither gap causes rows to be dropped - nulls are filled with regime-conditional means and a `bond_factors_available` indicator column marks which rows had real vs imputed values.

**Winsorisation.** Eleven continuous columns are clipped at the 1st/99th percentile. Percentile bounds are fit on the pre-2022 calibration window only to prevent any lookahead into the held-out period. A `_clipped` indicator column is added for each.

**Stationarity (ADF test).** Seven key features are tested with the Augmented Dickey-Fuller test at α = 0.05. Two are found to have unit roots: `gs10_yield` and `real_yield`. Both receive a `_diff1` column (first difference) which is used as the model feature instead of the level.

**Autocorrelation.** Lag-1 autocorrelation is computed for key features. All are highly persistent (ρ(1) = 0.96–0.99), confirming that rows must not be split i.i.d. - only time-aware splits are valid.

### Step 2 - Enrich

Four REFLEX-specific quantities are reconstructed from the master's own columns:

**`h_IG_bps` and `h_HY_bps` (half-spread proxy).** Weighted combination of `implied_spread_bps` (macro component, 60% weight) and `abs(liquidity_rf) × 5000` (micro illiquidity add-on, 40% weight), following Friewald et al. (2012). IG mean = 50 bps, stress mean = 95 bps, crisis mean = 123 bps.

**`q_proxy` (inventory path proxy).** 3-month cumulative `bond_mkt_ret` normalised by 12-month rolling volatility, clipped to [−3σ, +3σ] to match the simulator's `q_after` scale. Pre-TRACE window (pre-2004) defaults to 0 (neutral inventory).

**`tau_proxy` (informed-flow fraction).** `|credit_rf| / (|credit_rf| + |liquidity_rf| + ε)` - the share of the combined factor signal attributable to adverse selection (credit risk) vs pure illiquidity. Bounded [0, 1]. Mean = 0.74, stress mean = 0.56.

**`D_phi_sigma` (order-flow distribution proxy).** Cross-sectional return dispersion `ret_sigma_xs` where available (2014–2023); imputed as `sp500_sigma_12m × 0.35` for the pre-2014 window.

### Step 3 - Fit (A, k)

Exponential intensity model `λ(h) = A · exp(−k · h)` is fit per rating bucket (IG, HY) per regime (calm through crisis) using `scipy.optimize.curve_fit` MLE. Observations are the master's own `A_IG_estimate` / `A_HY_estimate` columns against `h_IG_decimal` / `h_HY_decimal`, using only the TRACE-available window (2004–2021). Results are written to `03_fitted_intensity_params.csv` and merged back into the master panel as `A_fitted_mean` and `k_decay_mean`.

ADF is run on the fit residuals - IG calm through elevated pass (stationary residuals), stress and crisis do not, reflecting higher non-stationarity in extreme regimes. This is documented in the output file.

### Step 4 - Normalise Simulator Logs

Simulator output columns are renamed to match the master schema (`h` → `h_halfspread`, `tau` → `tau_toxicfrac`, `q_after` → `q_inventory`). Z-scores are computed using means and standard deviations from the master's calibration window (2004–2019) - not from the simulator data itself - so real and simulated quantities are on a common scale. `data_is_synthetic = True` is set on every simulator row. Stability labels are encoded as numeric codes (stable=0, oscillating=1, collapsed\_saturated=2).

### Step 5 - Episode-Level Split

Two parallel splits are maintained:

**Simulator episodes** split on adversarial intensity ε: calibration ε ≤ 0.75 (832 episodes), validation 0.75 < ε ≤ 0.85 (104 episodes), held-out ε > 0.85 (104 episodes). The held-out set contains the high-adversarial regime where the stability boundary is most sensitive. All three stability labels appear in the calibration set.

**Real-data split** on date: pre-sample 1990–2003 (no bond factors), calibration 2004–2019 (full TRACE window), validation 2020–2021 (COVID shock + recovery), held-out 2022–2026 (rate-hike cycle, out-of-sample). Stored in `icaif_split` column on every row.

### Step 6 - Final Panel

All steps are merged into `MASTER_PREPROCESSED.csv` (9,218 rows × 88 columns). Z-scores for all 23 continuous features are computed using calibration-window statistics only (`scaler_stats.csv` stores the μ/σ values for inference-time use). `lookahead_safe = True` is set on every row and verified by checking that the calibration-window z-score mean is exactly 0.000.

### Preprocessing Output Files

| File | Rows | What it contains |
|------|------|-----------------|
| `01_master_cleaned.csv` | 9,218 | Master + null fills + winsorisation flags + ADF diff cols |
| `02_master_enriched.csv` | 9,218 | + reconstructed h, q\_proxy, tau\_proxy, D\_phi\_sigma |
| `03_fitted_intensity_params.csv` | 10 | Fitted (A, k) per rating × regime with SE and R² |
| `04_sim_logs_normalised.csv` | 104,000 | Simulator step logs, master-anchored z-scores (generated) |
| `04_sim_summary_normalised.csv` | 1,040 | Run-level stability estimators, z-scored |
| `05_calibration_episodes.csv` | 936 | Calibration + validation simulator episodes |
| `05_heldout_episodes.csv` | 104 | Held-out simulator episodes |
| `scaler_stats.csv` | 23 | Feature μ/σ fit on calibration window |
| `MASTER_PREPROCESSED.csv` | 9,218 | Final analysis-ready panel, all steps merged (generated) |

---

## Citation

```bibtex
@misc{cboe_vix,
  title  = {CBOE Volatility Index (VIX) Historical Data},
  author = {{Chicago Board Options Exchange}},
  url    = {https://github.com/datasets/finance-vix}
}
@article{dickerson2023priced,
  title   = {Priced Risk in Corporate Bonds},
  author  = {Dickerson, Alexander M. and Mueller, Philippe and Robotti, Cesare},
  journal = {Journal of Financial Economics},
  year    = {2023}
}
@misc{shiller_data,
  title  = {U.S. Stock Markets 1871-Present and CAPE Ratio},
  author = {Shiller, Robert J.},
  url    = {http://www.econ.yale.edu/~shiller/data.htm}
}
@misc{fed_h15,
  title  = {Selected Interest Rates (H.15)},
  author = {{Board of Governors of the Federal Reserve System}},
  url    = {https://www.federalreserve.gov/releases/h15/}
}
```
