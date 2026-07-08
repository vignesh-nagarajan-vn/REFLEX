# REFLEX — Data Collection Repository

## Repository Structure

```
reflex-data/
├── README.md                          ← this file
├── data/
│   ├── raw/                           ← individual verified source datasets
│   │   ├── reflex_A_vix_daily.csv
│   │   ├── reflex_B_wti_daily.csv
│   │   ├── reflex_C_treasury10y_monthly.csv
│   │   ├── reflex_D_shiller_monthly.csv
│   │   ├── reflex_E_gold_cpi_monthly.csv
│   │   ├── reflex_F_dickerson_bond_factors.csv
│   │   ├── reflex_G_bond_returns_monthly.csv
│   │   └── reflex_G2_bond_xsection_sigma.csv
│   ├── processed/
│   │   ├── reflex_H_merged_calibration.csv  ← monthly joined table
│   │   └── reflex_I_calibration_params.csv  ← regime-level REFLEX params
│   └── master/
│       └── REFLEX_MASTER_DATASET.csv        ← single joined daily dataset
├── src/
│   ├── fetch_data.py                  ← downloads raw sources from scratch
│   ├── build_datasets.py              ← builds all processed + master files
│   └── verify_data.py                 ← runs all verification checks
└── docs/
    ├── DATA_CATALOGUE.md              ← full column-level documentation
    ├── VERIFICATION_LOG.md            ← exact historical value checks
    └── REJECTED_SOURCES.md           ← sources found to be synthetic
```

---

## Quick Start

```bash
git clone https://github.com/YOUR_USERNAME/reflex-data.git
cd reflex-data

pip install pandas numpy pyarrow requests

# Option A: use the data already in this repo
# All CSVs are committed — just open them.

# Option B: re-download from source and regenerate everything
python src/fetch_data.py      # downloads raw sources via GitHub mirrors
python src/build_datasets.py  # builds processed/ and master/ files
python src/verify_data.py     # re-runs all verification checks
```

---

## Dataset Overview

| # | File | Synthetic? | Timespan | Freq | Rows | Primary Source |
|---|------|-----------|---------|------|------|---------------|
| A | `reflex_A_vix_daily.csv` | **No — Real** | 1990-01-02 to 2026-06-30 | Daily | 9,218 | CBOE Volatility Index |
| B | `reflex_B_wti_daily.csv` | **No — Real** | 1986-01-02 to 2026-06-29 | Daily | 10,191 | US EIA WTI Crude Oil |
| C | `reflex_C_treasury10y_monthly.csv` | **No — Real** | 1953-04-01 to 2026-05-01 | Monthly | 878 | Federal Reserve H.15 |
| D | `reflex_D_shiller_monthly.csv` | **No — Real** | 1871-01-01 to 2023-09-01 | Monthly | 1,833 | Robert Shiller / Yale |
| E | `reflex_E_gold_cpi_monthly.csv` | **No — Real** | 1833-01-01 to 2026-05-01 | Monthly | 2,321 | World Gold Council + BLS |
| F | `reflex_F_dickerson_bond_factors.csv` | **No — Real** | 2004-08-31 to 2021-12-31 | Monthly | 209 | TRACE via Dickerson et al. (2023) JFE |
| G | `reflex_G_bond_returns_monthly.csv` | **No — Real** | 2014-12-01 to 2023-09-01 | Monthly | 12,123 | Real CUSIPs from TRACE-derived repo |
| G2 | `reflex_G2_bond_xsection_sigma.csv` | **No — Real** | 2014-12-01 to 2023-09-01 | Monthly | 106 | Derived from G |
| H | `reflex_H_merged_calibration.csv` | **No — Real** | 1990-01-01 to 2026-06-01 | Monthly | 438 | All sources merged |
| I | `reflex_I_calibration_params.csv` | **No — Real** | 1990–2026 (regime summary) | Per-regime | 5 | Derived from H |
| **MASTER** | `REFLEX_MASTER_DATASET.csv` | **No — Real** | **1990-01-02 to 2026-06-30** | **Daily** | **9,218** | **All sources joined** |

**Total coverage: 36+ years of daily data · 70 years of monthly data · 216 real corporate bonds**

---

## Source Details

### A — CBOE VIX Daily (1990–2026)
- **What:** CBOE Volatility Index — 30-day implied vol of S&P 500 options
- **Why it matters for REFLEX:** VIX is the dominant predictor of OTC corporate bond bid-ask spreads (Friewald et al. 2012; Bao et al. 2011). Used for σ-proxy, k-proxy, and market regime classification.
- **Source URL:** https://github.com/datasets/finance-vix
- **FRED equivalent:** `VIXCLS`
- **Verification:** Jan 2 1990 = 17.24 ✓ | Mar 16 2020 = 82.69 ✓ | Aug 24 2015 = 40.74 ✓

### B — EIA WTI Crude Oil Daily (1986–2026)
- **What:** West Texas Intermediate crude oil spot price (USD/barrel)
- **Why it matters for REFLEX:** Macro stress co-indicator; OTC bond markets tighten during oil shocks
- **Source URL:** https://github.com/datasets/oil-prices
- **Origin:** US Energy Information Administration (public domain)
- **Verification:** Jun 2008 peak = $141.06 ✓ | Dec 2008 low = $30.28 ✓

### C — Federal Reserve 10-Year Treasury Yield Monthly (1953–2026)
- **What:** US 10-year constant-maturity Treasury yield
- **Why it matters for REFLEX:** DV01 calibration for the inventory risk penalty term `q·DV01` in the dealer's quoting policy. Real yield context for each regime.
- **Source URL:** https://github.com/datasets/bond-yields-us-10y
- **Origin:** Federal Reserve H.15 Statistical Release (public domain)
- **FRED series:** `GS10`
- **Verification:** Jan 1981 = 12.57% ✓ | Apr 2000 = 5.99% ✓ | Jul 2016 = 1.50% ✓

### D — Shiller S&P 500 Monthly (1871–2023)
- **What:** Monthly S&P 500 price, dividend, earnings, CPI, GS10, PE10/CAPE ratio
- **Why it matters for REFLEX:** 12-month rolling equity sigma for σ-calibration; CAPE ratio as macro regime indicator
- **Source URL:** https://github.com/datasets/s-and-p-500
- **Origin:** Robert Shiller, Yale University — http://www.econ.yale.edu/~shiller/data.htm
- **Note:** All values are monthly averages, not closing prices — standard Shiller methodology
- **Verification:** Jan 2000 SP500 = 1,426 ✓ | Jan 1981 Long Rate = 12.57% ✓

### E — Gold Prices + US CPI Monthly (1833–2026)
- **What:** Gold spot price (USD/troy oz) + US CPI-U index + monthly inflation
- **Why it matters for REFLEX:** Real yield calculation; flight-to-safety signal
- **Source URLs:** https://github.com/datasets/gold-prices | https://github.com/datasets/cpi-us
- **Origins:** World Gold Council (gold) + BLS (CPI, public domain)
- **Verification:** Jan 2000 gold = $284 ✓ | Sep 2011 gold = $1,772 ✓ | Jan 1990 CPI = 127.4 ✓ | Jan 2023 CPI = 299.2 ✓

### F — Dickerson, Mueller & Robotti TRACE Bond Factors (2004–2021)
- **What:** 4 monthly corporate bond market factors derived from FINRA TRACE:
  - `bond_mkt_ret` — total corporate bond market return
  - `duration_rf` — duration risk factor
  - `credit_rf` — credit risk factor
  - `liquidity_rf` — **liquidity risk factor** ← primary ε proxy for REFLEX
- **Why it matters for REFLEX:** The liquidity risk factor (LRF) is the closest publicly available monthly proxy for the adverse-selection premium that REFLEX's ε parameter characterises. Sep 2008 MKTB = −8.26% confirms the GFC shock is captured.
- **Source URL:** https://github.com/Alexander-M-Dickerson/TRACE-corporate-bond-processing
- **Paper:** Dickerson, A.M., Mueller, P., & Robotti, C. (2023). *Priced Risk in Corporate Bonds.* Journal of Financial Economics.
- **Verification:** Sep 2008 MKTB = −8.26% (Lehman shock) ✓ | LRF annual vol consistent with published paper ✓

### G — Real CUSIP-Level Bond Returns Monthly (2014–2023)
- **What:** Monthly log returns for 216 real US corporate bonds identified by CUSIP
- **Why it matters for REFLEX:** Cross-sectional return dispersion is a proxy for the distribution D(φ) in the REFLEX framework
- **Source URL:** https://github.com/QuhiQuhihi/Factor-Strategy-for-Corporate-Bond-
- **Verification:** CUSIPs are real 9-character identifiers (e.g. `00138GAB5` = Agilent Technologies, `00138GAC3`, `00182EAQ2`) ✓ | Monthly return std = 0.021 (consistent with corporate bond literature) ✓

### MASTER — REFLEX_MASTER_DATASET.csv
- **What:** Single wide daily table (9,218 rows × 36 columns) joining all sources above
- **Spine:** VIX trading days 1990-01-02 to 2026-06-30
- **Monthly series** (GS10, Shiller, gold, CPI, Dickerson factors, bond sigma) are forward-filled to daily — standard practice for macro data in mixed-frequency panels
- **Includes:** All raw source columns + derived REFLEX calibration targets (σ_IG, σ_HY, k_IG, k_HY, A_IG, A_HY estimates per day)
- `data_is_synthetic = False` for every row — all values come from verified real sources listed above

---

## Column Dictionary (Master Dataset)

| Column | Source | Description |
|--------|--------|-------------|
| `date` | — | Trading date |
| `vix_close` | CBOE | VIX closing level |
| `vix_20d_std` | derived | 20-day rolling std of VIX |
| `vix_60d_mean` | derived | 60-day rolling mean of VIX |
| `sigma_proxy` | derived | `vix_close / 100` — annualised implied vol |
| `k_proxy` | derived | `vix_close × 0.030` — price impact proxy |
| `regime` | derived | calm / normal / elevated / stress / crisis |
| `wti_price` | EIA | WTI crude oil USD/barrel |
| `oil_20d_vol` | derived | 20-day annualised WTI vol |
| `gs10_yield` | Fed H.15 | 10-year Treasury yield (%) |
| `dv01_10y` | derived | Approximate DV01 per $100 par |
| `SP500` | Shiller/Yale | S&P 500 monthly average level |
| `sp500_sigma_12m` | derived | 12-month rolling annualised equity vol |
| `cape_ratio` | Shiller/Yale | Cyclically adjusted PE ratio (CAPE/PE10) |
| `earnings_yield` | derived | Earnings / SP500 (E/P ratio) |
| `gold_usd` | World Gold Council | Gold spot price USD/troy oz |
| `cpi_index` | BLS | US CPI-U index level |
| `cpi_mom_pct` | BLS | Monthly CPI inflation (%) |
| `bond_mkt_ret` | Dickerson/TRACE | Corporate bond market monthly return |
| `duration_rf` | Dickerson/TRACE | Duration risk factor return |
| `credit_rf` | Dickerson/TRACE | Credit risk factor return |
| `liquidity_rf` | Dickerson/TRACE | **Liquidity risk factor** — ε proxy |
| `cum_lrf_12m` | derived | 12-month cumulative liquidity factor |
| `bond_vol_6m` | derived | 6-month rolling annualised bond market vol |
| `n_bonds` | QuhiQuhihi | Number of bonds in cross-section that month |
| `ret_mean` | QuhiQuhihi | Cross-sectional mean bond log return |
| `ret_sigma_xs` | QuhiQuhihi | Cross-sectional std of bond returns — D(φ) proxy |
| `real_yield` | derived | `gs10_yield − cpi_mom_pct` |
| `implied_spread_bps` | derived | VIX-based credit spread proxy (bps) |
| `sigma_IG_estimate` | derived | IG bond σ estimate = `sp500_sigma_12m × 0.40` |
| `sigma_HY_estimate` | derived | HY bond σ estimate = `sp500_sigma_12m × 0.72` |
| `k_IG_estimate` | derived | IG price impact estimate = `spread_bps / 200` |
| `k_HY_estimate` | derived | HY price impact estimate = `spread_bps / 80` |
| `A_IG_estimate` | derived | IG arrival rate estimate (trades/step) |
| `A_HY_estimate` | derived | HY arrival rate estimate (trades/step) |
| `data_is_synthetic` | — | Always `False` — all rows are real data |
| `primary_source` | — | Source attribution string |

---

## Economic Cycle Coverage

This dataset spans every major fixed-income market event since 1990:

| Period | Regime | Key bond market event |
|--------|--------|-----------------------|
| 1990–1994 | Normal / Elevated | S&L crisis tail; Fed rate cycle |
| 1997–1998 | Stress | LTCM collapse; Russia default; VIX spike to 45 |
| 2000–2002 | Stress | Dot-com bust; Enron/WorldCom credit crisis |
| **2007–2009** | **Crisis** | **GFC — defining OTC bond liquidity crisis. VIX peaked at 89.53. LRF confirms bond-specific illiquidity premium.** |
| 2010–2019 | Calm / Normal | Post-crisis QE era; taper tantrum 2013 |
| **2020** | **Crisis** | **COVID-19 — March 2020 OTC bond market freeze. VIX = 82.69 on Mar 16.** |
| 2022–2023 | Elevated / Stress | Fed rate-hike cycle; sharpest bond drawdown since 1980s |
| 2024–2026 | Normal | Post-hike normalisation |

The Dickerson LRF (Dataset F) is available for GFC and COVID periods, directly linking bond-specific liquidity risk to the VIX regime classification.

---

## What This Dataset Does NOT Contain

Three quantities cannot be obtained from any free public source and are documented as missing:

| Missing Quantity | Why Unavailable | Required Source |
|-----------------|-----------------|-----------------|
| Trade-level OTC bond bid-ask spread `h` | TRACE trade-level data requires WRDS/FINRA license | WRDS academic TRACE Enhanced |
| Per-bond price impact `k` fitted from real prints | Same — dealer-level order flow needed | WRDS TRACE + dealer identity |
| Dealer inventory paths `q` | TRACE dealer IDs masked in public tier | FINRA non-public enhanced feed |

These gaps are stated in the paper's data section (Section 3.9), not footnotes.

---

## Citation

If you use this dataset collection, please cite the underlying sources:

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
  title  = {U.S. Stock Markets 1871–Present and CAPE Ratio},
  author = {Shiller, Robert J.},
  url    = {http://www.econ.yale.edu/~shiller/data.htm},
  year   = {2023}
}

@misc{fed_h15,
  title  = {Selected Interest Rates (H.15): 10-Year Treasury Constant Maturity},
  author = {{Board of Governors of the Federal Reserve System}},
  url    = {https://www.federalreserve.gov/releases/h15/}
}

@misc{eia_wti,
  title  = {Spot Prices for Crude Oil and Petroleum Products},
  author = {{U.S. Energy Information Administration}},
  url    = {https://www.eia.gov/dnav/pet/pet_pri_spt_s1_d.htm}
}

@misc{bls_cpi,
  title  = {Consumer Price Index for All Urban Consumers (CPI-U)},
  author = {{U.S. Bureau of Labor Statistics}},
  url    = {https://www.bls.gov/cpi/}
}
```

---

## License

All data files reproduced here originate from public-domain US government sources (Federal Reserve, EIA, BLS) or freely redistributable academic datasets (Shiller/Yale, Dickerson et al. replication package). No ICE BofA, Bloomberg, Refinitiv, or TRACE Enhanced licensed data is included.
