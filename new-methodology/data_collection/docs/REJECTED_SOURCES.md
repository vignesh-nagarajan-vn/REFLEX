# Rejected Sources

Sources evaluated and rejected during data collection. Documented here to prevent future re-inclusion and to be transparent about the collection process.

---

## 1. Pooja2420/Liquidity-Scoring (all data files)

**URL:** https://github.com/Pooja2420/Liquidity-Scoring  
**Files evaluated:** `bond_universe.csv`, `trace_data.parquet`, `features.parquet`, `impact_params.parquet`, `macro_factors.parquet`  
**Rejection reason: Explicitly synthetic**

The source code states this unambiguously:

**`src/data/trace_loader.py`** (docstring):
> *"In production this reads from FINRA TRACE Enhanced (via WRDS or vendor feed). Here we generate realistic synthetic intraday bond trade data that mirrors the statistical properties of TRACE: bursty arrivals, size distributions, price dynamics with bid-ask bounce."*

**`src/data/bond_reference.py`** (docstring):
> *"In production this would connect to a data vendor (Bloomberg, ICE, Refinitiv). Here we generate a realistic synthetic universe of US corporate bonds."*

**`src/data/macro_factors.py`**:
> Falls back to `_generate_synthetic_macro()` when no FRED API key is configured. The committed parquet file uses this synthetic fallback — confirmed by the VIX series showing a mean-reverting synthetic path rather than matching CBOE published values.

The committed CSV/parquet files in this repo are the output of these synthetic generators, not real TRACE data. CUSIPs are `CUSIP000000`, `CUSIP000001`, etc. — fabricated placeholders. All five data files from this source have been removed from the REFLEX dataset.

---

## 2. QuhiQuhihi — weights_1n_df.csv and weights_mv_df.csv

**URL:** https://github.com/QuhiQuhihi/Factor-Strategy-for-Corporate-Bond-  
**Files rejected:** `weights_1n_df.csv`, `weights_mv_df.csv`  
**Rejection reason: Misnamed columns — not return data**

The column names (`LQD_Adjusted_Return`, `HYG_Adjusted_Return`, etc.) suggest ETF return series. They are in fact **portfolio weights** output by a mean-variance optimizer. The equal-weight file contains 0.20 for every cell in `LQD_Adjusted_Return` across all 113 months — a constant allocation weight, not a return.

The `Monthly_Log_Returns.csv` file from the same repo was retained, as it contains real CUSIP-level bond returns and passed verification.

---

## 3. ICE BofA OAS Series (FRED BAMLC0A*)

**URL:** https://fred.stlouisfed.org/series/BAMLC0A0CM  
**Not accessible for redistribution**

These are the ideal corporate bond spread series for REFLEX calibration. However FRED's own series notes state:

> *"The end of day Index values, Index returns, and Index statistics ('Top Level Data') are being provided for your internal use only and you are not authorized or permitted to publish, distribute or otherwise furnish Top Level Data to any third-party without prior written approval of ICE Data."*

This makes bundling the series into a redistributable dataset a license violation regardless of how it was accessed. Additionally, FRED only exposes the 5 most recent observations through the standard page view; full historical access requires the FRED API, which is blocked from the collection environment.

**Status:** Cannot be included. Acknowledged as a limitation in the paper's data section (Section 3.9).

---

## 4. MarketAxess / Tradeweb post-trade data

**Rejection reason:** Commercially licensed. Requires direct contract with the platform. No free tier for historical data.

---

## 5. Bloomberg / Refinitiv terminal data

**Rejection reason:** Terminal subscription required. No public API or free download path.

---

## 6. FINRA TRACE Enhanced (via WRDS)

**Rejection reason:** Requires institutional WRDS subscription and academic affiliation verification. Trade-level dealer-side data (price, size, timestamp, direction) is the gold standard for OTC bond microstructure research but is not freely redistributable. Application path: https://wrds-www.wharton.upenn.edu/

**Note:** Dickerson et al. (2023) processed TRACE Enhanced data and released their derived *factor returns* (Dataset F in this repo), which are freely redistributable. The underlying trade prints are not.
