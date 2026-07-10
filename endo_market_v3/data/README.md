# data/ — shipped calibration artifacts (provenance)

Self-contained copies of the REFLEX data pipeline's outputs so `endo_market_v3`
runs without reaching outside its folder. The **canonical pipeline** (fetch,
build, verify, preprocess — with full docs, verification logs and rejected
sources) lives in [`../../new-methodology/`](../../new-methodology/)
(`data_collection/` + `preprocessing/`). Regenerate everything from public
sources with its four scripts (`fetch_data.py`, `build_datasets.py`,
`verify_data.py`, `preprocess.py`).

| File | What it is | Used by |
|------|-----------|---------|
| `calibration/03_fitted_intensity_params.csv` | Exponential-intensity fits `λ(h) = A·exp(−k·h)` per (rating × regime), 2004–2021 window | `reflex.calibration` (simulator calibration), `reflex.analysis.fragility` (per-regime `k`, `h`) |
| `calibration/reflex_I_calibration_params.csv` | Per-regime macro summary (VIX, yields, spreads, σ ranges) | cross-checks / tests |
| `calibration/reflex_G2_bond_xsection_sigma.csv` | Monthly cross-sectional dispersion of 212 real-CUSIP bond returns | cross-checks (aggregated; cannot identify per-bond vol heterogeneity) |
| `calibration/reflex_G_bond_returns_monthly.csv` | Per-bond monthly log returns for the 212 real CUSIPs (raw, unwinsorised) | per-bond σ dispersion for factor scaling (1.5): `loader.bond_vol_dispersion` drops stale (zero-vol) bonds, winsorises σ at p05/p95, returns the CV |
| `calibration/scaler_stats.csv` | Feature μ/σ fit on the 2004–2019 calibration window | lookahead-safe normalisation |
| `master/REFLEX_MASTER_DATASET.csv` | Daily 1990–2026 joined panel (9,218 rows × 36 cols; VIX spine) | the fragility index (`run_fragility`) |

**Provenance / honesty.** All series derive from public, verified sources
(CBOE VIX, EIA WTI, Fed H.15, Shiller/Yale, World Gold Council + BLS,
Dickerson–Mueller–Robotti (2023 JFE) TRACE-derived factors, real-CUSIP bond
returns). This is **not trade-level TRACE**: per-dealer spreads/inventories and
per-bond trade prints require WRDS TRACE Enhanced (pending). The intensity fits
are VIX-proxy-based; consequently only the **(A, k, σ, h) regime structure** is
data-identified, while the toxic/informed channel in `reflex.calibration.mapping`
is *structurally scaled* (documented ratios). State this plainly in any paper
that uses these calibrations. Full source citations and verification values:
[`../../new-methodology/data_collection/README.md`](../../new-methodology/data_collection/README.md)
and its `docs/` folder.
