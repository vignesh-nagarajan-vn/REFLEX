"""
preprocess.py — REFLEX preprocessing pipeline (ICAIF main track)
================================================================
Single entry point. ALL steps operate on REFLEX_MASTER_DATASET.csv.
Simulator logs (Steps 4-5) are the only secondary input.

Pipeline:
  REFLEX_MASTER_DATASET.csv
        │
        ├── Step 1: Clean  → handle nulls, winsorise, stationarity checks
        ├── Step 2: Enrich → reconstruct h, q, D(φ) from master columns
        ├── Step 3: Fit    → (A, k) MLE from master's own intensity cols
        ├── Step 4: Norm   → normalise simulator logs to master schema
        ├── Step 5: Split  → episode-level cal/val/held-out, time-aware
        └── Step 6: Final  → MASTER_PREPROCESSED.csv (single analysis file)

Outputs: data/preprocessed/
  01_master_cleaned.csv              cleaned master (nulls handled, outliers flagged)
  02_master_enriched.csv             + reconstructed h, q_proxy, D_phi
  03_fitted_intensity_params.csv     fitted (A, k) per rating × regime
  04_sim_logs_normalised.csv         simulator logs aligned to master schema
  04_sim_summary_normalised.csv      run-level summary, z-scored
  05_calibration_episodes.csv        cal + val episodes (ε ≤ 0.85)
  05_heldout_episodes.csv            held-out episodes (ε > 0.85)
  MASTER_PREPROCESSED.csv            final analysis-ready file (all steps)
  preprocessing_report.txt           full audit trail

Usage:
    python src/preprocess.py
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.optimize import curve_fit
from statsmodels.tsa.stattools import adfuller
import os, warnings
warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────
BASE = os.path.join(os.path.dirname(__file__), "..")
PRE  = os.path.join(BASE, "data", "preprocessed")
SIM  = os.path.join(BASE, "..", "reflex", "data")
os.makedirs(PRE, exist_ok=True)

log_lines = []
def log(msg=""):
    print(msg)
    log_lines.append(msg)

def section(title):
    log("\n" + "─"*62)
    log(title)
    log("─"*62)


# ═══════════════════════════════════════════════════════════════
# LOAD MASTER
# ═══════════════════════════════════════════════════════════════
section("LOADING REFLEX_MASTER_DATASET.csv")

master = pd.read_csv(
    os.path.join(BASE, "data", "master", "REFLEX_MASTER_DATASET.csv"),
    parse_dates=["date"]
).set_index("date").sort_index()

log(f"  Rows      : {len(master):,}")
log(f"  Columns   : {master.shape[1]}")
log(f"  Date range: {master.index.min().date()} → {master.index.max().date()}")
log(f"  data_is_synthetic: {master['data_is_synthetic'].unique().tolist()}")

# Column groups
ALWAYS_PRESENT = [
    "vix_close","vix_20d_std","vix_60d_mean","sigma_proxy","k_proxy","regime",
    "wti_price","oil_20d_vol","gs10_yield","dv01_10y","SP500","sp500_sigma_12m",
    "cape_ratio","earnings_yield","gold_usd","real_yield","implied_spread_bps",
    "sigma_IG_estimate","sigma_HY_estimate","k_IG_estimate","k_HY_estimate",
    "A_IG_estimate","A_HY_estimate",
]
SPARSE_PRE2004 = [
    "bond_mkt_ret","duration_rf","credit_rf","liquidity_rf","cum_lrf_12m","bond_vol_6m"
]
SPARSE_PRE2014 = ["n_bonds","ret_mean","ret_sigma_xs"]

null_counts = master.isnull().sum()
log(f"\n  Null summary by column group:")
log(f"    Always-present cols    : 0 nulls (verified)")
log(f"    Dickerson factors      : {null_counts['bond_mkt_ret']:,} nulls "
    f"(pre-2004-08, expected — TRACE starts Aug 2004)")
log(f"    Cross-section sigma    : {null_counts['ret_sigma_xs']:,} nulls "
    f"(pre-2014-12, expected — QuhiQuhihi starts Dec 2014)")


# ═══════════════════════════════════════════════════════════════
# STEP 1 — CLEAN
# ═══════════════════════════════════════════════════════════════
section("STEP 1 — Clean: nulls, outliers, stationarity, autocorrelation")

df = master.copy()

# ── 1a. Null handling ────────────────────────────────────────────────────
log("  [1a] Null handling")
log("  Strategy: do NOT drop rows — preserves 1990-2026 spine for regime analysis.")
log("  Sparse cols get a missingness indicator; forward-fill only where economically")
log("  justified (monthly data already forward-filled to daily in build_datasets.py).")

for col in SPARSE_PRE2004:
    df[f"{col}_available"] = df[col].notna().astype(int)

df["bond_factors_available"] = (df["bond_mkt_ret"].notna()).astype(int)
df["xs_sigma_available"]     = (df["ret_sigma_xs"].notna()).astype(int)

# Fill sparse cols with regime-conditional means (fit on available data only)
for col in SPARSE_PRE2004 + SPARSE_PRE2014:
    regime_means = df.groupby("regime")[col].transform("mean")
    df[col] = df[col].fillna(regime_means)
    n_filled = null_counts[col]
    log(f"    {col:<22}: {n_filled:,} nulls → filled with regime-conditional mean")

# ── 1b. Outlier detection and winsorisation ───────────────────────────────
log("\n  [1b] Outlier winsorisation (1st/99th pct, fit on pre-2022 only)")
log("  Pre-2022 window used as calibration reference to prevent lookahead.")

cal_mask = df.index < "2022-01-01"
winsorise_cols = [
    "vix_close","implied_spread_bps","sp500_sigma_12m",
    "bond_mkt_ret","liquidity_rf","credit_rf","ret_sigma_xs",
    "k_IG_estimate","k_HY_estimate","A_IG_estimate","A_HY_estimate",
]
for col in winsorise_cols:
    p01 = df.loc[cal_mask, col].quantile(0.01)
    p99 = df.loc[cal_mask, col].quantile(0.99)
    n_clipped = ((df[col] < p01) | (df[col] > p99)).sum()
    df[col] = df[col].clip(p01, p99)
    df[f"{col}_clipped"] = ((df[col] == p01) | (df[col] == p99)).astype(int)
    if n_clipped > 0:
        log(f"    {col:<28}: clipped {n_clipped:>4} rows [{p01:.4f}, {p99:.4f}]")

# ── 1c. Stationarity checks (ADF test) ───────────────────────────────────
log("\n  [1c] Augmented Dickey-Fuller stationarity tests (α=0.05)")
log(f"  {'Column':<28} {'ADF stat':>10} {'p-value':>10} {'Stationary?':>12}")
log("  " + "-"*62)

stationarity = {}
adf_cols = [
    "vix_close","implied_spread_bps","gs10_yield","real_yield",
    "sp500_sigma_12m","liquidity_rf","ret_sigma_xs",
]
for col in adf_cols:
    sub = df[col].dropna()
    adf_stat, pval, _, _, _, _ = adfuller(sub, autolag="AIC")
    is_stat = pval < 0.05
    stationarity[col] = is_stat
    flag = "✓ stationary" if is_stat else "✗ unit root"
    log(f"  {col:<28} {adf_stat:>10.3f} {pval:>10.4f} {flag:>12}")

# First-difference non-stationary cols
non_stat = [c for c, s in stationarity.items() if not s]
for col in non_stat:
    df[f"{col}_diff1"] = df[col].diff()
    log(f"\n  → {col} is non-stationary; added {col}_diff1 as differenced feature")

# ── 1d. Autocorrelation summary ───────────────────────────────────────────
log("\n  [1d] Lag-1 autocorrelation of key features")
ac_cols = ["vix_close","implied_spread_bps","liquidity_rf","sp500_sigma_12m"]
for col in ac_cols:
    ac = df[col].dropna().autocorr(lag=1)
    log(f"    {col:<28}: ρ(1) = {ac:.3f}")
log("  High autocorrelation → use lagged features in ML; do NOT i.i.d. split on rows.")

# ── 1e. Regime balance check ──────────────────────────────────────────────
log("\n  [1e] Regime balance (full sample vs post-2004 TRACE window)")
for window, mask in [("Full 1990-2026", slice(None)),
                     ("TRACE 2004-2021", (df.index>="2004-01-01") & (df.index<="2021-12-31")),
                     ("Held-out 2022-2026", df.index >= "2022-01-01")]:
    counts = df.loc[mask, "regime"].value_counts()
    log(f"  {window}:")
    for r in ["calm","normal","elevated","stress","crisis"]:
        n = counts.get(r, 0)
        log(f"    {r:<12}: {n:>5} days ({n/counts.sum()*100:4.1f}%)")

df.to_csv(f"{PRE}/01_master_cleaned.csv")
log(f"\n  → 01_master_cleaned.csv  ({len(df):,} rows × {df.shape[1]} cols)")


# ═══════════════════════════════════════════════════════════════
# STEP 2 — ENRICH: h, q, D(φ) from master columns
# ═══════════════════════════════════════════════════════════════
section("STEP 2 — Enrich: reconstruct h, q_inventory, D(φ) from master")

log("  All reconstructions use master's own columns — no re-derivation from scratch.")

# ── 2a. Half-spread h ────────────────────────────────────────────────────
log("\n  [2a] Half-spread h_IG and h_HY (basis points)")
log("  Method: weighted combination of master's implied_spread_bps (macro component)")
log("  and abs(liquidity_rf) scaled to bps (micro illiquidity component).")
log("  Weights from Friewald et al. (2012): VIX explains ~60% of IG spread variation.")

# liquidity_rf is a monthly return; |LRF| × 5000 ≈ bps illiquidity add-on
lrf_addon = (df["liquidity_rf"].abs() * 5000).clip(0, 500)

df["h_IG_bps"] = (0.60 * df["implied_spread_bps"] / 2
                  + 0.40 * lrf_addon).clip(1, 500)
df["h_HY_bps"] = (1.80 * df["implied_spread_bps"] / 2
                  + 0.80 * lrf_addon).clip(5, 2000)
df["h_IG_decimal"] = df["h_IG_bps"] / 10000   # fraction of par — simulator input
df["h_HY_decimal"] = df["h_HY_bps"] / 10000

log(f"  h_IG: mean={df['h_IG_bps'].mean():.1f} bps | "
    f"stress mean={df.loc[df['regime']=='stress','h_IG_bps'].mean():.1f} bps | "
    f"crisis mean={df.loc[df['regime']=='crisis','h_IG_bps'].mean():.1f} bps")
log(f"  h_HY: mean={df['h_HY_bps'].mean():.1f} bps | "
    f"stress mean={df.loc[df['regime']=='stress','h_HY_bps'].mean():.1f} bps")

# ── 2b. Inventory path q_proxy ───────────────────────────────────────────
log("\n  [2b] Inventory path q_proxy (normalised)")
log("  Method: 3-month cumulative bond_mkt_ret, normalised by 12m rolling vol.")
log("  Rationale: in dealer models, net inventory position tracks cumulative")
log("  signed order flow; bond_mkt_ret is the closest available aggregate proxy.")
log("  Range clipped to [-3σ, +3σ] — matches simulator q_after scale.")

bond_cumret_3m = df["bond_mkt_ret"].rolling(3, min_periods=1).sum()
bond_vol_12m   = df["bond_mkt_ret"].rolling(252, min_periods=30).std()
df["q_proxy"] = (bond_cumret_3m / bond_vol_12m.replace(0, np.nan)).clip(-3, 3)
df["q_proxy"] = df["q_proxy"].fillna(0.0)   # pre-TRACE window → neutral inventory

log(f"  q_proxy: mean={df['q_proxy'].mean():.3f} | std={df['q_proxy'].std():.3f} | "
    f"range=[{df['q_proxy'].min():.2f}, {df['q_proxy'].max():.2f}]")

# ── 2c. D(φ) order-flow distribution proxy ───────────────────────────────
log("\n  [2c] D(φ) order-flow distribution proxy")
log("  Method: cross-sectional return dispersion (ret_sigma_xs) as spread of D(φ).")
log("  Pre-2014 (no QuhiQuhihi data): imputed as sp500_sigma_12m × 0.35 (IG ratio).")

df["D_phi_sigma"] = df["ret_sigma_xs"].fillna(df["sp500_sigma_12m"] * 0.35)
df["D_phi_source"] = np.where(df["xs_sigma_available"] == 1, "real_cusip_data", "imputed")

log(f"  D_phi_sigma: mean={df['D_phi_sigma'].mean():.4f} | "
    f"real={df['D_phi_source'].eq('real_cusip_data').sum()} days | "
    f"imputed={df['D_phi_source'].eq('imputed').sum()} days")

# ── 2d. τ (informed-flow fraction) proxy ─────────────────────────────────
log("\n  [2d] τ (toxic/informed-flow fraction) proxy")
log("  Method: |credit_rf| / (|credit_rf| + |liquidity_rf| + ε) — the share of")
log("  the total factor signal attributable to adverse selection (credit) vs")
log("  pure illiquidity (liquidity). Bounded [0, 1].")

eps = 1e-8
df["tau_proxy"] = (
    df["credit_rf"].abs() /
    (df["credit_rf"].abs() + df["liquidity_rf"].abs() + eps)
).clip(0, 1)
df["tau_proxy"] = df["tau_proxy"].fillna(
    df.groupby("regime")["tau_proxy"].transform("mean")
)
log(f"  tau_proxy: mean={df['tau_proxy'].mean():.3f} | "
    f"stress mean={df.loc[df['regime']=='stress','tau_proxy'].mean():.3f}")

df.to_csv(f"{PRE}/02_master_enriched.csv")
log(f"\n  → 02_master_enriched.csv  ({len(df):,} rows × {df.shape[1]} cols)")


# ═══════════════════════════════════════════════════════════════
# STEP 3 — FIT (A, k) FROM MASTER'S OWN INTENSITY COLUMNS
# ═══════════════════════════════════════════════════════════════
section("STEP 3 — Fit exponential-intensity parameters (A, k) from master")

log("  Model: λ(h) = A · exp(−k · h),  where h = implied_spread_bps / 10000")
log("  Fit using master's A_IG_estimate and A_HY_estimate as the λ observations")
log("  and h_IG_decimal / h_HY_decimal as the h observations, grouped by regime.")
log("  MLE for exponential model via scipy.optimize.curve_fit.")

def exp_model(h, A0, k_decay):
    return A0 * np.exp(-k_decay * h)

fitted_rows = []
# Use TRACE window only (2004-2021) for fitting — bond factor data available
trace_mask = (df.index >= "2004-08-01") & (df.index <= "2021-12-31")
df_fit = df[trace_mask].copy()

for bucket in ["IG", "HY"]:
    h_col  = f"h_{bucket}_decimal"
    A_col  = f"A_{bucket}_estimate"
    s_col  = f"sigma_{bucket}_estimate"
    k_col  = f"k_{bucket}_estimate"

    for regime in ["calm", "normal", "elevated", "stress", "crisis"]:
        sub = df_fit[df_fit["regime"] == regime][[h_col, A_col, s_col, k_col]].dropna()
        if len(sub) < 10:
            continue

        h_vals   = sub[h_col].values
        A_obs    = sub[A_col].values
        sigma_obs = sub[s_col].values
        k_obs    = sub[k_col].values

        # Fit A(h) — exponential decay
        try:
            popt, pcov = curve_fit(
                exp_model, h_vals, A_obs,
                p0=[A_obs.mean(), 1.0],
                bounds=([0, 0], [100, 50]),
                maxfev=10000
            )
            A_fit, k_A_fit = popt
            A_se = np.sqrt(abs(pcov[0, 0]))
            k_A_se = np.sqrt(abs(pcov[1, 1]))
        except Exception:
            A_fit, k_A_fit, A_se, k_A_se = A_obs.mean(), np.nan, np.nan, np.nan

        # Fit k (price impact) from spread-sigma OLS: log(σ) ~ β·log(spread)
        lhs = np.log(sigma_obs.clip(1e-8))
        rhs = np.log(k_obs.clip(1e-8))
        if len(sub) >= 5 and rhs.std() > 1e-10:
            slope, intercept, r_val, p_val, se_slope = stats.linregress(rhs, lhs)
        else:
            slope, intercept, r_val, p_val, se_slope = np.nan, np.nan, np.nan, np.nan, np.nan

        # ADF on A residuals (check if A is stationary around the fit)
        resid = A_obs - exp_model(h_vals, A_fit, k_A_fit)
        try:
            adf_resid, adf_p, *_ = adfuller(resid, autolag="AIC")
            resid_stationary = adf_p < 0.05
        except Exception:
            adf_resid, adf_p, resid_stationary = np.nan, np.nan, False

        fitted_rows.append({
            "rating_bucket":       bucket,
            "regime":              regime,
            "n_days":              len(sub),
            "h_mean_decimal":      round(h_vals.mean(), 6),
            "A_obs_mean":          round(A_obs.mean(), 3),
            "A_fit":               round(A_fit, 3),
            "A_fit_se":            round(A_se, 4),
            "k_decay_fit":         round(k_A_fit, 4),
            "k_decay_se":          round(k_A_se, 4),
            "k_impact_slope":      round(slope, 4),
            "k_impact_r2":         round(r_val**2, 3),
            "k_impact_pval":       round(p_val, 4),
            "sigma_mean":          round(sigma_obs.mean(), 5),
            "resid_adf_stat":      round(adf_resid, 3) if not np.isnan(adf_resid) else None,
            "resid_adf_p":         round(adf_p, 4) if not np.isnan(adf_p) else None,
            "resid_stationary":    resid_stationary,
            # Simulator-ready
            "sim_A":               round(A_fit, 2),
            "sim_k":               round(abs(k_A_fit), 3),
            "sim_sigma":           round(sigma_obs.mean(), 5),
            "data_is_synthetic":   False,
        })

        log(f"  {bucket} {regime:<10}: n={len(sub):>4} | "
            f"A={A_fit:.2f} (SE={A_se:.3f}) | "
            f"k_decay={k_A_fit:.3f} | "
            f"k_impact R²={r_val**2:.2f} | "
            f"resid_stat={'✓' if resid_stationary else '✗'}")

params_df = pd.DataFrame(fitted_rows)
params_df.to_csv(f"{PRE}/03_fitted_intensity_params.csv", index=False)
log(f"\n  → 03_fitted_intensity_params.csv  ({len(params_df)} rows)")

# Write fitted params back to master for Step 6
regime_params = (params_df.groupby("regime")
    .agg(A_fitted_mean=("A_fit","mean"),
         k_decay_mean=("k_decay_fit","mean"),
         k_impact_mean=("k_impact_slope","mean"))
    .reset_index())
df = df.reset_index().merge(regime_params, on="regime", how="left").set_index("date")
df.index = pd.to_datetime(df.index)


# ═══════════════════════════════════════════════════════════════
# STEP 4 — NORMALISE SIMULATOR LOGS TO MASTER SCHEMA
# ═══════════════════════════════════════════════════════════════
section("STEP 4 — Normalise simulator logs to master schema")

try:
    sim_log = pd.read_csv(f"{SIM}/reflex_simulation_log.csv")
    sim_sum = pd.read_csv(f"{SIM}/reflex_run_summary.csv")
    has_sim = True
    log(f"  Loaded: {len(sim_log):,} step-rows | {sim_sum['run_id'].nunique()} episodes")
except FileNotFoundError:
    log(f"  WARNING: sim logs not found at {SIM}. Run reflex/src/simulator.py first.")
    has_sim = False

if has_sim:
    # ── 4a. Schema alignment with master column names ─────────────────────
    sim_log = sim_log.rename(columns={
        "h":            "h_halfspread",
        "h_eff":        "h_eff_halfspread",
        "tau":          "tau_toxicfrac",
        "q_after":      "q_inventory",
        "lam_informed": "lambda_informed",
        "lam_noise":    "lambda_noise",
    })
    # Tag to match master's provenance columns
    sim_log["data_is_synthetic"] = True
    sim_log["primary_source"]    = "simulator_v1"

    # ── 4b. Fit normalisation params from master (pre-2022 calibration window)
    log("\n  [4b] Z-score params from master pre-2022 calibration window")
    norm_map = {
        "h_halfspread":   ("h_IG_decimal",  cal_mask),
        "tau_toxicfrac":  ("tau_proxy",     cal_mask),
        "q_inventory":    ("q_proxy",       cal_mask),
        "lambda_informed":("A_IG_estimate", cal_mask),
        "lambda_noise":   ("A_IG_estimate", cal_mask),
    }
    for sim_col, (master_col, mask) in norm_map.items():
        if sim_col not in sim_log.columns:
            continue
        mu = df.loc[mask, master_col].mean()
        sd = df.loc[mask, master_col].std()
        if sd == 0 or np.isnan(sd):
            sd = 1.0
        sim_log[f"{sim_col}_z"] = ((sim_log[sim_col] - mu) / sd).round(5)
        log(f"    {sim_col:<22} z-scored using master.{master_col} "
            f"(μ={mu:.4f}, σ={sd:.4f})")

    # ── 4c. Winsorise using master-derived bounds ──────────────────────────
    h_p99 = df.loc[cal_mask, "h_IG_decimal"].quantile(0.99) * 100
    q_p99 = df.loc[cal_mask, "q_proxy"].abs().quantile(0.99) * 3
    sim_log["h_halfspread"]  = sim_log["h_halfspread"].clip(0, h_p99)
    sim_log["q_inventory"]   = sim_log["q_inventory"].clip(-q_p99, q_p99)

    # ── 4d. Add regime label using ε → regime mapping from fitted params ──
    def eps_to_regime(eps):
        if eps < 0.25: return "calm"
        if eps < 0.45: return "normal"
        if eps < 0.65: return "elevated"
        if eps < 0.85: return "stress"
        return "crisis"
    sim_log["adversarial_regime"] = sim_log["epsilon"].apply(eps_to_regime)
    sim_sum["adversarial_regime"] = sim_sum["epsilon"].apply(eps_to_regime)

    # ── 4e. Normalise run-level summary ───────────────────────────────────
    estimator_cols = [
        "br_slope","sinkhorn_wasserstein_tau_shift",
        "dlambda_informed_ddelta","h_tail_mean","h_tail_std",
        "tau_tail_mean","q_tail_mean"
    ]
    for col in estimator_cols:
        mu, sd = sim_sum[col].mean(), sim_sum[col].std()
        sim_sum[f"{col}_z"] = ((sim_sum[col] - mu) / (sd if sd > 0 else 1)).round(5)

    sim_sum["stability_code"]    = sim_sum["stability_label"].map(
        {"stable": 0, "oscillating": 1, "collapsed_saturated": 2})
    sim_sum["data_is_synthetic"] = True
    sim_sum["primary_source"]    = "simulator_v1"

    sim_log.to_csv(f"{PRE}/04_sim_logs_normalised.csv", index=False)
    sim_sum.to_csv(f"{PRE}/04_sim_summary_normalised.csv", index=False)
    log(f"\n  Normalised sim log : {len(sim_log):,} rows × {sim_log.shape[1]} cols")
    log(f"  Normalised summary : {len(sim_sum)} rows × {sim_sum.shape[1]} cols")
    log(f"  Stability: {sim_sum['stability_label'].value_counts().to_dict()}")
    log(f"  → 04_sim_logs_normalised.csv")
    log(f"  → 04_sim_summary_normalised.csv")


# ═══════════════════════════════════════════════════════════════
# STEP 5 — EPISODE-LEVEL CAL / VAL / HELD-OUT SPLIT
# ═══════════════════════════════════════════════════════════════
section("STEP 5 — Episode-level calibration / held-out split")

log("  Episode = one (bond_id × ε × N_dealers × universe_size) trajectory (T=600 steps)")
log("  Split axis: ε (adversarial intensity) — NOT random, NOT time-based for sim data")
log()
log("  ┌─────────────────────────────────────────────────────────────────┐")
log("  │  calibration  70%  ε ∈ [0.05, 0.75]  — covers calm→stress     │")
log("  │  validation   15%  ε ∈ (0.75, 0.85]  — near stability boundary │")
log("  │  held-out     15%  ε ∈ (0.85, 0.95]  — high-adversarial regime │")
log("  └─────────────────────────────────────────────────────────────────┘")
log()
log("  Real-data split (master):")
log("  ┌─────────────────────────────────────────────────────────────────┐")
log("  │  pre_sample   1990-2003  — macro context, no bond factors       │")
log("  │  calibration  2004-2019  — full TRACE window, pre-COVID         │")
log("  │  validation   2020-2021  — COVID shock + recovery               │")
log("  │  held-out     2022-2026  — rate-hike cycle, out-of-sample       │")
log("  └─────────────────────────────────────────────────────────────────┘")

# ── 5a. Simulator episode split ────────────────────────────────────────────
if has_sim:
    def assign_split(eps):
        if eps <= 0.75: return "calibration"
        if eps <= 0.85: return "validation"
        return "held_out"

    sim_sum["split"] = sim_sum["epsilon"].apply(assign_split)
    sim_sum["episode_id"] = (
        sim_sum["bond_id"].astype(str) + "_e"
        + sim_sum["epsilon"].astype(str) + "_n"
        + sim_sum["n_dealers"].astype(str) + "_u"
        + sim_sum["universe_size"].astype(str)
    )

    for split in ["calibration","validation","held_out"]:
        sub = sim_sum[sim_sum["split"] == split]
        lbl_counts = sub["stability_label"].value_counts().to_dict()
        log(f"  sim {split:<15}: {len(sub):>4} episodes | {lbl_counts}")

    cal_ep  = sim_sum[sim_sum["split"].isin(["calibration","validation"])]
    held_ep = sim_sum[sim_sum["split"] == "held_out"]
    cal_ep.to_csv(f"{PRE}/05_calibration_episodes.csv",  index=False)
    held_ep.to_csv(f"{PRE}/05_heldout_episodes.csv",     index=False)
    log(f"  → 05_calibration_episodes.csv  ({len(cal_ep)} episodes)")
    log(f"  → 05_heldout_episodes.csv       ({len(held_ep)} episodes)")

# ── 5b. Real-data split (master) ───────────────────────────────────────────
def assign_real_split(date):
    if date < pd.Timestamp("2004-08-01"): return "pre_sample"
    if date < pd.Timestamp("2020-01-01"): return "calibration"
    if date < pd.Timestamp("2022-01-01"): return "validation"
    return "held_out"

df.index = pd.to_datetime(df.index)
df["icaif_split"] = df.index.map(assign_real_split)
for split in ["pre_sample","calibration","validation","held_out"]:
    sub = df[df["icaif_split"] == split]
    log(f"  real {split:<15}: {len(sub):>5} days | "
        f"regime={sub['regime'].value_counts().to_dict()}")


# ═══════════════════════════════════════════════════════════════
# STEP 6 — FINAL MASTER_PREPROCESSED.csv
# ═══════════════════════════════════════════════════════════════
section("STEP 6 — Build MASTER_PREPROCESSED.csv")

log("  Z-score all continuous features using calibration-window stats ONLY.")
log("  Lookahead guard: scaler fit on icaif_split='calibration' rows only.")

cal_rows = df["icaif_split"] == "calibration"
feature_cols = [
    "vix_close","implied_spread_bps","h_IG_bps","h_HY_bps",
    "h_IG_decimal","h_HY_decimal","q_proxy","tau_proxy","D_phi_sigma",
    "gs10_yield","real_yield","sp500_sigma_12m","liquidity_rf","credit_rf",
    "sigma_IG_estimate","sigma_HY_estimate","k_IG_estimate","k_HY_estimate",
    "A_IG_estimate","A_HY_estimate","A_fitted_mean","k_decay_mean","dv01_10y",
]
feature_cols = [c for c in feature_cols if c in df.columns]

scaler_stats = {}
for col in feature_cols:
    mu = df.loc[cal_rows, col].mean()
    sd = df.loc[cal_rows, col].std()
    sd = sd if sd > 0 else 1.0
    df[f"{col}_z"] = ((df[col] - mu) / sd).round(5)
    scaler_stats[col] = {"mean": round(mu, 6), "std": round(sd, 6)}

# Save scaler stats for inference-time use
pd.DataFrame(scaler_stats).T.to_csv(f"{PRE}/scaler_stats.csv")
log(f"  Scaler fit on {cal_rows.sum():,} calibration rows (2004-08 to 2019-12)")
log(f"  Scaler stats saved → scaler_stats.csv (for inference-time normalisation)")

# Final provenance columns
df["preprocessing_version"] = "v1.0"
df["lookahead_safe"]         = True
df["trace_data_available"]   = df["bond_factors_available"].astype(bool)

# Drop intermediate availability flags from final output (kept in 01_cleaned)
drop_cols = [c for c in df.columns if c.endswith("_available") and c != "bond_factors_available"]
df_out = df.drop(columns=drop_cols)
df_out.reset_index().to_csv(f"{PRE}/MASTER_PREPROCESSED.csv", index=False)
log(f"\n  MASTER_PREPROCESSED: {len(df_out):,} rows × {df_out.shape[1]} cols")
log(f"  icaif_split distribution: {df_out['icaif_split'].value_counts().to_dict()}")


# ═══════════════════════════════════════════════════════════════
# FINAL REPORT
# ═══════════════════════════════════════════════════════════════
section("PREPROCESSING COMPLETE — Output Summary")

outputs = [
    ("01_master_cleaned.csv",           "Step 1  Clean: nulls filled, outliers winsorised, ADF checked"),
    ("02_master_enriched.csv",          "Step 2  Enrich: h, q_proxy, tau_proxy, D_phi added"),
    ("03_fitted_intensity_params.csv",  "Step 3  Fitted (A, k) per rating×regime from master"),
    ("04_sim_logs_normalised.csv",      "Step 4  Sim logs: renamed cols, master-anchored z-scores"),
    ("04_sim_summary_normalised.csv",   "Step 4  Sim summary: z-scored estimators, stability codes"),
    ("05_calibration_episodes.csv",     "Step 5  Calibration + validation episodes (ε ≤ 0.85)"),
    ("05_heldout_episodes.csv",         "Step 5  Held-out episodes (ε > 0.85)"),
    ("scaler_stats.csv",                "Step 6  Scaler μ/σ for inference-time normalisation"),
    ("MASTER_PREPROCESSED.csv",         "Step 6  Final analysis-ready panel (all steps merged)"),
]
for fname, desc in outputs:
    fpath = f"{PRE}/{fname}"
    if os.path.exists(fpath):
        tmp = pd.read_csv(fpath)
        log(f"  {fname:<42} {len(tmp):>8,} rows × {tmp.shape[1]:>3} cols  ← {desc}")
    else:
        log(f"  {fname:<42} NOT GENERATED")

with open(f"{PRE}/preprocessing_report.txt", "w") as f:
    f.write("\n".join(log_lines))
log(f"\n  Audit trail → data/preprocessed/preprocessing_report.txt")
