"""
fetch_data.py — Download all REFLEX raw data sources from scratch.

All sources are freely available with no API key required.
Downloads via GitHub codeload zip archives (confirmed accessible).

Usage:
    python src/fetch_data.py
"""

import urllib.request
import zipfile
import io
import os
import sys

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
os.makedirs(RAW_DIR, exist_ok=True)

# ── Source registry ───────────────────────────────────────────────────────────
# Format: (github_repo, branch, description, licence_note)
SOURCES = [
    (
        "datasets/finance-vix", "main",
        "CBOE VIX daily 1990-2026",
        "Public domain — CBOE historical VIX data"
    ),
    (
        "datasets/bond-yields-us-10y", "main",
        "US 10-Year Treasury yield monthly 1953-2026 (Fed H.15)",
        "Public domain — US government publication"
    ),
    (
        "datasets/s-and-p-500", "main",
        "Shiller S&P500 + GS10 + CPI + PE10 monthly 1871-2023",
        "Freely redistributable academic dataset (Shiller/Yale)"
    ),
    (
        "datasets/oil-prices", "main",
        "EIA WTI crude oil daily 1986-2026",
        "Public domain — US government publication"
    ),
    (
        "datasets/gold-prices", "main",
        "Gold spot price monthly 1833-2026 (World Gold Council)",
        "Freely shared historical series"
    ),
    (
        "datasets/cpi-us", "main",
        "US CPI-U monthly 1913-2026 (BLS)",
        "Public domain — US government publication"
    ),
    (
        "Alexander-M-Dickerson/TRACE-corporate-bond-processing", "main",
        "TRACE-derived corporate bond factors monthly 2004-2021",
        "Academic replication data — Dickerson, Mueller & Robotti (2023) JFE"
    ),
    (
        "QuhiQuhihi/Factor-Strategy-for-Corporate-Bond-", "main",
        "Real CUSIP-level corporate bond returns monthly 2014-2023",
        "Freely redistributable academic/student project"
    ),
]

SOURCE_LINKS = {
    "finance-vix":                                  "https://github.com/datasets/finance-vix",
    "bond-yields-us-10y":                           "https://github.com/datasets/bond-yields-us-10y",
    "s-and-p-500":                                  "https://github.com/datasets/s-and-p-500",
    "oil-prices":                                   "https://github.com/datasets/oil-prices",
    "gold-prices":                                  "https://github.com/datasets/gold-prices",
    "cpi-us":                                       "https://github.com/datasets/cpi-us",
    "TRACE-corporate-bond-processing":              "https://github.com/Alexander-M-Dickerson/TRACE-corporate-bond-processing",
    "Factor-Strategy-for-Corporate-Bond-":          "https://github.com/QuhiQuhihi/Factor-Strategy-for-Corporate-Bond-",
}

# Files to SKIP even if present in the zip (known synthetic or unusable files)
SKIP_FILES = {
    # Liquidity-Scoring files (all synthetic — see docs/REJECTED_SOURCES.md)
    "trace_data.parquet", "features.parquet", "impact_params.parquet",
    "macro_factors.parquet", "bond_universe.csv",
    # QuhiQuhihi portfolio weight files (not return data — see REJECTED_SOURCES.md)
    "weights_1n_df.csv", "weights_mv_df.csv", "Backtest_Check.csv",
    "cumulative_returns_all.csv",
}


def download_repo(repo: str, branch: str, description: str, licence: str) -> list[str]:
    """Download a GitHub repo as a zip and extract CSV files to RAW_DIR."""
    url = f"https://codeload.github.com/{repo}/zip/refs/heads/{branch}"
    repo_name = repo.split("/")[-1]
    print(f"\n  Downloading: {repo_name}")
    print(f"    {description}")
    print(f"    Licence: {licence}")
    print(f"    URL: {SOURCE_LINKS.get(repo_name, url)}")

    try:
        response = urllib.request.urlopen(url, timeout=30)
        data = response.read()
        zf = zipfile.ZipFile(io.BytesIO(data))
        extracted = []
        for name in zf.namelist():
            fname = name.split("/")[-1]
            if not fname.endswith(".csv"):
                continue
            if fname in SKIP_FILES:
                print(f"    SKIP (rejected source): {fname}")
                continue
            out_path = os.path.join(RAW_DIR, f"{repo_name}__{fname}")
            content = zf.read(name)
            with open(out_path, "wb") as f:
                f.write(content)
            extracted.append(fname)
            print(f"    + {fname} ({len(content):,} bytes)")
        return extracted
    except Exception as e:
        print(f"    FAILED: {e}", file=sys.stderr)
        return []


def main():
    print("=" * 60)
    print("REFLEX Data Fetcher")
    print("=" * 60)
    print(f"Saving to: {os.path.abspath(RAW_DIR)}")
    print()
    print("Sources to download:")
    for repo, branch, desc, _ in SOURCES:
        rname = repo.split("/")[-1]
        print(f"  {rname}: {SOURCE_LINKS.get(rname, '')}")

    ok, failed = 0, []
    for repo, branch, desc, licence in SOURCES:
        files = download_repo(repo, branch, desc, licence)
        if files:
            ok += 1
        else:
            failed.append(repo)

    print(f"\n{'='*60}")
    print(f"Downloaded {ok}/{len(SOURCES)} repos successfully.")
    if failed:
        print(f"Failed: {failed}")
    print(f"\nNext: python src/build_datasets.py")
    print(f"Then: python src/verify_data.py")


if __name__ == "__main__":
    main()
