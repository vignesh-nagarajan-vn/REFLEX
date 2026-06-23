#!/usr/bin/env bash
# Download the ten open-access arXiv PDFs for the endo_market literature folder.
#
# Usage:   ./download_pdfs.sh
# Output:  ./pdfs/<id>__<short-name>.pdf
#
# Requires outbound access to arxiv.org. If you are running inside a sandbox with
# an egress allowlist, add `arxiv.org` (and optionally `export.arxiv.org`) to the
# allowed domains first. The script falls back to the export mirror automatically.

set -euo pipefail
cd "$(dirname "$0")"
mkdir -p pdfs

# id|short-name
papers=(
  "2002.06673|perdomo-performative-prediction"
  "2006.06887|mendler-dunner-stochastic-pp"
  "2102.08570|miller-outside-echo-chamber"
  "2102.07698|izzo-performative-gradient-descent"
  "2011.11173|drusvyatskiy-xiao-decision-dependent"
  "2202.00628|jagadeesan-performative-feedback"
  "2110.00800|li-wai-state-dependent-pp"
  "1105.3115|gueant-lehalle-inventory-risk"
  "1907.01225|bergault-gueant-size-matters-otc"
  "2508.20225|barzykin-adverse-selection-price-reading"
)

ok=0; fail=0
for entry in "${papers[@]}"; do
  id="${entry%%|*}"
  name="${entry##*|}"
  out="pdfs/${id}__${name}.pdf"

  if [[ -s "$out" ]]; then
    echo "skip  $out (already present)"
    ok=$((ok+1)); continue
  fi

  echo "fetch ${id} -> ${out}"
  if curl -fsSL --retry 3 --retry-delay 2 -o "$out" "https://arxiv.org/pdf/${id}" \
     || curl -fsSL --retry 3 --retry-delay 2 -o "$out" "https://export.arxiv.org/pdf/${id}"; then
    # sanity-check it is actually a PDF, not an HTML/error page
    if head -c 5 "$out" | grep -q "%PDF"; then
      ok=$((ok+1))
    else
      echo "  !! not a PDF (got an error/HTML page) — removing"
      rm -f "$out"; fail=$((fail+1))
    fi
  else
    echo "  !! download failed for ${id}"
    rm -f "$out" 2>/dev/null || true
    fail=$((fail+1))
  fi
done

echo
echo "done: ${ok} ok, ${fail} failed -> ./pdfs/"
[[ $fail -eq 0 ]] || { echo "Some downloads failed. If you are sandboxed, allow arxiv.org egress and re-run."; exit 1; }
