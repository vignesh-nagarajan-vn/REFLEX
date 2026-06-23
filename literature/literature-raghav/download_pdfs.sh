#!/usr/bin/env bash
# download_pdfs.sh
# Fetches all papers (original 10 + 8 extensions) as PDFs into pdfs/
# Requires: curl, internet access to arxiv.org

set -euo pipefail

DEST="pdfs"
mkdir -p "$DEST"

declare -A PAPERS=(
  # ── Original 10 ──────────────────────────────────────────────────────────
  ["01_perdomo_performative_prediction"]="https://arxiv.org/pdf/2002.06673"
  ["02_mendler_dunner_stochastic_pp"]="https://arxiv.org/pdf/2006.06887"
  ["03_miller_echo_chamber"]="https://arxiv.org/pdf/2102.08570"
  ["04_izzo_performative_gradient_descent"]="https://arxiv.org/pdf/2102.07698"
  ["05_drusvyatskiy_xiao_decision_dependent"]="https://arxiv.org/pdf/2011.11173"
  ["06_jagadeesan_regret_performative"]="https://arxiv.org/pdf/2202.00628"
  ["07_li_wai_state_dependent_pp"]="https://arxiv.org/pdf/2110.00800"
  ["08_gueant_lehalle_inventory_risk"]="https://arxiv.org/pdf/1105.3115"
  ["09_bergault_gueant_size_matters"]="https://arxiv.org/pdf/1907.01225"
  ["10_barzykin_bergault_adverse_selection_2025"]="https://arxiv.org/pdf/2508.20225"

  # ── Extension 11–18 ───────────────────────────────────────────────────────
  ["11_brown_sandholm_performative_games"]="https://arxiv.org/pdf/2106.09784"
  ["12_narang_faulkner_multiplayer_pp"]="https://arxiv.org/pdf/2207.05630"
  ["13_cao_shi_distributionally_robust_pp"]="https://arxiv.org/pdf/2206.01844"
  ["14_cuturi_peyre_computational_ot"]="https://arxiv.org/pdf/1803.00567"
  ["15_avellaneda_stoikov_hft_mm"]="https://arxiv.org/pdf/1105.3115"
  ["16_cartea_jaimungal_penalva_almgren"]="https://arxiv.org/pdf/1204.4051"
  ["17_lacker_mean_field_games_finance"]="https://arxiv.org/pdf/1510.01408"
  ["18_cont_kukanov_stoikov_price_impact"]="https://arxiv.org/pdf/1011.6402"
)

for NAME in "${!PAPERS[@]}"; do
  URL="${PAPERS[$NAME]}"
  OUT="$DEST/${NAME}.pdf"
  if [[ -f "$OUT" ]]; then
    echo "  [skip] $NAME already exists"
  else
    echo "  [fetch] $NAME ..."
    curl -L --silent --show-error --retry 3 --retry-delay 2 \
         -o "$OUT" "$URL" && echo "  [ok]   $OUT" || echo "  [FAIL] $NAME"
    sleep 0.5   # be polite to arxiv
  fi
done

echo ""
echo "Done. PDFs are in $DEST/"
