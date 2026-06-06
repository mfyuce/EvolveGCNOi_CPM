#!/usr/bin/env bash
# run_hp_search.sh — RAM-aware, resumable hyperparameter search for the CPM
# winners (gconvgru, hybrid) on the v2aug cache. Coordinate search around the
# proven default (H32 W24 lr0.01 do0.5 g2.0 ep40), which already scores
# gconvgru 48.9 / hybrid 43.8 on v2aug (existing zoo_results) — the anchor is
# NOT re-run here. Each config gets a distinct ds= label so aggregate_results.py
# separates it into its own mean±std block.
#
# Usage:  bash run_hp_search.sh [MAX_PAR] [FLOOR_GB]   (defaults 10, 10)
set -u
cd "$(dirname "$0")"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate egcnoi_env

MAX_PAR="${1:-10}"     # hard ceiling on concurrent jobs (kept low on purpose)
FLOOR_GB="${2:-10}"    # always keep at least this many GB 'available'
PERJOB_GB=5            # conservative per-job working-set estimate
CACHE=data/v2/cache_v2_aug.pt

OUT=hp_results
mkdir -p "$OUT"
SUMMARY="$OUT/RESULTS.txt"
touch "$SUMMARY"
SEEDS="1 2 3 4 5"

# label | model | extra flags  (default for all: --epochs 40 --window 24 --hidden 32 --lr 0.01 --gamma 2.0 --dropout 0.5)
CONFIGS=(
  "g_h64|gconvgru|--hidden 64"
  "g_w48|gconvgru|--window 48"
  "g_lr005|gconvgru|--lr 0.005"
  "g_g1|gconvgru|--gamma 1.0"
  "g_h64w48|gconvgru|--hidden 64 --window 48"
  "g_do3|gconvgru|--dropout 0.3"
  "h_proj64|hybrid|--proj 64"
  "h_lr005|hybrid|--lr 0.005"
  "h_w48|hybrid|--window 48"
  "h_h64|hybrid|--hidden 64"
)

avail_gb() { free -g | awk '/^Mem:/{print $7}'; }
njobs()    { pgrep -f run_zoo_cpm.py | wc -l; }
wait_for_slot() {
  while :; do
    local n a; n=$(njobs); a=$(avail_gb)
    if (( n < MAX_PAR )) && (( a - PERJOB_GB >= FLOOR_GB )); then return; fi
    sleep 5
  done
}
run_job() {
  local label=$1 model=$2 flags=$3 s=$4
  local log="$OUT/${label}_s${s}.log"
  if grep -q '^RESULT' "$log" 2>/dev/null; then            # resume: already done
    grep '^RESULT' "$log" | sed "s/^RESULT/RESULT ds=$label/" >> "$SUMMARY"
    return
  fi
  python run_zoo_cpm.py "$model" "$s" --cache "$CACHE" --tag "$label" $flags \
        > "$log" 2>&1
  grep '^RESULT' "$log" | sed "s/^RESULT/RESULT ds=$label/" >> "$SUMMARY"
}

N=$(( ${#CONFIGS[@]} * 5 ))
echo "=== hp-search: ${#CONFIGS[@]} configs x 5 seeds = $N jobs | PAR=$MAX_PAR FLOOR=${FLOOR_GB}GB ==="

for spec in "${CONFIGS[@]}"; do
  IFS='|' read -r label model flags <<< "$spec"
  for s in $SEEDS; do
    log="$OUT/${label}_s${s}.log"
    if grep -q '^RESULT' "$log" 2>/dev/null; then
      grep '^RESULT' "$log" | sed "s/^RESULT/RESULT ds=$label/" >> "$SUMMARY"
      continue                                              # skip done without a slot
    fi
    wait_for_slot
    run_job "$label" "$model" "$flags" "$s" &
    sleep 1   # stagger launches so RAM/njobs reads settle
  done
done
wait
sort -u "$SUMMARY" -o "$SUMMARY"
echo "=== hp-search done. $(grep -c '^RESULT' "$SUMMARY") results in $SUMMARY ==="
