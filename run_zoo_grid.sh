#!/usr/bin/env bash
# run_zoo_grid.sh — RAM-aware, resumable model-zoo sweep.
#
#   * RAM-aware : never launches a job unless `available` RAM stays >= FLOOR_GB
#                 afterwards; also caps concurrency at MAX_PAR.
#   * resumable : skips any (ds,model,seed) whose log already has a RESULT line.
#   * datasets  : v1 / v2 (all 11 models) + v1aug / v2aug (static,gconvgru,hybrid).
#
# Usage:  bash run_zoo_grid.sh [MAX_PAR] [FLOOR_GB]   (defaults 10, 10)
set -u
cd "$(dirname "$0")"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate egcnoi_env

MAX_PAR="${1:-10}"     # hard ceiling on concurrent jobs (kept low on purpose)
FLOOR_GB="${2:-10}"    # always keep at least this many GB 'available'
PERJOB_GB=5            # conservative per-job working-set estimate

OUT=zoo_results
mkdir -p "$OUT"
SUMMARY="$OUT/RESULTS.txt"
touch "$SUMMARY"

# dataset -> cache path
declare -A CACHE=(
  [v1]=data/cache_v1.pt
  [v2]=data/v2/cache_v2.pt
  [v1aug]=data/cache_v1_aug.pt
  [v2aug]=data/v2/cache_v2_aug.pt
)
ALL_MODELS="gconvgru tgcn static evolve_h evolve_h_shallow evolve_h_resid evolve_h_wide evolve_h_best evolve_h_jk evolve_o hybrid"
AUG_MODELS="static gconvgru hybrid"
SEEDS="1 2 3 4 5"

avail_gb() { free -g | awk '/^Mem:/{print $7}'; }
njobs()    { pgrep -f run_zoo_cpm.py | wc -l; }

# wait until both: concurrency < MAX_PAR  AND  available-PERJOB >= FLOOR
wait_for_slot() {
  while :; do
    local n a; n=$(njobs); a=$(avail_gb)
    if (( n < MAX_PAR )) && (( a - PERJOB_GB >= FLOOR_GB )); then return; fi
    sleep 5
  done
}

run_job() {
  local ds=$1 cache=$2 m=$3 s=$4
  local log="$OUT/${ds}_${m}_s${s}.log"
  if grep -q '^RESULT' "$log" 2>/dev/null; then          # resume: already done
    grep '^RESULT' "$log" | sed "s/^RESULT/RESULT ds=$ds/" >> "$SUMMARY"
    return
  fi
  python run_zoo_cpm.py "$m" "$s" --cache "$cache" --epochs 40 --window 24 \
        --hidden 32 --lr 0.01 --proj 32 > "$log" 2>&1
  grep '^RESULT' "$log" | sed "s/^RESULT/RESULT ds=$ds/" >> "$SUMMARY"
}

# build job list
JOBS=()
for ds in v1 v2; do
  for m in $ALL_MODELS; do for s in $SEEDS; do JOBS+=("$ds|${CACHE[$ds]}|$m|$s"); done; done
done
for ds in v1aug v2aug; do
  for m in $AUG_MODELS; do for s in $SEEDS; do JOBS+=("$ds|${CACHE[$ds]}|$m|$s"); done; done
done
echo "=== ${#JOBS[@]} jobs | MAX_PAR=$MAX_PAR FLOOR=${FLOOR_GB}GB ==="

for spec in "${JOBS[@]}"; do
  IFS='|' read -r ds cache m s <<< "$spec"
  log="$OUT/${ds}_${m}_s${s}.log"
  if grep -q '^RESULT' "$log" 2>/dev/null; then
    grep '^RESULT' "$log" | sed "s/^RESULT/RESULT ds=$ds/" >> "$SUMMARY"
    continue                                              # skip done without a slot
  fi
  wait_for_slot
  run_job "$ds" "$cache" "$m" "$s" &
  sleep 1   # stagger launches so RAM/njobs reads settle
done
wait
# de-dup summary (resume may append duplicates)
sort -u "$SUMMARY" -o "$SUMMARY"
echo "=== done. $(grep -c '^RESULT' "$SUMMARY") results in $SUMMARY ==="
