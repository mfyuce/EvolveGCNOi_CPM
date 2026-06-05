#!/usr/bin/env bash
# run_zoo_grid.sh — full model-zoo sweep on maya, RAM-aware parallelism.
# Builds .pt caches for v1 + v2 first (one full JSON load each), then runs
# every model × dataset × seed against the cheap caches.
#
# Usage:  bash run_zoo_grid.sh [PARALLEL]   (default PARALLEL=6)
set -u
cd "$(dirname "$0")"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate egcnoi_env

PAR="${1:-24}"   # maya has 64 cores; each run is single-threaded + ~0.5 GB via .pt cache
OUT=zoo_results
mkdir -p "$OUT"
SUMMARY="$OUT/RESULTS.txt"
: > "$SUMMARY"

V2_JSON=data/v2/burst_adma_with_cpm_multi_sensors999_cls_all_v2.json
CACHE_V1=data/cache_v1.pt
CACHE_V2=data/v2/cache_v2.pt

echo "=== building caches ==="
[ -f "$CACHE_V1" ] || python prep_cache.py --out "$CACHE_V1"
if [ ! -f "$V2_JSON" ] && [ -f data/v2/*.zip ]; then
  echo "unzipping v2 ..."; (cd data/v2 && unzip -o *.zip >/dev/null)
fi
[ -f "$CACHE_V2" ] || python prep_cache.py --dataset "$V2_JSON" --out "$CACHE_V2"

MODELS="gconvgru tgcn static evolve_h evolve_h_shallow evolve_h_resid evolve_h_wide evolve_h_best evolve_h_jk evolve_o hybrid"
SEEDS="1 2 3 4 5"

# build the job list: dataset|cache|model|seed
JOBS=()
for ds in v1 v2; do
  cache=$CACHE_V1; [ "$ds" = v2 ] && cache=$CACHE_V2
  for m in $MODELS; do
    for s in $SEEDS; do
      JOBS+=("$ds|$cache|$m|$s")
    done
  done
done
echo "=== ${#JOBS[@]} jobs, parallelism=$PAR ==="

run_job() {
  local spec="$1"
  IFS='|' read -r ds cache m s <<< "$spec"
  local log="$OUT/${ds}_${m}_s${s}.log"
  python run_zoo_cpm.py "$m" "$s" --cache "$cache" --epochs 40 --window 24 \
        --hidden 32 --lr 0.01 --proj 32 --tag "${ds}" > "$log" 2>&1
  grep '^RESULT' "$log" | sed "s/^RESULT/RESULT ds=$ds/" >> "$SUMMARY"
}
export -f run_job
export OUT

# throttle: keep ~PAR jobs running at once (wait -n needs bash 4.3+)
running=0
for spec in "${JOBS[@]}"; do
  run_job "$spec" &
  running=$((running+1))
  if (( running >= PAR )); then
    wait -n
    running=$((running-1))
  fi
done
wait
echo "=== done. summary: $SUMMARY ==="
sort "$SUMMARY"
