#!/usr/bin/env bash
# run_rich_followup.sh — try to close the GNN's 50->61 gap two ways, on cache_v1_rich.pt:
#   (2) hybrid:  GNN-embedding -> boosting        (hybrid_cpm.py, seeds 1-5)
#   (4) push:    gconvgru tuned on rich           (run_zoo_cpm, lr.005 / gamma1, seeds 1-5)
# RAM-aware + resumable (skips logs that already have a RESULT). PAR cap.
set -u
cd "$(dirname "$0")"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate egcnoi_env
MAX_PAR="${1:-10}"; FLOOR_GB="${2:-10}"; PERJOB=5
OUT=rich_results; mkdir -p "$OUT"
CACHE=data/cache_v1_rich.pt

avail() { free -g | awk '/^Mem:/{print $7}'; }
nj()    { pgrep -f 'hybrid_cpm.py|run_zoo_cpm.py' | wc -l; }
slot()  { while :; do local n a; n=$(nj); a=$(avail); if (( n<MAX_PAR )) && (( a-PERJOB>=FLOOR_GB )); then return; fi; sleep 5; done; }

JOBS=()   # "command|logfile"
for s in 1 2 3 4 5; do
  JOBS+=("python hybrid_cpm.py --cache $CACHE --seed $s|$OUT/hybrid_s$s.log")
done
for s in 1 2 3 4 5; do
  JOBS+=("python run_zoo_cpm.py gconvgru $s --cache $CACHE --tag grich_lr005 --lr 0.005|$OUT/grich_lr005_s$s.log")
  JOBS+=("python run_zoo_cpm.py gconvgru $s --cache $CACHE --tag grich_g1 --gamma 1.0|$OUT/grich_g1_s$s.log")
done

echo "=== ${#JOBS[@]} jobs | MAX_PAR=$MAX_PAR FLOOR=${FLOOR_GB}GB ==="
for j in "${JOBS[@]}"; do
  cmd=${j%|*}; log=${j##*|}
  if grep -q '^RESULT' "$log" 2>/dev/null; then continue; fi
  slot
  $cmd > "$log" 2>&1 &
  sleep 1
done
wait
echo "=== followup done ==="
