#!/usr/bin/env bash
# CPM in the SAME harness as BuST expC — 7 modes x 10 seeds, vehicle-disjoint, paired.
cd ~/cpm
PY=~/miniconda3/envs/bust/bin/python
C=data/cache_v1_rich.pt
mkdir -p logs
JOBS=$(mktemp)
for s in 0 1 2 3 4 5 6 7 8 9; do
  for m in gnn_eng gnn_rel gnn_edge static_eng static_rel rf_eng rf_rel; do
    echo "$PY expC_cpm.py $C $m $s 40 24 > logs/cpm_${m}_s$s.log 2>&1" >> "$JOBS"
  done
done
N=$(wc -l < "$JOBS")
echo "[run_cpm] $N jobs, P=10, start $(date)" | tee logs/run_cpm.status
cat "$JOBS" | xargs -P 10 -I CMD bash -c CMD
echo "[run_cpm] DONE $(date)" | tee -a logs/run_cpm.status
{ echo "=== CPM RESULTS $(date) ==="; grep -hE "RESULT expCcpm" logs/cpm_*.log | sort; } > ~/cpm/RESULTS_cpm.txt
echo "[run_cpm] collected"
