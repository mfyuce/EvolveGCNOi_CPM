#!/usr/bin/env bash
# CPM local_preserving — 10 seeds, vehicle-disjoint; SAME harness as gnn_edge/rf_rel
# (seed -> split is deterministic, so these rows are paired with the existing CPM results).
cd ~/cpm
PY=~/miniconda3/envs/bust/bin/python
C=data/cache_v1_rich.pt
mkdir -p logs
JOBS=$(mktemp)
for s in 0 1 2 3 4 5 6 7 8 9; do
  echo "$PY expC_cpm.py $C local_preserving $s 40 24 > logs/cpm_local_preserving_s$s.log 2>&1" >> "$JOBS"
done
echo "[run_lp] 10 jobs, P=10, start $(date)" | tee logs/run_lp.status
cat "$JOBS" | xargs -P 10 -I CMD bash -c CMD
echo "[run_lp] DONE $(date)" | tee -a logs/run_lp.status
{ grep -hE "RESULT expCcpm" logs/cpm_local_preserving_s*.log | sort; } > ~/cpm/RESULTS_lp_cpm.txt
echo "[run_lp] collected -> RESULTS_lp_cpm.txt" | tee -a logs/run_lp.status
