#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate egcnoi_env
cd ~/EvolveGCNOi_CPM

for model in gconvgru evolve; do
  for seed in 1 2 3 4 5; do
    echo "$(date) === Starting $model seed=$seed ==="
    python macro_cpm.py $model $seed > macro_${model}_s${seed}.log 2>&1
    echo "$(date) === Done $model seed=$seed ==="
  done
done
echo "ALL DONE"
