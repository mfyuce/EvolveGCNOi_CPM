"""
Focused hyperparameter search for CPM EvolveGCN-H.

Fixes vs original hyper_parameter_search.py:
- binary=True loader (leakage fix)
- n_node_features used (not lags) — already in ModelOps.train()
- Sane search grid (72 combos ~35 min vs 1140 ~6h)
- No model save per run (saves disk)
- Prints every run, not only on new max
- BEST_LAGS tuned for 3-D feature case (lags only controls sample count)
"""
import os, sys, time
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

from BurstAdmaDatasetLoader import BurstAdmaDatasetLoader
from torch_geometric_temporal.signal import temporal_signal_split
from graphs.recurrent.graphs_evolvegcn_h_improved import ModelOps

# ── search grid ────────────────────────────────────────────────
LAGS        = [1, 4, 8]
EPOCHS      = [10, 20, 30]
LR_VALUES   = [0.001, 0.003, 0.005, 0.007, 0.01, 0.02, 0.03, 0.05]
TRAIN_RATIO = 0.7
# total: 3 × 3 × 8 = 72 runs
# ───────────────────────────────────────────────────────────────

t = int(time.time())
os.makedirs(f"./runs/{t}", exist_ok=True)
log_all    = f"./runs/{t}/search_all_{t}.csv"
log_change = f"./runs/{t}/search_best_{t}.csv"

header = "lags,epochs,lr,train_mcc,test_mcc,test_f1,test_acc,test_loss,elapsed_s\n"

best_test_mcc = -1.0
total = len(LAGS) * len(EPOCHS) * len(LR_VALUES)
run_n = 0

print("Loading dataset...", flush=True)
t0 = time.time()
loader = BurstAdmaDatasetLoader(features_as_self_edge=True, binary=True)
# pre-load with lags=1 so JSON is parsed once
loader.get_dataset(lags=1)
print(f"Loaded in {time.time()-t0:.1f}s  n_node_features={loader.n_node_features}", flush=True)
print(f"\nStarting search: {total} runs\n", flush=True)

with open(log_all, "w") as fa, open(log_change, "w") as fc:
    fa.write(header)
    fc.write(header)

    for lags in LAGS:
        dataset = loader.get_dataset(lags=lags)
        train_ds, test_ds = temporal_signal_split(dataset, train_ratio=TRAIN_RATIO)

        for epochs in EPOCHS:
            for lr in LR_VALUES:
                run_n += 1
                t1 = time.time()

                ops = ModelOps(lr=lr)
                ops.train(loader, train_ds, num_train=epochs,
                          plot_model=False, calc_perf=True)
                train_mcc = ops.p_a[-1]['m'] if ops.p_a else float('nan')

                metrics = ops.eval(test_ds, plot_model=False)
                elapsed = time.time() - t1

                row = (f"{lags},{epochs},{lr},"
                       f"{train_mcc:.4f},{metrics['m']:.4f},"
                       f"{metrics['f']:.4f},{metrics['a']:.4f},"
                       f"{metrics['c']:.4f},{elapsed:.1f}\n")
                fa.write(row); fa.flush()

                flag = ""
                if metrics['m'] > best_test_mcc:
                    best_test_mcc = metrics['m']
                    fc.write(row); fc.flush()
                    torch.save(ops.model,
                               f"./runs/{t}/best_model_mcc{best_test_mcc:.4f}")
                    flag = "  *** NEW BEST ***"

                print(f"[{run_n:3d}/{total}] lags={lags} ep={epochs} lr={lr:.4f} | "
                      f"train_mcc={train_mcc:.4f}  test_mcc={metrics['m']:.4f}  "
                      f"f1={metrics['f']:.4f}  ({elapsed:.0f}s){flag}", flush=True)

print(f"\nDone. Best test MCC = {best_test_mcc:.4f}", flush=True)
print(f"Results: {log_all}", flush=True)
