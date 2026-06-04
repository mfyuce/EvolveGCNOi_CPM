"""Quick smoke-test: load real JSON, train 1 epoch, report metrics."""
import os, sys, time
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from BurstAdmaDatasetLoader import BurstAdmaDatasetLoader
from torch_geometric_temporal.signal import temporal_signal_split
from graphs.recurrent.graphs_evolvegcn_h_improved import ModelOps

print("Loading dataset...", flush=True)
t0 = time.time()
loader = BurstAdmaDatasetLoader(features_as_self_edge=True, binary=True)
dataset = loader.get_dataset(lags=8)
print(f"Loaded in {time.time()-t0:.1f}s  n_node_features={loader.n_node_features}", flush=True)

train_dataset, test_dataset = temporal_signal_split(dataset, train_ratio=0.7)

print("Training 1 epoch...", flush=True)
t1 = time.time()
ops = ModelOps(lr=0.01)
ops.train(loader, train_dataset, num_train=10, plot_model=False, calc_perf=True)
print(f"Train done in {time.time()-t1:.1f}s", flush=True)

print("\nTrain history per epoch:")
print("epoch\tMCC\tF1\tacc\tloss")
for i, ep in enumerate(ops.p_a):
    c = ep['c'].item() if hasattr(ep['c'], 'item') else ep['c']
    print(f"{i+1}\t{ep['m']:.4f}\t{ep['f']:.4f}\t{ep['a']:.4f}\t{c:.4f}", flush=True)

print("\nEvaluating...", flush=True)
metrics = ops.eval(test_dataset, plot_model=False)
print(f"\nTest: MCC={metrics['m']:.4f}  F1={metrics['f']:.4f}  acc={metrics['a']:.4f}  loss={metrics['c']:.4f}")
print("DONE", flush=True)
