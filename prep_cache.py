"""
prep_cache.py — convert a CPM JSON dataset into a compact torch cache (.pt).

The JSON loader balloons the 5 GB file to ~35 GB of Python objects and takes
~3 min. For a 100+ run sweep that is the dominant cost. This script loads once
and saves tensors (X/EI/EW/Y/active + node ever-attacker label) so run_zoo_cpm
can reload in seconds with a fraction of the RAM.

Usage
-----
  python prep_cache.py --dataset data/v2/...v2.json --out data/v2/cache_v2.pt
  python prep_cache.py                              --out data/cache_v1.pt   # default v1
"""
import os, sys, time, argparse
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
import numpy as np
import torch
from BurstAdmaDatasetLoader import BurstAdmaDatasetLoader


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default=None)
    p.add_argument("--out", required=True)
    p.add_argument("--lags", type=int, default=1)
    args = p.parse_args()

    t0 = time.time()
    print(f"Loading {args.dataset or 'v1(default)'} ...", flush=True)
    lb = BurstAdmaDatasetLoader(features_as_self_edge=True, binary=True,
                                json_path=args.dataset)
    ds = lb.get_dataset(lags=args.lags)
    alls = list(ds)
    T = lb._dataset["time_periods"]
    lags = lb.lags
    n_nodes = len(lb._dataset["node_labels"])
    in_f = lb.n_node_features
    print(f"  loaded T={T} N={n_nodes} F={in_f} ({time.time()-t0:.0f}s)", flush=True)

    X  = [torch.as_tensor(a.x, dtype=torch.float32) for a in alls]
    EI = [torch.as_tensor(a.edge_index, dtype=torch.long) for a in alls]
    EW = [torch.as_tensor(a.edge_attr, dtype=torch.float32) for a in alls]
    Y  = [torch.tensor(lb.targets[i], dtype=torch.long) for i in range(T - lags)]
    active = [(X[i].abs().sum(1) != 0.0) for i in range(T - lags)]
    yf = np.stack([lb.targets[i] for i in range(T - lags)])
    node_label = torch.tensor((yf.max(0) > 0).astype(np.int64))

    cache = {
        "X": X, "EI": EI, "EW": EW, "Y": Y, "active": active,
        "node_label": node_label, "T": T, "lags": lags,
        "n_nodes": n_nodes, "in_f": in_f,
    }
    torch.save(cache, args.out)
    sz = os.path.getsize(args.out) / 1e6
    print(f"Saved {args.out}  ({sz:.0f} MB, {time.time()-t0:.0f}s total)", flush=True)


if __name__ == "__main__":
    main()
