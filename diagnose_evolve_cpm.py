"""
diagnose_evolve_cpm.py — measure EvolveGCN-H weight-recurrence collapse (B2).

The EvolveGCN-H GRU evolves the in_f×in_f conv weight across time. If the
optimizer drives that recurrence to a (near-)constant W, the temporal signal
is unused. We log the relative weight jump  ‖W_t − W_{t-1}‖ / ‖W_{t-1}‖
across windows during training. A decay toward ~0 confirms the collapse and
motivates moving the recurrence to node states (the M6 hybrid).

Usage
-----
  python diagnose_evolve_cpm.py --cache data/v2/cache_v2.pt --model evolve_h --seed 3
  python diagnose_evolve_cpm.py --cache data/v2/cache_v2.pt --model hybrid  --seed 3
"""
import os, argparse, time
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np, torch, torch.nn.functional as F
torch.set_num_threads(1)
import models_cpm
from run_zoo_cpm import focal, class_weights


def get_weight(model):
    """Return a detached copy of the evolving conv weight, or None."""
    rec = getattr(model, "rec", None)
    if rec is None:
        return None
    w = getattr(rec, "weight", None)
    if w is None:
        return None
    return w.detach().clone()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True)
    p.add_argument("--model", default="evolve_h")
    p.add_argument("--seed", type=int, default=3)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--window", type=int, default=24)
    p.add_argument("--hidden", type=int, default=32)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--proj", type=int, default=32)
    args = p.parse_args()

    c = torch.load(args.cache)
    X, EI, EW, Y, active = c["X"], c["EI"], c["EW"], c["Y"], c["active"]
    node_label = c["node_label"].numpy()
    n_nodes, T, lags, in_f = c["n_nodes"], c["T"], c["lags"], c["in_f"]

    rng = np.random.default_rng(args.seed); trm = np.zeros(n_nodes, dtype=bool)
    for cl in (0, 1):
        idx = np.where(node_label == cl)[0]; rng.shuffle(idx)
        trm[idx[:int(0.7 * len(idx))]] = True
    trv = torch.tensor(trm)
    n_tr = int(0.7 * (T - lags))

    torch.manual_seed(args.seed)
    model = models_cpm.build(args.model, n_nodes, in_f, h=args.hidden, proj=args.proj)
    has_reset = hasattr(model, "reset")
    kind = model.kind
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    print(f"# model={args.model} seed={args.seed} in_f={in_f}")
    print("epoch mean_rel_jump max_rel_jump final_W_norm")
    for ep in range(args.epochs):
        model.train(); H = None
        if has_reset:
            model.reset()
        jumps = []
        for st in range(0, n_tr, args.window):
            if kind == "node_rec" and H is not None:
                H = H.detach()
            if has_reset:
                model.reset()
            opt.zero_grad(); wl = 0.0; cc = 0
            W_prev = None
            for i in range(st, min(st + args.window, n_tr)):
                m = active[i] & trv
                if kind == "node_rec":
                    lg, H = model(X[i], EI[i], EW[i], H)
                else:
                    lg = model(X[i], EI[i], EW[i])
                W_cur = get_weight(model)
                if W_cur is not None and W_prev is not None:
                    num = (W_cur - W_prev).norm().item()
                    den = W_prev.norm().item() + 1e-12
                    jumps.append(num / den)
                W_prev = W_cur
                if m.sum() == 0:
                    continue
                wl = wl + focal(lg[m], Y[i][m], class_weights(Y[i][m])); cc += 1
            if cc:
                (wl / cc).backward(); opt.step()
        if jumps:
            wn = get_weight(model)
            wn = wn.norm().item() if wn is not None else float("nan")
            print(f"{ep+1:3d} {np.mean(jumps):.5f} {np.max(jumps):.5f} {wn:.4f}", flush=True)
        else:
            print(f"{ep+1:3d} (no evolving weight to track)", flush=True)


if __name__ == "__main__":
    main()
