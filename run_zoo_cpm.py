"""
run_zoo_cpm.py — train/eval any model from models_cpm on the CPM dataset,
vehicle-disjoint split, truncated BPTT, focal loss, best-MCC threshold.

Usage
-----
  python run_zoo_cpm.py <model> <seed> [--dataset PATH] [--epochs N]
                        [--window W] [--hidden H] [--lr LR] [--proj P]
                        [--pool_ratio R] [--lags L] [--tag NAME]

Prints a full metrics block and a final machine-parseable RESULT line:
  RESULT model=<m> seed=<s> mcc=<..> macro_f1=<..> mal_f1=<..> roc_auc=<..> acc=<..> thr=<..>
"""
import os, sys, time, argparse
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (precision_recall_fscore_support, accuracy_score,
                             matthews_corrcoef, roc_auc_score)
torch.set_num_threads(1)

from BurstAdmaDatasetLoader import BurstAdmaDatasetLoader
import models_cpm


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("model")
    p.add_argument("seed", type=int, nargs="?", default=3)
    p.add_argument("--dataset", default=None, help="path to JSON (default: v1 in repo)")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--window", type=int, default=24)
    p.add_argument("--hidden", type=int, default=32)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--proj", type=int, default=32)
    p.add_argument("--pool_ratio", type=float, default=0.25)
    p.add_argument("--lags", type=int, default=1)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--gamma", type=float, default=2.0, help="focal loss gamma")
    p.add_argument("--tag", default="")
    return p.parse_args()


def focal(lg, t, a, g=2.0):
    lp = F.log_softmax(lg, -1).gather(1, t.unsqueeze(1)).squeeze(1)
    return (((1 - lp.exp()) ** g) * (-lp) * a.to(lg.device)[t]).mean()


def class_weights(y):
    # inverse-frequency weights for the 2 classes present in this snapshot
    w = torch.ones(2)
    for c in (0, 1):
        n = (y == c).sum().item()
        if n > 0:
            w[c] = len(y) / (2.0 * n)
    return w


def main():
    args = parse_args()
    tag = args.tag or args.model
    t0 = time.time()
    def out(m): print(m, flush=True)

    ds_name = os.path.basename(args.dataset) if args.dataset else "v1(default)"
    out(f"=== ZOO — {args.model} (seed={args.seed}) dataset={ds_name} "
        f"H={args.hidden} W={args.window} ep={args.epochs} lr={args.lr} "
        f"proj={args.proj} pool={args.pool_ratio} ===")

    out("Loading dataset...")
    lb = BurstAdmaDatasetLoader(features_as_self_edge=True, binary=True,
                                json_path=args.dataset)
    ds = lb.get_dataset(lags=args.lags)
    alls = list(ds)
    n_nodes = len(lb._dataset["node_labels"])
    T = lb._dataset["time_periods"]
    lags = lb.lags
    in_f = lb.n_node_features
    out(f"Loaded: T={T} N={n_nodes} F={in_f}  ({time.time()-t0:.1f}s)")

    active = [torch.tensor(np.asarray(alls[i].x).__abs__().sum(1) != 0.0)
              for i in range(T - lags)]
    X  = [torch.as_tensor(a.x, dtype=torch.float32) for a in alls]
    Y  = [torch.tensor(lb.targets[i], dtype=torch.long) for i in range(T - lags)]
    EI = [torch.as_tensor(a.edge_index, dtype=torch.long) for a in alls]
    EW = [torch.as_tensor(a.edge_attr, dtype=torch.float32) for a in alls]

    # vehicle-disjoint split, stratified by ever-attacker
    yf = np.stack([lb.targets[i] for i in range(T - lags)])
    node_label = (yf.max(0) > 0).astype(int)
    rng = np.random.default_rng(args.seed); trm = np.zeros(n_nodes, dtype=bool)
    for c in (0, 1):
        idx = np.where(node_label == c)[0]; rng.shuffle(idx)
        trm[idx[:int(0.7 * len(idx))]] = True
    trv, tev = torch.tensor(trm), torch.tensor(~trm)
    n_tr = int(0.7 * (T - lags))
    out(f"Split: {trm.sum()} train / {(~trm).sum()} test nodes | {n_tr} train steps")

    torch.manual_seed(args.seed)
    model = models_cpm.build(args.model, n_nodes, in_f, h=args.hidden,
                             d=args.dropout, proj=args.proj,
                             pool_ratio=args.pool_ratio)
    kind = model.kind
    has_reset = hasattr(model, "reset")
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    out(f"Model kind={kind}  params={sum(p.numel() for p in model.parameters()):,}")

    # ── train ──────────────────────────────────────────────────────────────────
    out(f"\nTraining {args.epochs} epochs (window={args.window})...")
    for ep in range(args.epochs):
        model.train(); H = None
        if has_reset:
            model.reset()
        for st in range(0, n_tr, args.window):
            if kind == "node_rec" and H is not None:
                H = H.detach()
            if has_reset:
                model.reset()
            opt.zero_grad(); wl = 0.0; c = 0
            for i in range(st, min(st + args.window, n_tr)):
                m = active[i] & trv
                if kind == "node_rec":
                    lg, H = model(X[i], EI[i], EW[i], H)
                else:
                    lg = model(X[i], EI[i], EW[i])
                if m.sum() == 0:
                    continue
                wl = wl + focal(lg[m], Y[i][m], class_weights(Y[i][m]), g=args.gamma); c += 1
            if c:
                (wl / c).backward(); opt.step()
        if (ep + 1) % 10 == 0:
            out(f"  epoch {ep+1}/{args.epochs}  ({time.time()-t0:.1f}s)")

    # ── evaluate on test vehicles ────────────────────────────────────────────────
    model.eval(); P, L = [], []; H = None
    if has_reset:
        model.reset()
    with torch.no_grad():
        for i in range(T - lags):
            if kind == "node_rec":
                lg, H = model(X[i], EI[i], EW[i], H)
            else:
                lg = model(X[i], EI[i], EW[i])
            m = active[i] & tev
            if m.sum() > 0:
                P.append(torch.softmax(lg, 1)[m, 1].numpy())
                L.append(Y[i][m].numpy())
    P = np.concatenate(P); L = np.concatenate(L)

    # best-MCC threshold sweep
    best_thr, best_mcc = 0.5, -1
    for thr in np.linspace(0.01, 0.99, 99):
        mcc = matthews_corrcoef(L, (P >= thr).astype(int))
        if mcc > best_mcc:
            best_mcc, best_thr = mcc, thr
    pred = (P >= best_thr).astype(int)

    macro = precision_recall_fscore_support(L, pred, average="macro", zero_division=0)
    perc  = precision_recall_fscore_support(L, pred, average=None, labels=[0, 1], zero_division=0)
    acc   = accuracy_score(L, pred) * 100
    try:
        auc = roc_auc_score(L, P)
    except Exception:
        auc = float("nan")
    ben_f1 = perc[2][0] * 100
    mal_f1 = perc[2][1] * 100
    macro_f1 = macro[2] * 100
    mcc100 = best_mcc * 100

    out(f"\noperating threshold (best-MCC) = {best_thr:.2f}")
    out(f"  macro-F1={macro_f1:.2f}  benign-F1={ben_f1:.2f}  malicious-F1={mal_f1:.2f}")
    out(f"  Accuracy={acc:.2f}  ROC-AUC={auc:.4f}  MCC={mcc100:.2f}  | {(time.time()-t0)/60:.1f}min")
    out(f"RESULT model={args.model} seed={args.seed} dataset={ds_name} "
        f"mcc={mcc100:.2f} macro_f1={macro_f1:.2f} mal_f1={mal_f1:.2f} "
        f"ben_f1={ben_f1:.2f} roc_auc={auc:.4f} acc={acc:.2f} thr={best_thr:.2f} "
        f"H={args.hidden} W={args.window} ep={args.epochs} lr={args.lr} tag={tag}")


if __name__ == "__main__":
    main()
