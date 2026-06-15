"""
run_consensus_rf.py — decisive paired RF test for the CPM consensus features.

Compares, seed-by-seed (vehicle-disjoint, same split logic / best-MCC threshold as
expC_cpm.py rf_eng), the per-node RICH cache vs the same cache with 3 collective-
perception CONSENSUS features appended (self-vs-observers mismatch).

  A = cache_v1_rich.pt        (physics only; must reproduce the committed ~69.0)
  B = cache_v1_consensus.pt   (physics + consensus)

Usage: python run_consensus_rf.py [nseeds]
"""
import sys, time
import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import matthews_corrcoef, roc_auc_score, f1_score

NSEED = int(sys.argv[1]) if len(sys.argv) > 1 else 10
RICH = "data/cache_v1_rich.pt"
CONS = "data/cache_v1_consensus.pt"


def bal_sw(y):
    w = np.ones(len(y))
    for c in (0, 1):
        n = (y == c).sum()
        if n > 0:
            w[y == c] = len(y) / (2.0 * n)
    return w


def best_eval(y, p):
    bm, bt = -1.0, 0.5
    for thr in np.arange(0.05, 1.0, 0.025):
        pred = (p >= thr).astype(int)
        if pred.sum() == 0:
            continue
        m = matthews_corrcoef(y, pred)
        if m > bm:
            bm, bt = m, thr
    pred = (p >= bt).astype(int)
    try:
        auc = roc_auc_score(y, p)
    except Exception:
        auc = float("nan")
    return bm * 100, auc, f1_score(y, pred, average="macro", zero_division=0) * 100


def prep(cache):
    c = torch.load(cache)
    X, Y, active = c["X"], c["Y"], c["active"]
    n_nodes, T, lags, in_f = c["n_nodes"], c["T"], c["lags"], c["in_f"]
    nstep = T - lags
    node_label = c["node_label"].numpy()
    Xc, yb, vid = [], [], []
    for t in range(nstep):
        m = active[t].numpy().astype(bool)
        Xc.append(X[t].numpy()[m]); yb.append((Y[t].numpy()[m] != 0).astype(int)); vid.append(np.where(m)[0])
    return (np.concatenate(Xc), np.concatenate(yb), np.concatenate(vid), node_label, n_nodes, in_f)


def split(node_label, n_nodes, seed):
    rng = np.random.default_rng(seed); trm = np.zeros(n_nodes, dtype=bool)
    for cc in (0, 1):
        idx = np.where(node_label == cc)[0]; rng.shuffle(idx)
        trm[idx[:int(0.7 * len(idx))]] = True
    return trm


def rf_mcc(prepped, seed):
    Xc, yb, vid, node_label, n_nodes, in_f = prepped
    trm = split(node_label, n_nodes, seed)
    tr = trm[vid]; te = ~tr
    rf = RandomForestClassifier(n_estimators=300, max_features=0.5, n_jobs=-1, random_state=seed)
    rf.fit(Xc[tr], yb[tr], sample_weight=bal_sw(yb[tr]))
    return best_eval(yb[te], rf.predict_proba(Xc[te])[:, 1])


def main():
    t0 = time.time()
    print(f"prepping caches ...", flush=True)
    A = prep(RICH); B = prep(CONS)
    print(f"  A in_f={A[5]}  B in_f={B[5]}  ({time.time()-t0:.0f}s)", flush=True)
    da, db = [], []
    for s in range(NSEED):
        ma, aa, _ = rf_mcc(A, s); mb, ab, _ = rf_mcc(B, s)
        da.append(ma); db.append(mb)
        print(f"RESULT seed={s} rich={ma:.2f} consensus={mb:.2f} delta={mb-ma:+.2f} "
              f"(auc {aa:.4f}->{ab:.4f})", flush=True)
    da, db = np.array(da), np.array(db)
    print(f"\n=== CPM consensus paired RF ({NSEED} seeds, vehicle-disjoint) ===", flush=True)
    print(f"  rich (physics)        {da.mean():5.1f} +/- {da.std():4.1f}", flush=True)
    print(f"  + consensus           {db.mean():5.1f} +/- {db.std():4.1f}", flush=True)
    print(f"  consensus lift        {(db-da).mean():+5.1f}   ({(db>da).sum()}/{len(db)} seeds>0)", flush=True)
    try:
        from scipy.stats import wilcoxon
        if np.any((db - da) != 0):
            print(f"  Wilcoxon p={wilcoxon(db-da).pvalue:.4f}", flush=True)
    except Exception:
        pass
    print(f"(total {time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
