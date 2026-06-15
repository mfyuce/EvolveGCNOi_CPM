"""
run_consensus_diag.py — WHY do consensus features hurt? redundant vs non-generalizing.

Loads cache_v1_consensus.pt (cols 0..19 = physics, 20..22 = consensus mismatch).
For each seed, RF MCC on:
  consensus-only  under VEHICLE-DISJOINT split  (held-out vehicles)
  consensus-only  under RANDOM node-step split   (same vehicles in train+test)
  physics-only    under both, for reference

If consensus-only is high on RANDOM but low on VEHICLE-DISJOINT, the 7.4x in-sample
separation is vehicle-specific memorization, not a generalizing attack signal —
which is exactly why adding it hurts the unseen-attacker protocol.
"""
import sys, time
import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import matthews_corrcoef, roc_auc_score

NSEED = int(sys.argv[1]) if len(sys.argv) > 1 else 5
CONS = "data/cache_v1_consensus.pt"


def bal_sw(y):
    w = np.ones(len(y))
    for c in (0, 1):
        n = (y == c).sum()
        if n > 0:
            w[y == c] = len(y) / (2.0 * n)
    return w


def best_mcc(y, p):
    bm = -1.0
    for thr in np.arange(0.05, 1.0, 0.025):
        pred = (p >= thr).astype(int)
        if pred.sum() == 0:
            continue
        m = matthews_corrcoef(y, pred)
        if m > bm:
            bm = m
    try:
        auc = roc_auc_score(y, p)
    except Exception:
        auc = float("nan")
    return bm * 100, auc


def main():
    t0 = time.time()
    c = torch.load(CONS)
    X, Y, active = c["X"], c["Y"], c["active"]
    n_nodes, T, lags = c["n_nodes"], c["T"], c["lags"]
    nstep = T - lags
    node_label = c["node_label"].numpy()
    Xc, yb, vid = [], [], []
    for t in range(nstep):
        m = active[t].numpy().astype(bool)
        Xc.append(X[t].numpy()[m]); yb.append((Y[t].numpy()[m] != 0).astype(int)); vid.append(np.where(m)[0])
    Xc = np.concatenate(Xc); yb = np.concatenate(yb); vid = np.concatenate(vid)
    PHYS = slice(0, 20); CONSF = slice(20, 23)
    print(f"  rows={len(yb)} pos={yb.mean()*100:.1f}%  in_f={Xc.shape[1]} ({time.time()-t0:.0f}s)", flush=True)

    def rf(Xtr, ytr, Xte, yte, seed):
        m = RandomForestClassifier(n_estimators=300, max_features=0.5, n_jobs=-1, random_state=seed)
        m.fit(Xtr, ytr, sample_weight=bal_sw(ytr))
        return best_mcc(yte, m.predict_proba(Xte)[:, 1])

    rows = {"cons_vdisj": [], "cons_rand": [], "phys_vdisj": [], "phys_rand": []}
    for s in range(NSEED):
        # vehicle-disjoint split
        rng = np.random.default_rng(s); trm = np.zeros(n_nodes, dtype=bool)
        for cc in (0, 1):
            idx = np.where(node_label == cc)[0]; rng.shuffle(idx)
            trm[idx[:int(0.7 * len(idx))]] = True
        trv = trm[vid]; tev = ~trv
        # random node-step split (same vehicles both sides)
        rr = np.random.default_rng(1000 + s)
        perm = rr.permutation(len(yb)); ntr = int(0.7 * len(yb))
        trr = np.zeros(len(yb), bool); trr[perm[:ntr]] = True; ter = ~trr

        cvd = rf(Xc[trv][:, CONSF], yb[trv], Xc[tev][:, CONSF], yb[tev], s)
        crd = rf(Xc[trr][:, CONSF], yb[trr], Xc[ter][:, CONSF], yb[ter], s)
        pvd = rf(Xc[trv][:, PHYS], yb[trv], Xc[tev][:, PHYS], yb[tev], s)
        prd = rf(Xc[trr][:, PHYS], yb[trr], Xc[ter][:, PHYS], yb[ter], s)
        rows["cons_vdisj"].append(cvd[0]); rows["cons_rand"].append(crd[0])
        rows["phys_vdisj"].append(pvd[0]); rows["phys_rand"].append(prd[0])
        print(f"RESULT seed={s} cons_vdisj={cvd[0]:.2f} cons_rand={crd[0]:.2f} "
              f"phys_vdisj={pvd[0]:.2f} phys_rand={prd[0]:.2f}", flush=True)

    print(f"\n=== consensus-only: vehicle-disjoint vs random ({NSEED} seeds) ===", flush=True)
    for k in rows:
        v = np.array(rows[k]); print(f"  {k:12s} {v.mean():5.1f} +/- {v.std():4.1f}", flush=True)
    cv, cr = np.array(rows["cons_vdisj"]).mean(), np.array(rows["cons_rand"]).mean()
    print(f"\n  consensus random - vdisj gap = {cr-cv:+.1f}  "
          f"(large gap => vehicle-specific memorization, not a generalizing attack cue)", flush=True)
    print(f"(total {time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
