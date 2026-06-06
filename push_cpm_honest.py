"""
push_cpm_honest.py — can an HONEST CPM pipeline reach the published ~0.61 MCC?

Builds richer physics-consistency + temporal + structural node features from a base
cache (no label leakage), then evaluates strong tabular models (RandomForest,
HistGradientBoosting) on the SAME vehicle-disjoint split as the GNN, 5 seeds,
best-MCC threshold, full metrics. Compares RAW vs RICH vs RAW+RICH.

Rich features (per active node-step, vs the node's previous active appearance):
  dx, dy, disp(=pos_jump),
  speed_resid = |disp - speed_detected|        physics: displacement vs reported speed
  dspeed, dheading(circular), daccel,
  accel_resid = |Δspeed - accel_detected|      physics: Δspeed vs reported accel
  in_degree, d_in_degree,                       structural: #sensors seeing it (+ change)
  speed                                         raw speed for context
Base kinematic indices (v1/v2): 0 x  1 y  2 heading(deg)  3 speed  4 accel.
"""
import argparse, time
import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
try:
    from sklearn.experimental import enable_hist_gradient_boosting  # noqa  (older sklearn)
except Exception:
    pass
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (matthews_corrcoef, roc_auc_score,
                             precision_recall_fscore_support)

IX, IY, IH, IS, IA = 0, 1, 2, 3, 4


def circ_abs_deg(a, b):
    return np.abs((a - b + 180.0) % 360.0 - 180.0)


def build(cache):
    c = torch.load(cache)
    X, active, Y, EI = c['X'], c['active'], c['Y'], c['EI']
    node_label = c['node_label'].numpy()
    n_nodes, T, lags = c['n_nodes'], c['T'], c['lags']
    last = np.zeros((n_nodes, 5)); last_indeg = np.zeros(n_nodes)
    seen = np.zeros(n_nodes, dtype=bool)
    raws, richs, labs, vids = [], [], [], []
    for t in range(T - lags):
        x = X[t].numpy(); m = active[t].numpy().astype(bool)
        ei = EI[t].numpy()
        indeg = np.zeros(n_nodes)
        if ei.size:
            s, d = ei[0], ei[1]; pr = s != d
            if pr.any():
                np.add.at(indeg, d[pr], 1.0)
        dx = np.abs(x[:, IX] - last[:, 0]); dy = np.abs(x[:, IY] - last[:, 1])
        disp = np.sqrt(dx ** 2 + dy ** 2)
        speed_resid = np.abs(disp - x[:, IS])
        dspeed = x[:, IS] - last[:, 3]
        dheading = circ_abs_deg(x[:, IH], last[:, 2])
        daccel = np.abs(x[:, IA] - last[:, 4])
        accel_resid = np.abs(dspeed - x[:, IA])
        d_indeg = np.abs(indeg - last_indeg)
        for arr in (dx, dy, disp, speed_resid, dheading, daccel, accel_resid, d_indeg):
            arr[~seen] = 0.0
        dspeed_abs = np.abs(dspeed); dspeed_abs[~seen] = 0.0
        rich = np.stack([dx, dy, disp, speed_resid, dspeed_abs, dheading, daccel,
                         accel_resid, indeg, d_indeg, x[:, IS]], axis=1)
        rich[~m] = 0.0
        raws.append(x[m]); richs.append(rich[m])
        labs.append(Y[t].numpy()[m]); vids.append(np.where(m)[0])
        last[m, 0] = x[m, IX]; last[m, 1] = x[m, IY]; last[m, 2] = x[m, IH]
        last[m, 3] = x[m, IS]; last[m, 4] = x[m, IA]
        last_indeg[m] = indeg[m]; seen[m] = True
    return (np.concatenate(raws), np.concatenate(richs),
            (np.concatenate(labs) != 0).astype(int), np.concatenate(vids),
            node_label, n_nodes)


def disjoint(node_label, n, seed):
    rng = np.random.default_rng(seed); trm = np.zeros(n, dtype=bool)
    for c in (0, 1):
        idx = np.where(node_label == c)[0]; rng.shuffle(idx)
        trm[idx[:int(0.7 * len(idx))]] = True
    return trm


def bal_sw(y):
    w = np.ones(len(y))
    for c in (0, 1):
        n = (y == c).sum()
        if n > 0:
            w[y == c] = len(y) / (2.0 * n)
    return w


def best_mcc(y, p):
    bt, bm = 0.5, -1.0
    for thr in np.linspace(0.05, 0.95, 19):
        mc = matthews_corrcoef(y, (p >= thr).astype(int))
        if mc > bm:
            bm, bt = mc, thr
    return bm, bt


def evalm(model, Xtr, ytr, Xte, yte):
    model.fit(Xtr, ytr, sample_weight=bal_sw(ytr))
    p = model.predict_proba(Xte)[:, 1]
    mcc, thr = best_mcc(yte, p)
    try:
        auc = roc_auc_score(yte, p)
    except Exception:
        auc = float('nan')
    pr = precision_recall_fscore_support(yte, (p >= thr).astype(int),
                                         average=None, labels=[0, 1], zero_division=0)
    return mcc * 100, auc, pr[2][1] * 100, pr[2][0] * 100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', required=True)
    ap.add_argument('--seeds', default='1,2,3,4,5')
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(',')]
    t0 = time.time()
    RAW, RICH, y, v, node_label, n_nodes = build(a.cache)
    sets = {'RAW': RAW, 'RICH': RICH, 'RAW+RICH': np.concatenate([RAW, RICH], axis=1)}
    print(f"cache={a.cache}  RAW={RAW.shape[1]}f RICH={RICH.shape[1]}f  "
          f"samples={len(y):,}  malicious={y.mean()*100:.2f}%  ({time.time()-t0:.0f}s)")
    models = {
        'RF':  lambda s: RandomForestClassifier(n_estimators=150, n_jobs=16, random_state=s),
        'HGB': lambda s: HistGradientBoostingClassifier(max_iter=400, learning_rate=0.06,
                            max_leaf_nodes=63, l2_regularization=1.0, random_state=s),
    }
    print(f"\n{'feat':9s} {'model':4s} {'MCC(mean±std)':>14s} {'ROC':>7s} {'malF1':>7s} {'benF1':>7s}")
    print('-' * 52)
    for fn, Xf in sets.items():
        for mn, mk in models.items():
            R = []
            for s in seeds:
                trm = disjoint(node_label, n_nodes, s); tr = trm[v]; te = ~tr
                R.append(evalm(mk(s), Xf[tr], y[tr], Xf[te], y[te]))
            R = np.array(R)
            print(f"{fn:9s} {mn:4s}  {R[:,0].mean():5.1f}±{R[:,0].std():4.1f}  "
                  f"{R[:,1].mean():.3f}  {R[:,2].mean():6.1f}  {R[:,3].mean():6.1f}",
                  flush=True)
    print(f"\n[targets: published 0.61 | GNN gconvgru raw45.9/aug48.6/tuned50.0 | "
          f"RF-aug probe 55.0]  ({time.time()-t0:.0f}s)")


if __name__ == '__main__':
    main()
