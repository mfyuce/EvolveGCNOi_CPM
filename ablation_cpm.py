"""
ablation_cpm.py — feature-group ablation + sensor_type-artifact robustness for the
honest CPM detector. Operates on cache_v1_rich.pt (RAW 0-8 + RICH 9-19), RF/HGB,
SAME vehicle-disjoint split as the GNN, 5 seeds, best-MCC threshold.

Index map (cache_v1_rich, in_f=20):
  RAW  0 x  1 y  2 heading  3 speed  4 accel  5 sensor_range  6 visibility
       7 sensor_type  8 weather
  RICH 9 dx 10 dy 11 disp 12 speed_resid 13 dspeed 14 dheading 15 daccel
       16 accel_resid 17 in_degree 18 d_in_degree 19 speed

Answers: (a) how much each feature group contributes; (b) does removing the suspected
sensor_type artifact (§7) drop the score? — if RAW+RICH ≈ RAW+RICH−sensor_type, the
65.1 result does NOT rest on the artifact.
"""
import argparse, time
import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
try:
    from sklearn.experimental import enable_hist_gradient_boosting  # noqa
except Exception:
    pass
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import matthews_corrcoef, roc_auc_score


def load(cache):
    c = torch.load(cache)
    X, Y, active = c['X'], c['Y'], c['active']
    nl = c['node_label'].numpy(); n = c['n_nodes']; T = c['T']; lags = c['lags']
    Xs, ys, vs = [], [], []
    for t in range(T - lags):
        m = active[t].numpy().astype(bool)
        Xs.append(X[t].numpy()[m]); ys.append((Y[t].numpy()[m] != 0).astype(int))
        vs.append(np.where(m)[0])
    return np.concatenate(Xs), np.concatenate(ys), np.concatenate(vs), nl, n


def disjoint(nl, n, s):
    rng = np.random.default_rng(s); trm = np.zeros(n, dtype=bool)
    for c in (0, 1):
        idx = np.where(nl == c)[0]; rng.shuffle(idx)
        trm[idx[:int(0.7 * len(idx))]] = True
    return trm


def bal_sw(y):
    w = np.ones(len(y))
    for c in (0, 1):
        nn = (y == c).sum()
        if nn > 0:
            w[y == c] = len(y) / (2.0 * nn)
    return w


def bmcc(y, p):
    bm = -1.0
    for t in np.linspace(0.05, 0.95, 19):
        m = matthews_corrcoef(y, (p >= t).astype(int))
        if m > bm:
            bm = m
    return bm


def ev(model, Xtr, ytr, Xte, yte):
    model.fit(Xtr, ytr, sample_weight=bal_sw(ytr))
    p = model.predict_proba(Xte)[:, 1]
    try:
        auc = roc_auc_score(yte, p)
    except Exception:
        auc = float('nan')
    return bmcc(yte, p) * 100, auc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', default='data/cache_v1_rich.pt')
    ap.add_argument('--seeds', default='1,2,3,4,5')
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(',')]
    t0 = time.time()
    X, y, v, nl, n = load(a.cache)
    ALL = list(range(20))
    subsets = {
        'RAW(9)': list(range(9)),
        'physics+struct(4)': [12, 16, 17, 18],         # speed_resid, accel_resid, indeg, d_indeg
        'RICH(11)': list(range(9, 20)),
        'RAW+RICH(20)': ALL,
        'RAW+RICH -sensor_type': [i for i in ALL if i != 7],
        'RAW+RICH -position': [i for i in ALL if i not in (0, 1)],
    }
    models = {
        'RF':  lambda s: RandomForestClassifier(n_estimators=150, n_jobs=16, random_state=s),
        'HGB': lambda s: HistGradientBoostingClassifier(max_iter=400, learning_rate=0.06,
                            max_leaf_nodes=63, l2_regularization=1.0, random_state=s),
    }
    print(f"cache={a.cache} samples={len(y):,} malicious={y.mean()*100:.2f}% ({time.time()-t0:.0f}s)")
    print(f"\n{'subset':24s} {'model':4s} {'MCC(mean±std)':>13s} {'AUC':>7s}")
    print('-' * 50)
    for sn, cols in subsets.items():
        for mn, mk in models.items():
            R = []
            for s in seeds:
                trm = disjoint(nl, n, s); tr = trm[v]; te = ~tr
                R.append(ev(mk(s), X[tr][:, cols], y[tr], X[te][:, cols], y[te]))
            R = np.array(R)
            print(f"{sn:24s} {mn:4s}  {R[:,0].mean():5.1f}±{R[:,0].std():4.1f}  {R[:,1].mean():.3f}",
                  flush=True)
    print("\n[robustness: RAW+RICH vs -sensor_type ~equal => 65.1 does NOT rest on the artifact]")


if __name__ == '__main__':
    main()
