"""tabular_cache.py — RF/HGB on ALL features of a cache, vehicle-disjoint, 5 seeds,
best-MCC threshold. A pure-tabular reference (no GNN) for any feature set."""
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
    return np.concatenate(Xs), np.concatenate(ys), np.concatenate(vs), nl, n, c['in_f']


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', required=True)
    ap.add_argument('--seeds', default='1,2,3,4,5')
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(',')]
    X, y, v, nl, n, in_f = load(a.cache)
    models = {
        'RF':  lambda s: RandomForestClassifier(n_estimators=150, n_jobs=16, random_state=s),
        'HGB': lambda s: HistGradientBoostingClassifier(max_iter=400, learning_rate=0.06,
                            max_leaf_nodes=63, l2_regularization=1.0, random_state=s),
    }
    for mn, mk in models.items():
        R = []
        for s in seeds:
            trm = disjoint(nl, n, s); tr = trm[v]; te = ~tr
            m = mk(s); m.fit(X[tr], y[tr], sample_weight=bal_sw(y[tr]))
            p = m.predict_proba(X[te])[:, 1]
            try:
                auc = roc_auc_score(y[te], p)
            except Exception:
                auc = float('nan')
            R.append((bmcc(y[te], p) * 100, auc))
        R = np.array(R)
        print(f"RESULT tabular cache={a.cache} F={in_f} model={mn} "
              f"mcc={R[:,0].mean():.2f} mcc_std={R[:,0].std():.2f} auc={R[:,1].mean():.4f}", flush=True)


if __name__ == '__main__':
    main()
