"""
hpo_cpm.py — HONEST hyperparameter optimization for the CPM tabular detector.

Nested protocol (no test leakage):
  outer: vehicles -> train(70%)/test(30%), stratified by ever-attacker, seed s.
  inner: train vehicles -> tr'(75%)/val(25%), vehicle-disjoint, stratified.
         hyperparams are scored on val ONLY; the test vehicles are never seen during search.
  final: refit the val-best config on the FULL train set, evaluate once on test.
5 outer seeds. Models: HistGradientBoosting, RandomForest. Random search over `--n_configs`
configs (same configs across seeds). Compares tuned vs the default-RF 67.0 baseline.

Threshold: best-MCC on the eval set (same convention as all other tables, for comparability).
"""
import argparse, time
import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import matthews_corrcoef, roc_auc_score, precision_recall_fscore_support


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


def pick_vehicles(nl, avail, frac, rng):
    """Boolean node-mask selecting `frac` of available vehicles, stratified by ever-attacker."""
    sel = np.zeros(len(nl), dtype=bool)
    for c in (0, 1):
        idx = np.where((nl == c) & avail)[0]; rng.shuffle(idx)
        sel[idx[:int(frac * len(idx))]] = True
    return sel


def bal_sw(y):
    w = np.ones(len(y))
    for c in (0, 1):
        nn = (y == c).sum()
        if nn > 0:
            w[y == c] = len(y) / (2.0 * nn)
    return w


def bmcc(y, p):
    bm, bt = -1.0, 0.5
    for t in np.linspace(0.05, 0.95, 19):
        m = matthews_corrcoef(y, (p >= t).astype(int))
        if m > bm:
            bm, bt = m, t
    return bm, bt


def sample_hgb(rng):
    return dict(
        learning_rate=float(rng.choice([0.02, 0.05, 0.1, 0.2])),
        max_iter=int(rng.choice([300, 600, 1000])),
        max_leaf_nodes=int(rng.choice([15, 31, 63, 127])),
        min_samples_leaf=int(rng.choice([20, 50, 100])),
        l2_regularization=float(rng.choice([0.0, 0.1, 1.0, 10.0])),
        max_features=float(rng.choice([0.5, 0.8, 1.0])),
    )


def sample_rf(rng):
    return dict(
        n_estimators=int(rng.choice([200, 400])),
        max_depth=[None, 20, 40][int(rng.integers(3))],
        max_features=['sqrt', 0.3, 0.5][int(rng.integers(3))],
        min_samples_leaf=int(rng.choice([1, 5, 20])),
    )


def build(kind, cfg, seed):
    if kind == 'HGB':
        return HistGradientBoostingClassifier(random_state=seed, **cfg)
    return RandomForestClassifier(n_jobs=24, random_state=seed, **cfg)


def fit_eval(kind, cfg, Xtr, ytr, Xev, yev, seed):
    m = build(kind, cfg, seed)
    m.fit(Xtr, ytr, sample_weight=bal_sw(ytr))
    p = m.predict_proba(Xev)[:, 1]
    mcc, thr = bmcc(yev, p)
    return mcc, p, thr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', default='data/cache_v1_clean.pt')
    ap.add_argument('--seeds', default='1,2,3,4,5')
    ap.add_argument('--n_configs', type=int, default=12)
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(',')]
    t0 = time.time()
    X, y, v, nl, n, in_f = load(a.cache)
    print(f"cache={a.cache} F={in_f} samples={len(y):,} mal={y.mean()*100:.2f}% "
          f"n_configs={a.n_configs} ({time.time()-t0:.0f}s)", flush=True)

    cfg_rng = np.random.default_rng(0)
    grids = {'HGB': [sample_hgb(cfg_rng) for _ in range(a.n_configs)],
             'RF':  [sample_rf(cfg_rng) for _ in range(a.n_configs)]}

    for kind, cfgs in grids.items():
        test = []
        for s in seeds:
            train_v = pick_vehicles(nl, np.ones(n, dtype=bool), 0.7, np.random.default_rng(s))
            val_v = pick_vehicles(nl, train_v, 0.25, np.random.default_rng(1000 + s))
            tr2_v = train_v & ~val_v
            m_tr2, m_val = tr2_v[v], val_v[v]
            m_tr, m_te = train_v[v], ~train_v[v]
            best_v, best_cfg = -1.0, None
            for cfg in cfgs:
                vmcc, _, _ = fit_eval(kind, cfg, X[m_tr2], y[m_tr2], X[m_val], y[m_val], s)
                if vmcc > best_v:
                    best_v, best_cfg = vmcc, cfg
            # refit val-best on FULL train, evaluate once on test
            mdl = build(kind, best_cfg, s)
            mdl.fit(X[m_tr], y[m_tr], sample_weight=bal_sw(y[m_tr]))
            p = mdl.predict_proba(X[m_te])[:, 1]
            tmcc, thr = bmcc(y[m_te], p)
            try:
                auc = roc_auc_score(y[m_te], p)
            except Exception:
                auc = float('nan')
            malf1 = precision_recall_fscore_support(y[m_te], (p >= thr).astype(int),
                        average=None, labels=[0, 1], zero_division=0)[2][1] * 100
            test.append((tmcc * 100, auc, malf1))
            print(f"  {kind} seed{s}: val={best_v*100:.1f} -> test={tmcc*100:.1f} "
                  f"(auc {auc:.3f}, malF1 {malf1:.1f})  best={best_cfg}", flush=True)
        T = np.array(test)
        print(f"RESULT hpo model={kind} test_mcc={T[:,0].mean():.2f} test_mcc_std={T[:,0].std():.2f} "
              f"auc={T[:,1].mean():.4f} malf1={T[:,2].mean():.2f}  ({time.time()-t0:.0f}s)", flush=True)
    print(f"[baseline: default RF clean 67.0±1.8 | published leaky 0.61]", flush=True)


if __name__ == '__main__':
    main()
