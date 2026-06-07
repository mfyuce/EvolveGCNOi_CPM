"""
honest_eval_cpm.py — fully-honest threshold check for the HPO-winning RF config.

Isolates the ONE remaining optimism in the HPO (the operating threshold was picked on
test). Here the threshold is chosen on a train-internal validation split (vehicle-disjoint)
and applied to test, on the SAME tr'-trained model — so the gap to the test-optimised
threshold is purely the threshold's contribution.

  fit RF(winning cfg) on tr' (75% of train vehicles)
  thr_val = best-MCC threshold on val (25% of train vehicles)        [honest]
  report test MCC at thr_val (deployable-honest) vs best-MCC-on-test (optimistic)
5 seeds, clean 17-d physics features.
"""
import argparse, time
import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
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


def pick(nl, avail, frac, rng):
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


def best_thr(y, p):
    bm, bt = -1.0, 0.5
    for t in np.linspace(0.05, 0.95, 19):
        m = matthews_corrcoef(y, (p >= t).astype(int))
        if m > bm:
            bm, bt = m, t
    return bt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', default='data/cache_v1_clean.pt')
    ap.add_argument('--seeds', default='1,2,3,4,5')
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(',')]
    X, y, v, nl, n = load(a.cache)
    cfg = dict(n_estimators=400, max_features=0.5, min_samples_leaf=1, max_depth=None)
    print(f"winning cfg = {cfg}")
    hon, opt, aucs = [], [], []
    for s in seeds:
        train_v = pick(nl, np.ones(n, dtype=bool), 0.7, np.random.default_rng(s))
        val_v = pick(nl, train_v, 0.25, np.random.default_rng(1000 + s))
        tr2_v = train_v & ~val_v
        m_tr2, m_val, m_te = tr2_v[v], val_v[v], (~train_v)[v]
        rf = RandomForestClassifier(n_jobs=24, random_state=s, **cfg)
        rf.fit(X[m_tr2], y[m_tr2], sample_weight=bal_sw(y[m_tr2]))
        p_val = rf.predict_proba(X[m_val])[:, 1]
        thr_val = best_thr(y[m_val], p_val)                         # honest threshold
        p_te = rf.predict_proba(X[m_te])[:, 1]
        honest_mcc = matthews_corrcoef(y[m_te], (p_te >= thr_val).astype(int)) * 100
        opt_thr = best_thr(y[m_te], p_te)
        opt_mcc = matthews_corrcoef(y[m_te], (p_te >= opt_thr).astype(int)) * 100
        auc = roc_auc_score(y[m_te], p_te)
        hon.append(honest_mcc); opt.append(opt_mcc); aucs.append(auc)
        print(f"  seed{s}: thr_val={thr_val:.2f} -> test MCC honest={honest_mcc:.1f} | "
              f"test-opt thr={opt_thr:.2f} MCC={opt_mcc:.1f} | AUC {auc:.3f}", flush=True)
    hon, opt, aucs = np.array(hon), np.array(opt), np.array(aucs)
    print(f"\nHONEST (threshold on val, test untouched): MCC {hon.mean():.1f}±{hon.std():.1f}  AUC {aucs.mean():.3f}")
    print(f"test-optimistic threshold (reference)      : MCC {opt.mean():.1f}±{opt.std():.1f}")
    print(f"threshold optimism = {opt.mean()-hon.mean():.1f} MCC")
    print(f"[ROC-AUC is threshold-free: {aucs.mean():.3f} — the honest separability either way]")


if __name__ == '__main__':
    main()
