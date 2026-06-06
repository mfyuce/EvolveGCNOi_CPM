"""
diagnose_cpm_difficulty.py — why is CPM harder than BuST-ADMA?

Model-agnostic (RandomForest) vehicle-disjoint separability probe on a prep_cache
.pt. The vehicle-disjoint split (stratified by ever-attacker, 70/30) is IDENTICAL
to run_zoo_cpm.py, so the RF numbers are directly comparable to the GNN ones.

Reports, per cache:
  * class balance (malicious %)
  * RF with ALL features  -> MCC / ROC-AUC / malicious-F1  (data separability ceiling)
  * RF without position [x,y] -> MCC drop   (is absolute position the crutch?)
  * RF position-only [x,y]    -> MCC         (how much position alone carries)
  * permutation feature importance (MCC drop when each feature is shuffled)

Feature layout (generate_gnn_datasets_v2.py):
  0 x  1 y  2 heading  3 speed  4 accel  5 sensor_range  6 visibility
  7 sensor_type  8 weather  [9 mean_det_conf (v2)]  [+8 aug consistency feats]
"""
import argparse, time
import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import matthews_corrcoef, roc_auc_score, f1_score

POS = [0, 1]  # x_detected, y_detected (UTM)
BASE = ['x', 'y', 'heading', 'speed', 'accel', 'sensor_range',
        'visibility', 'sensor_type', 'weather']
AUG = ['|dx|', '|dy|', '|dhead|', '|dspeed|', '|daccel|', 'pos_jump', 'kin_resid', 'in_degree']


def feat_names(F):
    if F == 9:  return BASE
    if F == 10: return BASE + ['mean_det_conf']
    if F == 17: return BASE + AUG
    if F == 18: return BASE + ['mean_det_conf'] + AUG
    return [f'f{i}' for i in range(F)]


def build_xy(cache):
    c = torch.load(cache)
    X, Y, active = c['X'], c['Y'], c['active']
    node_label = c['node_label'].numpy()
    n_nodes, T, lags, in_f = c['n_nodes'], c['T'], c['lags'], c['in_f']
    feats, labs, vids = [], [], []
    for t in range(T - lags):
        m = active[t].numpy().astype(bool)
        if m.sum() == 0:
            continue
        feats.append(X[t].numpy()[m])
        labs.append(Y[t].numpy()[m])
        vids.append(np.where(m)[0])           # node (vehicle) ids
    return (np.concatenate(feats), (np.concatenate(labs) != 0).astype(int),
            np.concatenate(vids), node_label, n_nodes, in_f)


def disjoint_train_mask(node_label, n_nodes, seed):
    rng = np.random.default_rng(seed); trm = np.zeros(n_nodes, dtype=bool)
    for c in (0, 1):
        idx = np.where(node_label == c)[0]; rng.shuffle(idx)
        trm[idx[:int(0.7 * len(idx))]] = True
    return trm                                 # True = train vehicle


def best_mcc(y, p):
    bt, bm = 0.5, -1.0
    for thr in np.linspace(0.05, 0.95, 19):
        m = matthews_corrcoef(y, (p >= thr).astype(int))
        if m > bm:
            bm, bt = m, thr
    return bm, bt


def rf_fit_eval(Xtr, ytr, Xte, yte, cols, seed):
    rf = RandomForestClassifier(n_estimators=200, n_jobs=8,
                                class_weight='balanced', random_state=seed)
    rf.fit(Xtr[:, cols], ytr)
    p = rf.predict_proba(Xte[:, cols])[:, 1]
    mcc, thr = best_mcc(yte, p)
    try:
        auc = roc_auc_score(yte, p)
    except Exception:
        auc = float('nan')
    malf1 = f1_score(yte, (p >= thr).astype(int), pos_label=1, zero_division=0)
    return rf, mcc, auc, malf1, p, thr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', required=True)
    ap.add_argument('--seeds', default='1,2,3')
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(',')]
    t0 = time.time()

    X, y, v, node_label, n_nodes, in_f = build_xy(a.cache)
    names = feat_names(in_f)
    allcols = list(range(in_f)); nopos = [i for i in allcols if i not in POS]
    print(f"=== cache={a.cache}  F={in_f}  node-step samples={len(y):,}  "
          f"({time.time()-t0:.0f}s) ===")
    print(f"class balance: malicious = {y.mean()*100:.2f}%  ({y.sum():,}/{len(y):,})")

    m_all, m_np, m_po, aucs, malf1s = [], [], [], [], []
    for s in seeds:
        trm = disjoint_train_mask(node_label, n_nodes, s)
        tr = trm[v]; te = ~tr
        Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]
        _, ma, auc, mf1, _, _ = rf_fit_eval(Xtr, ytr, Xte, yte, allcols, s)
        _, mn, _,  _,   _, _ = rf_fit_eval(Xtr, ytr, Xte, yte, nopos, s)
        _, mp, _,  _,   _, _ = rf_fit_eval(Xtr, ytr, Xte, yte, POS, s)
        m_all.append(ma); m_np.append(mn); m_po.append(mp); aucs.append(auc); malf1s.append(mf1)
        print(f" seed{s}: train={tr.sum():,} test={te.sum():,} | "
              f"MCC all={ma*100:5.1f} (AUC {auc:.3f}, malF1 {mf1*100:4.1f}) | "
              f"no-pos={mn*100:5.1f} | pos-only={mp*100:5.1f}")

    print(f"\n-- {a.cache}  mean over {len(seeds)} seeds --")
    print(f"RF all features : MCC {np.mean(m_all)*100:5.1f}   ROC-AUC {np.mean(aucs):.3f}   "
          f"malicious-F1 {np.mean(malf1s)*100:.1f}")
    print(f"RF NO position  : MCC {np.mean(m_np)*100:5.1f}   (drop {(np.mean(m_all)-np.mean(m_np))*100:+.1f})")
    print(f"RF position only: MCC {np.mean(m_po)*100:5.1f}")

    # permutation importance on the last seed
    trm = disjoint_train_mask(node_label, n_nodes, seeds[-1]); tr = trm[v]; te = ~tr
    rf, _, _, _, p0, thr = rf_fit_eval(X[tr], y[tr], X[te], y[te], allcols, seeds[-1])
    yte = y[te]; base = matthews_corrcoef(yte, (p0 >= thr).astype(int))
    Xte = X[te].copy(); rng = np.random.default_rng(0); imp = []
    for j in range(in_f):
        col = Xte[:, j].copy(); rng.shuffle(Xte[:, j])
        pj = rf.predict_proba(Xte)[:, 1]
        imp.append(base - matthews_corrcoef(yte, (pj >= thr).astype(int)))
        Xte[:, j] = col
    print(f"\npermutation importance (MCC drop x100 when feature shuffled, seed {seeds[-1]}):")
    for j in np.argsort(imp)[::-1]:
        bar = '#' * max(0, int(imp[j] * 200))
        print(f"  {names[j]:14s} {imp[j]*100:+5.1f}  {bar}")
    print(f"\n[BuST-ADMA ref: vd MCC ~56 raw / ~75 eng; abs-x is top feature, ~31 MCC drop if removed]")


if __name__ == '__main__':
    main()
