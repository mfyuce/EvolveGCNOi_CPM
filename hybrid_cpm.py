"""
hybrid_cpm.py — GNN-embedding -> gradient-boosting hybrid (single seed).

Train gconvgru (node-state) on the cache (train vehicles, train-time), extract its
per-node temporal hidden state H (+ its own malicious prob) as features, then train
HGB / RF on [H, gnn_prob, cache features] with the vehicle-disjoint split. Tests if
the GNN's temporal context + boosting's feature exploitation together reach ~0.61,
keeping a GNN component in the pipeline. Honest: GNN trained only on train vehicles,
embeddings for test vehicles are forward passes (no label use).

Prints: RESULT hybrid seed=.. gnn_mcc=.. hgb_mcc=.. hgb_auc=.. hgb_malf1=.. rf_mcc=.. ..
"""
import argparse, time
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (matthews_corrcoef, roc_auc_score,
                             precision_recall_fscore_support)
import models_cpm

torch.set_num_threads(4)


def focal(lg, t, a, g=2.0):
    lp = F.log_softmax(lg, -1).gather(1, t.unsqueeze(1)).squeeze(1)
    return (((1 - lp.exp()) ** g) * (-lp) * a[t]).mean()


def class_weights(y):
    w = torch.ones(2)
    for c in (0, 1):
        n = (y == c).sum().item()
        if n > 0:
            w[c] = len(y) / (2.0 * n)
    return w


def bal_sw(y):
    w = np.ones(len(y))
    for c in (0, 1):
        n = (y == c).sum()
        if n > 0:
            w[y == c] = len(y) / (2.0 * n)
    return w


def bmcc(y, p):
    bt, bm = 0.5, -1.0
    for thr in np.linspace(0.05, 0.95, 19):
        m = matthews_corrcoef(y, (p >= thr).astype(int))
        if m > bm:
            bm, bt = m, thr
    return bm, bt


def metrics(y, p):
    mcc, thr = bmcc(y, p)
    try:
        auc = roc_auc_score(y, p)
    except Exception:
        auc = float('nan')
    pr = precision_recall_fscore_support(y, (p >= thr).astype(int), average=None,
                                         labels=[0, 1], zero_division=0)
    return mcc * 100, auc, pr[2][1] * 100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', required=True)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--window', type=int, default=24)
    ap.add_argument('--hidden', type=int, default=32)
    a = ap.parse_args()
    t0 = time.time()
    c = torch.load(a.cache)
    X, EI, EW, Y, active = c['X'], c['EI'], c['EW'], c['Y'], c['active']
    node_label = c['node_label'].numpy()
    n_nodes, T, lags, in_f = c['n_nodes'], c['T'], c['lags'], c['in_f']
    nstep = T - lags

    # fixed-order active node-step feature matrix / labels / vehicle ids
    Xc, yb, vid = [], [], []
    for t in range(nstep):
        m = active[t].numpy().astype(bool)
        Xc.append(X[t].numpy()[m]); yb.append((Y[t].numpy()[m] != 0).astype(int))
        vid.append(np.where(m)[0])
    Xc = np.concatenate(Xc); yb = np.concatenate(yb); vid = np.concatenate(vid)

    rng = np.random.default_rng(a.seed); trm = np.zeros(n_nodes, dtype=bool)
    for cc in (0, 1):
        idx = np.where(node_label == cc)[0]; rng.shuffle(idx)
        trm[idx[:int(0.7 * len(idx))]] = True
    trv = torch.tensor(trm); n_tr = int(0.7 * nstep)

    torch.manual_seed(a.seed)
    model = models_cpm.build('gconvgru', n_nodes, in_f, h=a.hidden, d=0.5)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    for ep in range(a.epochs):
        model.train(); H = None
        for st in range(0, n_tr, a.window):
            if H is not None:
                H = H.detach()
            opt.zero_grad(); wl = 0.0; cnt = 0
            for i in range(st, min(st + a.window, n_tr)):
                mm = active[i] & trv
                lg, H = model(X[i], EI[i], EW[i], H)
                if mm.sum() == 0:
                    continue
                wl = wl + focal(lg[mm], Y[i][mm], class_weights(Y[i][mm])); cnt += 1
            if cnt:
                (wl / cnt).backward(); opt.step()

    # extract temporal hidden state H + GNN prob for ALL node-steps
    model.eval(); embs, probs, H = [], [], None
    with torch.no_grad():
        for t in range(nstep):
            Hn = model.rec(X[t], EI[t], EW[t], H)
            z = F.relu(model.l1(Hn)); lg = model.cl(z)
            H = Hn
            m = active[t].numpy().astype(bool)
            embs.append(Hn.numpy()[m]); probs.append(torch.softmax(lg, 1)[:, 1].numpy()[m])
    EMB = np.concatenate(embs); PB = np.concatenate(probs)
    Xhy = np.concatenate([EMB, PB[:, None], Xc], axis=1)

    tr = trm[vid]; te = ~tr
    gnn_mcc, _ = bmcc(yb[te], PB[te])
    hgb = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.06,
                                         max_leaf_nodes=63, l2_regularization=1.0,
                                         random_state=a.seed)
    rf = RandomForestClassifier(n_estimators=150, n_jobs=8, random_state=a.seed)
    out = {}
    for name, mk in (('hgb', hgb), ('rf', rf)):
        mk.fit(Xhy[tr], yb[tr], sample_weight=bal_sw(yb[tr]))
        out[name] = metrics(yb[te], mk.predict_proba(Xhy[te])[:, 1])
    print(f"RESULT hybrid seed={a.seed} gnn_mcc={gnn_mcc*100:.2f} "
          f"hgb_mcc={out['hgb'][0]:.2f} hgb_auc={out['hgb'][1]:.4f} hgb_malf1={out['hgb'][2]:.2f} "
          f"rf_mcc={out['rf'][0]:.2f} rf_auc={out['rf'][1]:.4f} rf_malf1={out['rf'][2]:.2f} "
          f"feat={Xhy.shape[1]} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == '__main__':
    main()
