"""Exp C on CPM — SAME harness as the BuST expC_relational.py, run on the CPM cache to test whether
the BuST 'GNN beats RF' result is a DATA property or a harness artifact. Reads a CPM rich cache
(X,EI,EW,Y,active,node_label,in_f); 'eng' = cache features (raw+rich, already z-scored); 'rel' = the
SAME 5 relational neighbour-consistency features computed from X[:,0:5]; same 7 modes, same vehicle-
disjoint paired protocol (stratified by node_label, default_rng(SEED), 70/30).

NOTE: cache kinematics are z-scored, so relational head_dev uses plain (non-circular) difference —
applied identically to rf_rel/gnn_rel/static_rel/gnn_edge, so the comparison stays valid.

Usage: python expC_cpm.py <cache.pt> <mode:rf_eng|rf_rel|gnn_rel|gnn_edge|static_eng|static_rel> <seed> [epochs] [window]
Prints: RESULT expCcpm mode=.. seed=.. mcc=.. auc=.. malf1=.. macrof1=.. in_f=..
"""
import os, sys, time
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (matthews_corrcoef, roc_auc_score, f1_score,
                             precision_recall_fscore_support)
torch.set_num_threads(2)
from torch_geometric_temporal.nn.recurrent import GConvGRU
from torch_geometric.nn import GATv2Conv, GCNConv

CACHE  = sys.argv[1]
MODE   = sys.argv[2] if len(sys.argv) > 2 else "gnn_edge"
SEED   = int(sys.argv[3]) if len(sys.argv) > 3 else 3
EPOCHS = int(sys.argv[4]) if len(sys.argv) > 4 else 40
WINDOW = int(sys.argv[5]) if len(sys.argv) > 5 else 24
HIDDEN, LR = 32, 0.01
need_rel = MODE in ("rf_rel", "gnn_rel", "gnn_edge", "static_rel")
need_edge = MODE == "gnn_edge"


def compute_relational(X, EI, EW, nstep, N):
    rel = np.zeros((nstep, N, 5), dtype=np.float32)
    edge_attrs = []
    for t in range(nstep):
        ei_t = EI[t].numpy()
        xt = X[t].numpy()
        src, dst = ei_t[0].astype(int), ei_t[1].astype(int)
        px, py, head, spd, acc = xt[:, 0], xt[:, 1], xt[:, 2], xt[:, 3], xt[:, 4]
        nm = src != dst
        s, d = src[nm], dst[nm]
        deg = np.bincount(d, minlength=N).astype(np.float32)
        degc = np.maximum(deg, 1.0)
        spd_nb = np.bincount(d, weights=spd[s], minlength=N) / degc
        acc_nb = np.bincount(d, weights=acc[s], minlength=N) / degc
        head_nb = np.bincount(d, weights=head[s], minlength=N) / degc
        cx = np.bincount(d, weights=px[s], minlength=N) / degc
        cy = np.bincount(d, weights=py[s], minlength=N) / degc
        spd_dev = spd - spd_nb
        acc_dev = acc - acc_nb
        head_dev = np.abs(head - head_nb)          # plain diff (kinematics already z-scored)
        pos_dev = np.sqrt((px - cx) ** 2 + (py - cy) ** 2)
        iso = deg == 0
        for arr in (spd_dev, acc_dev, head_dev, pos_dev):
            arr[iso] = 0.0
        rel[t] = np.stack([deg, spd_dev, acc_dev, head_dev, pos_dev], axis=1)
        if need_edge:
            ea = np.stack([np.abs(spd[src] - spd[dst]), np.abs(head[src] - head[dst]),
                           np.abs(acc[src] - acc[dst]),
                           EW[t].numpy().astype(np.float32)], axis=1).astype(np.float32)
            edge_attrs.append(ea)
    return rel, edge_attrs


def focal(lg, t, a, g=2.0):
    lp = F.log_softmax(lg, -1).gather(1, t.unsqueeze(1)).squeeze(1)
    return (((1 - lp.exp()) ** g) * (-lp) * a[t]).mean()


def cw(y):
    w = torch.ones(2)
    for c in (0, 1):
        n = int((y == c).sum())
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


def best_eval(y, p):
    bt, bm = 0.5, -1.0
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
        auc = float('nan')
    pr = precision_recall_fscore_support(y, pred, average=None, labels=[0, 1], zero_division=0)
    return bm * 100, auc, pr[2][1] * 100, f1_score(y, pred, average="macro", zero_division=0) * 100


class RecGNN(nn.Module):
    def __init__(self, in_f, hidden=HIDDEN, dropout=0.5):
        super().__init__()
        self.rec = GConvGRU(in_f, hidden, 2)
        self.lin1 = nn.Linear(hidden, hidden); self.classifier = nn.Linear(hidden, 2)
        self.dropout = dropout

    def forward(self, x, ei, ew, ea=None, H=None):
        H = self.rec(x, ei, ew, H)
        h = F.relu(self.lin1(H)); h = F.dropout(h, p=self.dropout, training=self.training)
        return self.classifier(h), H


class RelEdgeGNN(nn.Module):
    def __init__(self, in_f, edge_dim, hidden=HIDDEN, dropout=0.5):
        super().__init__()
        self.rec = GConvGRU(in_f, hidden, 2)
        self.gat = GATv2Conv(hidden, hidden, heads=2, concat=False, edge_dim=edge_dim, add_self_loops=False)
        self.lin1 = nn.Linear(hidden, hidden); self.classifier = nn.Linear(hidden, 2)
        self.dropout = dropout

    def forward(self, x, ei, ew, ea=None, H=None):
        H = self.rec(x, ei, ew, H)
        g = F.elu(self.gat(H, ei, edge_attr=ea))
        h = F.relu(self.lin1(g)); h = F.dropout(h, p=self.dropout, training=self.training)
        return self.classifier(h), H


class StaticGCN(nn.Module):
    def __init__(self, in_f, hidden=HIDDEN, dropout=0.5):
        super().__init__()
        self.c1 = GCNConv(in_f, hidden); self.c2 = GCNConv(hidden, hidden)
        self.c3 = GCNConv(hidden, hidden); self.classifier = nn.Linear(hidden, 2)
        self.dropout = dropout

    def forward(self, x, ei, ew, ea=None, H=None):
        h = F.relu(self.c1(x, ei, ew)); h = F.relu(self.c2(h, ei, ew))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = F.relu(self.c3(h, ei, ew))
        return self.classifier(h), None


def main():
    t0 = time.time()
    c = torch.load(CACHE)
    X, EI, EW, Y, active = c["X"], c["EI"], c["EW"], c["Y"], c["active"]
    node_label = c["node_label"].numpy()
    n_nodes, T, lags, in_f0 = c["n_nodes"], c["T"], c["lags"], c["in_f"]
    nstep = T - lags
    N = n_nodes

    actnp = [active[t].numpy().astype(bool) for t in range(nstep)]

    if need_rel:
        rel_raw, edge_attrs = compute_relational(X, EI, EW, nstep, N)
        mask = np.stack(actnp)                          # (nstep, N)
        flat = rel_raw[mask]
        mu, sd = flat.mean(0), flat.std(0); sd[sd == 0] = 1.0
        relz = (rel_raw - mu) / sd
        for t in range(nstep):
            relz[t][~actnp[t]] = 0.0
        feats = [torch.cat([X[t], torch.tensor(relz[t])], dim=1) for t in range(nstep)]
        in_f = in_f0 + 5
    else:
        feats = X
        in_f = in_f0

    if need_edge:
        allea = np.concatenate(edge_attrs, axis=0)
        em, es = allea.mean(0, keepdims=True), allea.std(0, keepdims=True); es[es == 0] = 1.0
        edge_attrs = [torch.tensor((ea - em) / es, dtype=torch.float) for ea in edge_attrs]

    rng = np.random.default_rng(SEED); trm = np.zeros(N, dtype=bool)
    for cc in (0, 1):
        idx = np.where(node_label == cc)[0]; rng.shuffle(idx)
        trm[idx[:int(0.7 * len(idx))]] = True
    n_train = int(0.7 * nstep)

    # ---------- RF modes ----------
    if MODE in ("rf_eng", "rf_rel"):
        Xc, yb, vid = [], [], []
        for t in range(nstep):
            m = actnp[t]
            Xc.append(feats[t].numpy()[m]); yb.append((Y[t].numpy()[m] != 0).astype(int)); vid.append(np.where(m)[0])
        Xc = np.concatenate(Xc); yb = np.concatenate(yb); vid = np.concatenate(vid)
        tr = trm[vid]; te = ~tr
        rf = RandomForestClassifier(n_estimators=300, max_features=0.5, n_jobs=4, random_state=SEED)
        rf.fit(Xc[tr], yb[tr], sample_weight=bal_sw(yb[tr]))
        mcc, auc, malf1, macrof1 = best_eval(yb[te], rf.predict_proba(Xc[te])[:, 1])
        print(f"RESULT expCcpm mode={MODE} seed={SEED} mcc={mcc:.2f} auc={auc:.4f} "
              f"malf1={malf1:.2f} macrof1={macrof1:.2f} in_f={in_f} ({(time.time()-t0)/60:.1f}min)", flush=True)
        return

    # ---------- GNN / static modes ----------
    trv = torch.tensor(trm); tev = torch.tensor(~trm)
    act = [active[t] for t in range(nstep)]

    torch.manual_seed(SEED)
    if MODE == "gnn_edge":
        model = RelEdgeGNN(in_f, edge_dim=edge_attrs[0].shape[1])
    elif MODE in ("static_eng", "static_rel"):
        model = StaticGCN(in_f)
    else:
        model = RecGNN(in_f)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    def ea_of(i):
        return edge_attrs[i] if need_edge else None

    def evaluate():
        model.eval(); P, L = [], []; H = None
        with torch.no_grad():
            for i in range(nstep):
                logits, H = model(feats[i], EI[i], EW[i], ea_of(i), H)
                m = act[i] & tev
                if m.sum() > 0:
                    P.append(torch.softmax(logits, 1)[m, 1].numpy()); L.append(Y[i][m].numpy())
        model.train()
        return best_eval(np.concatenate(L), np.concatenate(P))

    for ep in range(1, EPOCHS + 1):
        H = None
        for st in range(0, n_train, WINDOW):
            if H is not None:
                H = H.detach()
            opt.zero_grad(); wl = 0.0; cnt = 0
            for i in range(st, min(st + WINDOW, n_train)):
                logits, H = model(feats[i], EI[i], EW[i], ea_of(i), H)
                m = act[i] & trv
                if m.sum() == 0:
                    continue
                wl = wl + focal(logits[m], Y[i][m], cw(Y[i][m])); cnt += 1
            if cnt:
                (wl / cnt).backward(); opt.step()

    mcc, auc, malf1, macrof1 = evaluate()
    print(f"RESULT expCcpm mode={MODE} seed={SEED} mcc={mcc:.2f} auc={auc:.4f} "
          f"malf1={malf1:.2f} macrof1={macrof1:.2f} in_f={in_f} ({(time.time()-t0)/60:.1f}min)", flush=True)


if __name__ == "__main__":
    main()
