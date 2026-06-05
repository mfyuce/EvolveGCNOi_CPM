"""
smoke_test_models.py — API/shape/backward correctness check for every model in
models_cpm, on a tiny synthetic temporal graph. No real dataset needed.
Run locally before launching the full suite on maya.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import torch, torch.nn.functional as F
import numpy as np
import models_cpm

torch.manual_seed(0)
N, FIN, T = 50, 10, 12      # nodes, features, timesteps
n_edges = 120

# synthetic temporal graph
X, EI, EW, Y, ACT = [], [], [], [], []
for t in range(T):
    X.append(torch.randn(N, FIN))
    ei = torch.randint(0, N, (2, n_edges))
    EI.append(ei)
    EW.append(torch.rand(n_edges))
    y = (torch.rand(N) < 0.1).long()        # ~10% positive
    Y.append(y)
    ACT.append(torch.ones(N, dtype=torch.bool))

WINDOW = 6


def focal(lg, t, g=2.0):
    lp = F.log_softmax(lg, -1).gather(1, t.unsqueeze(1)).squeeze(1)
    return (((1 - lp.exp()) ** g) * (-lp)).mean()


def run_one(name):
    model = models_cpm.build(name, N, FIN, h=16, proj=16)
    kind = model.kind
    has_reset = hasattr(model, "reset")
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    # one training "epoch" with truncated BPTT
    model.train(); H = None
    if has_reset:
        model.reset()
    for st in range(0, T, WINDOW):
        if kind == "node_rec" and H is not None:
            H = H.detach()
        if has_reset:
            model.reset()
        opt.zero_grad(); loss = 0.0; c = 0
        for i in range(st, min(st + WINDOW, T)):
            if kind == "node_rec":
                lg, H = model(X[i], EI[i], EW[i], H)
            else:
                lg = model(X[i], EI[i], EW[i])
            assert lg.shape == (N, 2), f"{name}: bad logit shape {lg.shape}"
            loss = loss + focal(lg, Y[i]); c += 1
        (loss / c).backward(); opt.step()
    # eval pass
    model.eval(); H = None
    if has_reset:
        model.reset()
    with torch.no_grad():
        for i in range(T):
            if kind == "node_rec":
                lg, H = model(X[i], EI[i], EW[i], H)
            else:
                lg = model(X[i], EI[i], EW[i])
            p = torch.softmax(lg, 1)[:, 1]
            assert p.shape == (N,)
    n_params = sum(p.numel() for p in model.parameters())
    return kind, n_params


print(f"{'model':22s} {'kind':10s} {'params':>10s}  status")
print("-" * 55)
ok, fail = 0, 0
for name in models_cpm.ALL_MODELS:
    try:
        kind, np_ = run_one(name)
        print(f"{name:22s} {kind:10s} {np_:>10,}  OK")
        ok += 1
    except Exception as e:
        print(f"{name:22s} {'?':10s} {'-':>10s}  FAIL: {type(e).__name__}: {e}")
        fail += 1

print("-" * 55)
print(f"{ok} OK, {fail} FAIL")
