"""
build_features_cpm_rich.py — append RICH physics-consistency features to a CPM cache,
so the node-state GNN (gconvgru) can be fed the same signals that lift a tabular model
to ~0.65 MCC honestly (see push_cpm_honest.py).

Appends 11 features (z-scored over active node-steps), all derived from REPORTED
kinematics vs the node's previous active appearance (no label use):
  dx, dy, disp(=pos_jump),
  speed_resid = |disp - speed|        physics: displacement vs reported speed
  |dspeed|, dheading(circular), |daccel|,
  accel_resid = |Δspeed - accel|      physics: Δspeed vs reported acceleration
  in_degree, |d_in_degree|,           structural: #sensors seeing it (+ change)
  speed                               raw speed for context
in_f -> in_f + 11.  Output: <cache>_rich.pt (same structure as build_features_cpm.py).

Usage
-----
  python build_features_cpm_rich.py --cache data/cache_v1.pt    --out data/cache_v1_rich.pt
  python build_features_cpm_rich.py --cache data/v2/cache_v2.pt --out data/v2/cache_v2_rich.pt
"""
import argparse, time
import numpy as np
import torch

IX, IY, IH, IS, IA = 0, 1, 2, 3, 4
EXTRA = 11


def circ_abs_deg(a, b):
    return torch.abs((a - b + 180.0) % 360.0 - 180.0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    t0 = time.time()
    c = torch.load(args.cache)
    X, EI, active = c["X"], c["EI"], c["active"]
    n_nodes, T, lags, in_f = c["n_nodes"], c["T"], c["lags"], c["in_f"]
    n_steps = T - lags
    print(f"loaded {args.cache}: T={T} N={n_nodes} F={in_f} ({time.time()-t0:.0f}s)", flush=True)

    rich = [torch.zeros(n_nodes, EXTRA, dtype=torch.float32) for _ in range(n_steps)]
    last = torch.zeros(n_nodes, 5)          # last active [x,y,heading,speed,accel]
    last_indeg = torch.zeros(n_nodes)
    seen = torch.zeros(n_nodes, dtype=torch.bool)

    for t in range(n_steps):
        x = X[t]; act = active[t]
        ei = EI[t]
        indeg = torch.zeros(n_nodes)
        if ei.numel():
            src, dst = ei[0], ei[1]; prox = src != dst
            if prox.any():
                indeg.scatter_add_(0, dst[prox], torch.ones(int(prox.sum())))
        dx = (x[:, IX] - last[:, 0]).abs(); dy = (x[:, IY] - last[:, 1]).abs()
        disp = torch.sqrt(dx ** 2 + dy ** 2)
        speed_resid = (disp - x[:, IS]).abs()
        dspeed = x[:, IS] - last[:, 3]
        dheading = circ_abs_deg(x[:, IH], last[:, 2])
        daccel = (x[:, IA] - last[:, 4]).abs()
        accel_resid = (dspeed - x[:, IA]).abs()
        d_indeg = (indeg - last_indeg).abs()
        for arr in (dx, dy, disp, speed_resid, dheading, daccel, accel_resid, d_indeg):
            arr[~seen] = 0.0
        dspeed_abs = dspeed.abs(); dspeed_abs[~seen] = 0.0

        feat = torch.stack([dx, dy, disp, speed_resid, dspeed_abs, dheading, daccel,
                            accel_resid, indeg, d_indeg, x[:, IS]], dim=1)   # (N,11)
        feat[~act] = 0.0
        rich[t] = feat

        upd = act
        last[upd, 0] = x[upd, IX]; last[upd, 1] = x[upd, IY]; last[upd, 2] = x[upd, IH]
        last[upd, 3] = x[upd, IS]; last[upd, 4] = x[upd, IA]
        last_indeg[upd] = indeg[upd]; seen[upd] = True

    # z-score appended features over all active node-steps
    stacked = torch.cat([rich[t][active[t]] for t in range(n_steps)], dim=0)
    mu = stacked.mean(0); sd = stacked.std(0); sd[sd == 0] = 1.0
    print(f"rich-feature means: {np.round(mu.numpy(),3)}", flush=True)

    X_rich = []
    for t in range(n_steps):
        a = (rich[t] - mu) / sd
        a[~active[t]] = 0.0
        X_rich.append(torch.cat([X[t], a], dim=1))   # (N, in_f+11)

    c_out = dict(c)
    c_out["X"] = X_rich
    c_out["in_f"] = in_f + EXTRA
    torch.save(c_out, args.out)
    print(f"saved {args.out}  in_f {in_f}->{in_f+EXTRA}  ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
